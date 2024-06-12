import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, ceil
import numpy as np

import librosa
import numpy as np

"""
The original cardiogan implementation (in tensorflow) uses 2D convolutions, so the size of each 
item is (batch_size x 1 x 1 x sample_len(512)). 
I'd previously been using 1D convolutions (in models/cardiogan.py) so that my datasets could be (batch_size x 1 x 512)
This seems to be a problem (particularly the LayerNorms) -- found this:
https://github.com/YouHojoon/CardioGAN/blob/master/CardioGAN.ipynb
So this 'yh' version assumes data is 4D: batch_size x 1 x 1 x 512, and has shortcuts for the 2D convs etc
"""

class TorchLayerNorm(nn.Module):
    """
    There's this ongoing issue with keras v torch -- keras's channel-last format makes it easy for them to do LayerNorms on the channel dimension,
    which seems like what the AttnUNet/Cardiogan people do. 
    Pytorch's LayerNorm only leets you norm the last <n> dimensions, so either W, or H+W, or C+H+W. Can't do solely along the channel dim with one mean/std per channel. (If you do C+H+W it gives you a mean/std for each element.)
    Ongoing discussion: https://github.com/pytorch/pytorch/issues/71465
    Found this implementation in BERT, which seems to make sense: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/layer_norm.py
    But have to permute back and forth.
    NOTE nn.Parameters dont show up in torchsummary.summary, just FYI
    """

    def __init__(self, in_shape, eps=1e-6):
        super(TorchLayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(in_shape))
        self.b_2 = nn.Parameter(torch.zeros(in_shape))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # to channel last
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        res = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return res.permute(0, 3, 1, 2)


def Conv1d(in_channels, out_channels, kernel_size, padding=0, stride=1, bias=False):
    """ based on this: https://github.com/pritamqu/ppg2ecg-cardiogan/blob/d7239b0004a138101bdf597961be4d0af8ea2d5e/codes/layers.py#L14
        padding='same' doesnt exist for pytorch (when stride > 1), but for the sizes to work out, it needs to be 7
        padding=(0,7) just means 0-padding in the singleton dimension and 7-zeropadding at the sample dimension. 
        I assume pytorch splits up the 7 to either end.
    """
    if isinstance(kernel_size, int):  kernel_size=(1, kernel_size)
    if isinstance(stride, int):  stride=(1, stride)
    if isinstance(padding, int):  padding=(0,padding)
        
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride ,bias=bias)
    return layer

def DeConv1d(in_channels, out_channels, kernel_size, padding=0, stride=1, bias=False):
    layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1,kernel_size), padding = (0, padding), stride=(1,stride),bias=bias)
    return layer

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.layer_x = Conv1d(in_channels, in_channels, kernel_size=1, padding=0, stride=1, bias=True)  # "lin transforms with bias"
        self.layer_g = Conv1d(in_channels, in_channels, kernel_size=1, padding=0, stride=1, bias=True)
        self.out_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True), # "next channel-wise 1x1 convolutions"
            nn.Sigmoid()  # "followed by passing through a sigmoid activation in order to obtain the attention weights in [0,1]"
        )

    def forward(self,conn_out, curr_out):
        ' conn_out is x, which is a down layer stored in connections. curr_out from an up layer '
        x = self.layer_x(conn_out)
        g = self.layer_g(curr_out)

        out = torch.add(x,g)
        out = self.out_layer(out)
        out = torch.mul(out,conn_out)

        return out


def downsample(in_channels, out_channels, norm_dim=None, kernel_size = 16, stride=2, padding=7):
    layer = []
    layer.append(Conv1d(in_channels, out_channels,kernel_size, padding=padding,stride=stride))
    if norm_dim != None:
        # hack for now
        if isinstance(kernel_size, int):
            norm_shape = (out_channels, 1, norm_dim)
        else:
            norm_shape = (out_channels, norm_dim, norm_dim)  # tried nn.LayerNorm with norm_shape, with norm_dim.. didn't work
        layer.append(TorchLayerNorm(out_channels))

    layer.append(nn.LeakyReLU(0.2))

    return nn.Sequential(*layer)

def upsample(in_channels, out_channels, norm_dim=None, padding=7,kernel_size=16,stride=2):
    layer = []
    layer.append(DeConv1d(in_channels, out_channels, kernel_size, padding=padding,stride=stride))
    if norm_dim != None:
        if isinstance(kernel_size, int):
            norm_shape = (out_channels, 1, norm_dim)
        else:
            norm_shape = (out_channels, norm_dim, norm_dim)
        layer.append(TorchLayerNorm(out_channels))
    layer.append(nn.ReLU())

    return nn.Sequential(*layer)

class Generator(nn.Module):

    INPUT_NDIM = 4

    def __init__(self, IN_CHANNELS=1, OUT_CHANNELS=1):
        super(Generator,self).__init__()
        self.IN_CHANNELS = IN_CHANNELS   # unused, 
        self.OUT_CHANNELS = OUT_CHANNELS  # just for debugging
        #down (params are in_channel, out_channel, out_width) in NCWH format
        self.down_layer1 = downsample(IN_CHANNELS,64,None)  # [1, 1, 512] -> [64, 1, 256]
        self.down_layer2 = downsample(64,128,128)  # 64, 1, 256] --> 128, 1, 128]
        self.down_layer3 = downsample(128,256,64)  # 128, 1, 128] --> 256, 1, 64]
        self.down_layer4 = downsample(256,512,32)  # 256, 1, 64] --> 512, 1, 32]
        self.down_layer5 = downsample(512,512,16)  # 512, 1, 32] --> 512, 1, 16]
        self.down_layer6 = downsample(512,512,8)  # 512, 1, 16] --> 512, 1, 8]

        #attention
        self.attention_layer1 = AttentionBlock(512)  # in_chan 512 needs to be out_chan of down_6 as well as up_1
        self.attention_layer2 = AttentionBlock(512)  # down_5 and up_2
        self.attention_layer3 = AttentionBlock(512)  # down_4 and up_3
        self.attention_layer4 = AttentionBlock(256)  # down_3 and up_4
        self.attention_layer5 = AttentionBlock(128)   # down_2 as well as up_5
        self.attention_layer6 = AttentionBlock(64) 

        #up
        self.up_layer1 = upsample(512,512,kernel_size=16,padding=8,stride=1,norm_dim=8)  # the closest we can get # Goodfellow says no norm? 
        #self.up_layer1 = upsample(512,512,kernel_size=15,padding=7,stride=1,norm_dim=8)  # the closest we can get, see note below when this is called in forward
        self.up_layer2 = upsample(512,512,16)
        self.up_layer3 = upsample(512,512,32)
        self.up_layer4 = upsample(512,256,64)
        self.up_layer5 = upsample(256,128,128)
        self.up_layer6 = upsample(128,64,256)

        self.out_layer = nn.Sequential(
            DeConv1d(64,OUT_CHANNELS,16,7,2,bias=True),
            nn.Tanh()
        )

    def get_bottleneck(self,x):
        out = x

        #down
        for i in range(6):
            layer_name = 'down_layer' + str(i+1)
            layer = getattr(self,layer_name)
            out = layer(out)

        # just going with my gut here... hopefully 1x1x8 is enough to have enough info... think so cause we dont actually have that many subjects
        last_conv = downsample(512, 1, None, kernel_size=8).to(out.device)  # conv to get it down to bsx1x1xn
        out = last_conv(out)
        out = out.contiguous().view(-1,  8) # Flatten the third dimension
        return out  # this ends as bs x 8, with InstanceNorm on the 8


    def forward(self,x):
        connection = []
        out = x

        #down
        for i in range(6):
            layer_name = 'down_layer' + str(i+1)
            layer = getattr(self,layer_name)
            #print(out.shape, end='  --> ')
            #print(layer_name, end='  --> ')
            out = layer(out)
            #print(out.shape)
            connection.append(out)

   
        # Extra outer padding here: keras='same' does magic to make the output dim the same as the input, but 
        # not easy to do this in pytorch. output ends as (512, 1, 7) instead of (512, 1, 8), so needed the pad
        # But that's adding a 0 to everything, which isn't great
        # The way to do this is to use torch's 'outer_padding' input, but that's not working
        # Easiest thing to do now is make pading=7,kernel=15 or padding=8,kernel=17. Think the smaller kernel/pad
        # But it looks like all keras does (and torch with 'outer_padding' is just pad (append) 
        out = F.pad(out,(0,1))  
        #up
        for i in range(6):
            layer_name = 'up_layer'+str(i+1)
            att_layer_name = 'attention_layer'+str(i+1)
            layer = getattr(self,layer_name)
            att_layer = getattr(self,att_layer_name)

            out = layer(out)  # "x"
            att = att_layer(connection[5-i],out)  # "g"
            out = torch.add(out,att)

        out = self.out_layer(out)

        return out



class IdentityEncoder(nn.Module):
    ''' For completeion, but unused. '''
    def __init__(self, IN_CHANNELS=1):
        super().__init__()

        # Let's just do the same as the time discriminator
        self.layer1 = downsample(IN_CHANNELS, 64, None)
        self.layer2 = downsample(64,128,128)
        self.layer3 = downsample(128,256,64)
        self.layer4 = downsample(256,512,32)
        self.flatten = nn.Flatten()  # starts at 1 by default
        self.fc1 = nn.Linear(512*32, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.sig = nn.Sigmoid()
           
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x



class TimeDiscriminator(nn.Module):

    def __init__(self, IN_CHANNELS, final_sigmoid=False, out_1d=False):
        super().__init__()

        self.final_sigmoid = final_sigmoid
        if out_1d:
            self.OUTPUT_SHAPE = (1, 1)
        else:
            self.OUTPUT_SHAPE = (1, 16)

        "  4 convolution layers, where the number of filterss are gradually increased (64, 128, 256, 512 with a fixed kernel of 1x16 "
        " stride 2 "
        self.layer1 = downsample(IN_CHANNELS,64,None)
        self.layer2 = downsample(64,128,128)
        self.layer3 = downsample(128,256,64)
        self.layer4 = downsample(256,512,32)
        " finally, the output is obtained from a single-channel convolutional layer "
        if out_1d:
            self.out_layer = Conv1d(512,1,kernel_size=46,padding=7,stride=2)  # 46 comes from the desire for this to be 1
        else:
            self.out_layer = Conv1d(512,1,kernel_size=16,padding=7,stride=2)  # same as the other layers; this makes this 1x16
        
        self.sigmoid = nn.Sigmoid()  # optional 

    def get_features(self, x):
        # Penultimate layer, seems that that's what people usually do
        out = self.layer1(x)
        out = self.layer2(out)  # [bs, 64, 1, 256] --> [bs, 128, 1, 128]
        out = self.layer3(out)  # [bs, 128, 1, 128] --> [bs, 256, 1, 64]
        out = self.layer4(out)  # [bs, 256, 1, 64] --> [bs, 512, 1, 32]
        out = self.out_layer(out)  # [bs, 512, 1, 32] -> [bs, 1, 1, 16]
        return out

    def forward(self,x):
        out = self.layer1(x)  # [batch_size, 1, 1, 512] --> [bs, 64, 1, 256]
        out = self.layer2(out)  # [bs, 64, 1, 256] --> [bs, 128, 1, 128]
        out = self.layer3(out)  # [bs, 128, 1, 128] --> [bs, 256, 1, 64]
        out = self.layer4(out)  # [bs, 256, 1, 64] --> [bs, 512, 1, 32]
        out = self.out_layer(out)  # [bs, 512, 1, 32] -> [bs, 1, 1, 16] or [bs, 1, 1, 1] if self.out_1d
        if self.final_sigmoid:
            out = self.sigmoid(out)
        return out



class FrequencyDiscriminator(nn.Module):


    def __init__(self, IN_CHANNELS, final_sigmoid=False, out_1d=False):
        super().__init__()
        self.IN_CHANNELS = IN_CHANNELS  # unused, just for debugging
        self.final_sigmoid = final_sigmoid
        if out_1d:
            self.OUTPUT_SHAPE = (1, 1)
        else:
            self.OUTPUT_SHAPE = (4, 4)

        self.layer1 = downsample(IN_CHANNELS, 64, kernel_size=(7,7), norm_dim=None, stride=(2,2),padding=(3,3))  # 
        self.layer2 = downsample(64, 128, kernel_size=(7,7), norm_dim=32, stride=(2,2), padding=(3,3))
        self.layer3 = downsample(128, 256, kernel_size=(7,7), norm_dim=16, stride=(2,2), padding=(3,3))
        self.layer4 = downsample(256, 512, kernel_size=(7,7), norm_dim=8, stride=(2,2), padding=(3,3))

        if out_1d:
            # kernel_size 14 comes from the desire for the output to be single-dimensional
            self.out_layer = nn.Conv2d(512,IN_CHANNELS,(14,14),stride=2,padding=3,bias=False)  # BCEWithLogitsLoss does the sigmoid
        else:
            self.out_layer = nn.Conv2d(512,IN_CHANNELS,(7,7),stride=2,padding=3,bias=False)  # same as the others, ends up being 4x4
       
        self.sigmoid = nn.Sigmoid() 

        """ Do this in main 
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight.data,0.0,0.02) 
        """
    
    def forward(self,x):
        out = self.stft(x)  # [bs, 1, 1, 512] --> [bs, 1, 128, 128]  (because the stft has real and complex)

        out = self.layer1(out)  # [bs, 1, 128, 128] --> [bs, 64, 64, 64]
        out = self.layer2(out)  # [bs, 64, 64, 64] --> [bs, 128, 32, 32]
        out = self.layer3(out)  # [bs, 128, 32, 32] --> [bs, 256, 16, 16]
        out = self.layer4(out)  # [bs, 256, 16, 16] --> [bs, 512, 8, 8]
        out = self.out_layer(out)  # [bs, 512, 8, 8] --> [bs, 1, 4, 4] or [bs, 1, 1, 1] if self.out_1d
        if self.final_sigmoid:
            out = self.sigmoid(out)

        return out


    def stft(self, batch):
        """ 
        sftf for the freq-based discriminator. 
        NOTE here that torchaudio.stft isn't actually the same as librosa.stft, even though it would be much quicker
        """
        device = batch.device
        dtype = batch.dtype
        signal_np = batch.cpu().detach()  # I guess that since this is the first step it doesnt need backward
        signal_np=np.array(signal_np, dtype='float64')
        freq_list = []
            
        for signal in signal_np:
            lead_list = []
            for lead in signal:  # up to 3 
                freq = librosa.stft(lead,n_fft=255,hop_length=4,center=True)
                freq = np.abs(freq)
                freq = np.log(freq+1e-10)
                lead_list.append(freq)
            freq_list.append(np.concatenate(lead_list))

        freq_batch = torch.from_numpy(np.stack(freq_list))
        freq_batch = freq_batch.type(dtype).to(device)

        return freq_batch


import torch
import torch.nn as nn
from math import floor


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.01)

class DeepECGModel(nn.Module):

    def _calculate_template_dim(self, input_dim):
        # For input to flatten, changes cause each dataset len and fs is different
        
        def conv_calc(in_w, k, p, s):
            out_w = ( (in_w-k+2*p)/s ) + 1
            return floor(out_w)

        w = conv_calc(input_dim, k=3, p=0, s=2)
        w = conv_calc(w, k=3, p=0, s=2)
        w = conv_calc(w, k=3, p=0, s=2)
        return w * 256
        

        # bidmc is 1048, capno is 19200 somehow but okay. maybe see what the fs of the datasets they used were 

    def __init__(self, input_dim=200, output_dim=256):
        """
        Six convolution layers that use ReLus, 3 max pooling, 3 LRNs, 1 dropout,
        a FC, and a Softmax 

        Output_dim is the number of patients

        template_len is the size of the template after going through all the conv blocks. For 15sec samples for bidmc, ends up being 2048
        need this as input because it's input to flatten() at end before preds

        """
        super().__init__()

        # Input 200x1

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        ) 

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding='same'),
            #nn.LocalResponseNorm(size=5, alpha=2e-4, beta=0.75, k=1), # more trouble than worth, just use batchnorm
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding='same'),
            #nn.LocalResponseNorm(size=5, alpha=2e-4, beta=0.75, k=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding='same'),
            nn.ReLU()
        )

        self.block6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7, stride=1, padding='same'),
            nn.ReLU(),
            #nn.LocalResponseNorm(size=5, alpha=2e-4, beta=0.75, k=1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5)
        )
      
        self.flatten = nn.Flatten() 
        in_features = self._calculate_template_dim(input_dim)   # should be 19200 for capno, 2048 for bidmc
        self.fc = nn.Linear(in_features=in_features, out_features=output_dim)  # verification # TODO figure out what this is in terms of num_inputs
        self.softmax = nn.Softmax(dim=1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def get_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x

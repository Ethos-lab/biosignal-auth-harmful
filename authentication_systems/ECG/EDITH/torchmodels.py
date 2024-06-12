import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.01)

class MultiResBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters*(1+2+4), kernel_size=1, padding='same'),
            nn.BatchNorm1d(n_filters*(1+2+4)),
            nn.ReLU()
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=15, padding='same'),
            nn.BatchNorm1d(n_filters),
            nn.ReLU()
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters*2, kernel_size=15, padding='same'),
            nn.BatchNorm1d(n_filters*2),
            nn.ReLU()
        )

        self.conv7x7 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters*4, kernel_size=15, padding='same'),
            nn.BatchNorm1d(n_filters*4),
            nn.ReLU()
        )

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(n_filters*(1+2+4))

    def forward(self, x):
        shortcut = self.shortcut(x)

        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x7 = self.conv7x7(x)

        x = torch.cat([x3, x5, x7], axis=1)
        x = self.bn(x)

        x = torch.add(x, shortcut)
        x = self.relu(x)
        x = self.bn(x)

        return x
        

class SPPLayer(nn.Module):
    def __init__(self, spp_windows):
        super().__init__()

        self.p_poolings = []

        for p in spp_windows:
            
            self.p_poolings.append(nn.Sequential(
                nn.MaxPool1d(kernel_size=p),
                nn.Flatten()
                )
            )


    def forward(self, x):
        out = torch.cat([layer(x) for layer in self.p_poolings], axis=1)
        return out
        

class BaseModel(nn.Module):
    def __init__(self, seq_len, n_classes):
        super().__init__()

        self.multiresblock = MultiResBlock(32)
        self.p = SPPLayer([8,16,32])
        
        size_out_multiresblock = 32*(1+2+4)  # annoying
        size_out_spp = size_out_multiresblock*(8+16+32)  # seq_len

        self.f = nn.Linear(size_out_spp, 128)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        self.out = nn.Linear(128, n_classes)
        self.softmax = nn.Softmax()

        self.sigmoid = nn.Sigmoid()

    def get_embedding(self, x):
        # Eval only
        x = self.multiresblock(x)
        x = self.p(x)
        x = self.f(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        return x 

    def forward(self, x):
        x = self.multiresblock(x)
        x = self.p(x)
        x = self.f(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.softmax(x)
        return x



class SiameseModel(nn.Module):
    def __init__(self, feat_len=128):
        super().__init__()

        # Input is 'L2'
        self.path1 = nn.Sequential(
            nn.Linear(in_features=feat_len, out_features=64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # Input is 'prod'
        self.path2 = nn.Sequential(
            nn.Linear(in_features=feat_len, out_features=64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # Input is 'combine'
        self.path3 = nn.Sequential(
            nn.Linear(in_features=feat_len*2, out_features=64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # Input is 'paths', output of concatenate(path1, path2, path3)
        self.top = nn.Sequential(
            nn.Linear(in_features=64*3, out_features=256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.out = nn.Linear(in_features=256, out_features=1)
    

    def forward(self, x1, x2):

        diff = torch.subtract(x1, x2)
        L2 = torch.multiply(diff, diff)
        prod = torch.multiply(x1, x2)
        combine = torch.cat([L2, prod], axis=1)

        path1 = self.path1(L2)
        path2 = self.path2(prod)
        path3 = self.path3(combine)

        concat = torch.cat([path1, path2, path3], axis=1)
        out = self.top(concat)
        out = self.out(out)
        out = out.flatten()
       
        return out 

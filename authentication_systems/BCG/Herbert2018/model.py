import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.01)

class Model(nn.Module):
    def __init__(self, sample_len=1000):
        super().__init__()

        # Each CNN layer is a 1D CNN with ReLU, then batchnorm, maxpool, dropout
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=2, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=2, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        if sample_len == 1000:  # fs=128
            end_len = 752
        elif sample_len == 1536:  # fs=512 but i mean this doenst actually make sense but whatever
            end_len = 3056
        else:
            raise NotImplementedError

        self.dense1 = nn.Linear(end_len, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.flatten(1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
        



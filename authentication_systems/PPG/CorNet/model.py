import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.01)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Each CNN layer is a 1D CNN with ReLU, then batchnorm, maxpool, dropout
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=40),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.1)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=40),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.1)
        )


        self.lstm1 = nn.LSTM(input_size=50, hidden_size=128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=128, num_layers=1, batch_first=True)

        self.dense = nn.Linear(4096, 50)
        self.relu = nn.ReLU()

        self.denseHR = nn.Linear(50, 1)

        self.denseID = nn.Linear(50, 1)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        # We think that input is 1000 len
        x = self.block1(x)  # [bs, 1, 1000] -> [bs, 32, 961] -> [bs, 32, 240] -> [bs, 32, 240] (paper says 250 but think typo)
        #  their data is 32 filters by 961 after conv, then 240 after pooling
        x = self.block2(x)   # [bs, 32, 240] -> [bs, 32, 201] -> [bs, 32, 50]
        #  their data is 32 filters by 201 after the conv, then 50 after the pool/dropout


       # Their data is 32 filters x 50 len by here
        x, hidden = self.lstm1(x)  # [bs, 32, 50] -> 
        #x, _ = self.lstm2(x)  # TODO come back to this

        x = x.flatten(1)  # [bs, 32, 128] -> [bs, 4096]
        x = self.dense(x)
        x = self.relu(x)

        hr = self.denseHR(x)
        hr = hr.flatten()  # to make my  life easier

        bid = self.denseID(x)
        bid = self.softmax(bid)

        return hr, bid
        



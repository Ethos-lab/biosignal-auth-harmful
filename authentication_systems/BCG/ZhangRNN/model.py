import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, T=10, num_classes=10, block_type='lstm', num_layers=1, input_size=450):
        super().__init__()

        if num_layers > 1:  raise NotImplementedError  # not yet

        self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=num_layers, batch_first=True)

        #self.dense = nn.Linear(T*256, num_classes)
        self.dense = nn.Linear(T*256, 1024)
        self.softmax = nn.Softmax()


    def forward(self, x):
        # Do the whole sequence at once: x is the all the hidden states throughout the sequence; out is the most recent hidden state and cell state
        # initial random hidden states
        # first should be initial hidden state for each element in the input sequence (so batch-item * 256)
        # second is intial cell state for each element in the input sequence 
        #hidden = (torch.randn(1, x.shape[0], 256).to(x.device), torch.randn(1, x.shape[0], 256).to(x.device))  # initialize random

        x, hidden = self.rnn1(x)
        x = x.flatten(1)  # I have no idea here, they dont say, but this makes sense
        x = self.dense(x) 
        x = self.softmax(x)
        return x

    def get_template(self, x):
        # Remove the softmax
        x, _ = self.rnn1(x)
        x = x.flatten(1)
        x = self.dense(x)
        x = self.softmax(x)
        return x

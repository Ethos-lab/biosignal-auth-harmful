import torch
import torch.nn as nn

# Tempalte can be found here: 
# https://github.com/eoduself/PPG-Verification-System/blob/master/2_channel_CNN_LSTM_single_session.ipynb

# It seems like they processed the data in matlab, then read it in and predicted in python (tensorflow), which 
# is probably what we'll have to do as well

def get_model(in_channels, num_classes, lstm=False, binary=False):
    return OneChannelCNN(in_channels, num_classes, lstm, binary)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.01)

        
class OneChannelCNN(nn.Module):
    def __init__(self, in_channels, num_classes, lstm=False, binary=False):
        """
        2 CNN layers and optionally 1 LSTM 
            CNN layer1: 50 filters, filter size is 60x1, 
            ReLU
            Dropout(0.50)
            CNN layer2: 70 filters, filter size is 60x50 
            ReLU
            Dropout(0.50)
            LSTM layer: 32 hidden units
            FullyConnected (unlcear how many)
            Binary classification 

        Note the filter size varies with the lenght of the input signal. Quoted are
        for input len 150 
        """
        super().__init__()
        assert binary  # no longer supporting the other kind
        self.do_lstm = lstm 
        self.binary = binary

        if in_channels == 2:
            kernel_size = (30, 60)  # layer 1, layer 2
        else:
            kernel_size = (60, 60)

        # padding None =96.  padding 5=> 106. padding 3=>102.
        # note: maybe they used sample_len as channels, which explains 50 and 70 choice here
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=50, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(0.5)
        )


        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=70, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
        if lstm:
            # NOTE hidden size is hardcoded for fs=128 and sample_len=1.25*fs
            self.lstm = nn.LSTM(input_size=72, hidden_size=32)
            flattened_len = 70*32
        else:
            flattened_len=70*72

        self.out =  nn.Linear(flattened_len, 1)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # tensor sizes assuming fs=128 and len=1.25*fs
        x = self.cnn1(x)  # [bs, 2, 160] => [bs, 50, 131]
        x = self.cnn2(x)  #             => [bs, 70, 72]
        if self.do_lstm:
            x, hidden = self.lstm(x)  # i guess if we dont flatten, it'll do each channel sep
        x = x.flatten(1)  # flatten the LSTM output before the FC 
        x = self.out(x)
        x = self.sigmoid(x)
        return x
        


"""
model = Sequential()
  model.add(Conv1D(filters=25, kernel_size=6, padding='valid',strides=1, input_shape=[60,1]))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling1D(pool_size=4))
  model.add(Dropout(0.25))

  #second CNN
  model.add(Conv1D(filters=40, kernel_size=10, padding='valid'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling1D(pool_size=4))
  model.add(Dropout(0.25))

  #first LSTM. note that we need to do a timedistributed flatten as a transition from CNN to LSTM
  model.add(TimeDistributed(Flatten()))
  model.add(Bidirectional(LSTM(units=32, return_sequences=True, dropout=0.25)))
  model.add(Flatten())

  #activation layer
  model.add(Dense(2, activation='softmax'))

"""

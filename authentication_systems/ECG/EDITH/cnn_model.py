"""
    CNN Classification Model
"""

from keras.models import Model
from keras.layers import Input, Activation, Dropout, Dense, Flatten, BatchNormalization, MaxPool1D, Add, Conv1D, concatenate



def MultiresBlock(inp,n_filters,act):
    """
    Implementation of 1D MultiRes Block

    Args:
        inp (keras layer): input layer 
        n_filters (int): number of filters
        act (string): activation function

    Returns:
        [keras layer]: returns the multires block
    """

    

    shortcut = inp                                                                                      # shortcut connection

    shortcut = Conv1D(filters = n_filters * (1+2+4), kernel_size = 1, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation(act)(shortcut)

    conv3x3 = Conv1D(filters = n_filters, kernel_size = 15, padding='same')(inp)                        # 1st level convolution
    conv3x3 = BatchNormalization()(conv3x3)
    conv3x3 = Activation(act)(conv3x3)

    conv5x5 = Conv1D(filters = n_filters*2, kernel_size = 15, padding='same')(conv3x3)                  # 2nd level convolution
    conv5x5 = BatchNormalization()(conv5x5)
    conv5x5 = Activation(act)(conv5x5)


    conv7x7 = Conv1D(filters = n_filters*4, kernel_size = 15, padding='same')(conv5x5)                  # 3rd level convolution
    conv7x7 = BatchNormalization()(conv7x7)
    conv7x7 = Activation(act)(conv7x7)

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)                                             # merging the featuremaps
    out = BatchNormalization(axis=-1)(out)

    out = Add()([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=-1)(out)

    return out



def SPPLayer(inp, spp_windows):
    """
    Implementaion of 1D Spatial Pyramid Pooling

    Args:
        inp (keras layer): input layer 
        spp_windows (int []): array of different pooling window size

    Returns:
        [keras layer]: returns the spp block
    """

    p_poolings = []                                                                     # pooling operations

    for pi in range(len(spp_windows)):
        
        p_poolings.append(Flatten()(MaxPool1D(pool_size=spp_windows[pi],padding='same')(inp)))      # adding the different pooling operations

    out = concatenate(p_poolings, axis=-1)

    return out


def getModel(seq_len,n_classes,dp_rate=0.25):
    """
    Implementation of the CNN model for classification

    Args:
        seq_len (int): length in input ECG signal
        n_classes (int): number of output classes, i.e. subjects 
        dp_rate (float, optional): dropout probability. defaults to 0.25.

    Returns:
        [keras model]: returns the proposed model
    """

    inp = Input(shape=(seq_len,1))                      # input layer

    x = MultiresBlock(inp, 32, 'relu')                  # multires block for feature extraction

    p = SPPLayer(x, [8,16,32])                          # pooling with spp block
                 
    f = Dense(128)(p)                                                    
    f = BatchNormalization()(f)
    top = Activation('relu')(f)   # VEENA note: this is the embedding that gets sent to Siamese. TBD can model.layers.pop(ix)

    top = Dropout(dp_rate)(top)                             # dropout

    out = Dense(n_classes, activation='softmax')(top)       # output layer

    model = Model(inputs=[inp],outputs=[out])               # keras model

    return model


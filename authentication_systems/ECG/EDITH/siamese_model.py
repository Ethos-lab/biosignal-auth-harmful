"""
    Proposed Siamese Model
"""

from keras.models import Model
from keras.layers import Input, Subtract, Multiply, concatenate, Dense, BatchNormalization, Dropout, Activation

def getSiameseModel(feat_len):
    """
    Implementation of the Siamese Model

    To make the training (also testing) process faster, instead of including 
    the base models, we precompute the features obtained from the base models
    and directly input them in the Siamese model.

    Args:
        feat_len (int): number of features obtained from the base model

    Returns:
        [keras model]: the siamese model
    """

    inp1 = Input(shape=(feat_len,))
    inp2 = Input(shape=(feat_len,))

    diff = Subtract()([inp1,inp2])
    L2 = Multiply()([diff,diff])                    # squared difference 

    prod = Multiply()([inp1,inp2])                  # product proximity

    combine = concatenate([L2,prod])                # combined metric

    path1 = Dense(64)(L2)
    path1 = BatchNormalization()(path1)
    path1 = Dropout(0.25)(path1)
    path1 = Activation('relu')(path1)

    path2 = Dense(64)(prod)
    path2 = BatchNormalization()(path2)
    path2 = Dropout(0.25)(path2)
    path2 = Activation('relu')(path2)

    path3 = Dense(64)(combine)
    path3 = BatchNormalization()(path3)
    path3 = Dropout(0.25)(path3)
    path3 = Activation('relu')(path3)


    paths = concatenate([path1,path2,path3])        # combining everything


    top = Dense(256)(paths)
    top = BatchNormalization()(top)
    top = Dropout(0.25)(top)#
    top = Activation('relu')(top)


    out = Dense(1)(top)                             # output similarity score

    siamese_model = Model(inputs=[inp1,inp2],outputs=[out])

    return siamese_model

import tensorflow as tf
from keras.layers import Conv2D, Input, MaxPooling2D, Permute, Reshape, Dropout, Bidirectional, GRU, TimeDistributed, BatchNormalization, Activation, Dense
from keras.models import Model


def create_model(num_features, rnn_sizes, dropout_rate, num_classes):
    # 1 acts as channel for theConv2D layer
    visible = Input(shape=(num_features, 375, 1))
    conv1 = Conv2D(8, kernel_size=(3, 3), padding='same')(visible)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 1))(conv1)
    conv2 = Conv2D(16, kernel_size=(3, 3),
                   activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 1))(conv2)
    conv3 = Conv2D(8, kernel_size=(3, 3),
                   activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)
    pool3 = Permute((2, 1, 3))(pool3)

    pool3 = Reshape((375, -1))(pool3)
    pool3 = Dropout(rate=0.2)(pool3)

    for rnn_size in rnn_sizes:
        pool3 = Bidirectional(GRU(rnn_size,
                                  activation='tanh',
                                  dropout=dropout_rate,
                                  recurrent_dropout=dropout_rate,
                                  return_sequences=True), merge_mode='mul')(pool3)

    # dense = TimeDistributed(Dense(fc_size))(pool3)
    dense = TimeDistributed(Dense(num_classes))(pool3)
    output = Activation('sigmoid')(dense)
    model = Model(inputs=visible, outputs=output)
    return model

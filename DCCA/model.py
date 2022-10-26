import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, Concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model


def create_DCCA_model(dcca_params: dict):
    visible1 = Input(shape=(375, dcca_params['in_shape1']))
    conv11 = Conv1D(128, kernel_size=3, padding='same')(visible1)
    conv11 = BatchNormalization(axis=1)(conv11)
    conv11 = Activation('relu')(conv11)
    conv21 = Conv1D(128, kernel_size=3, activation='relu',
                    padding='same')(conv11)
    conv21 = BatchNormalization(axis=1)(conv21)
    conv21 = Activation('relu')(conv21)
    conv21 = Conv1D(128, kernel_size=3, activation='relu',
                    padding='same')(conv21)
    conv21 = BatchNormalization(axis=1)(conv21)
    conv21 = Activation('relu')(conv21)
    conv31 = Conv1D(dcca_params['out_shape1'], kernel_size=3, activation='relu',
                    padding='same')(conv21)
    conv31 = BatchNormalization(axis=1)(conv31)
    output1 = Activation('relu')(conv31)

    visible2 = Input(shape=(375, dcca_params['in_shape2']))
    conv12 = Conv1D(128, kernel_size=3, padding='same')(visible2)
    conv12 = BatchNormalization(axis=1)(conv12)
    conv12 = Activation('relu')(conv12)
    conv22 = Conv1D(128, kernel_size=3, activation='relu',
                    padding='same')(conv12)
    conv22 = BatchNormalization(axis=1)(conv22)
    conv22 = Activation('relu')(conv22)
    conv22 = Conv1D(128, kernel_size=3, activation='relu',
                    padding='same')(conv22)
    conv22 = BatchNormalization(axis=1)(conv22)
    conv22 = Activation('relu')(conv22)
    conv32 = Conv1D(dcca_params['out_shape2'], kernel_size=3, activation='relu',
                    padding='same')(conv22)
    conv32 = BatchNormalization(axis=1)(conv32)
    output2 = Activation('relu')(conv32)

    merge_layer = Concatenate(name="merge_layer", axis=2)([output1, output2])

    model = Model(inputs=[visible1, visible2], outputs=merge_layer)

    return model

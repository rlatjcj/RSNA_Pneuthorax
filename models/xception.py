import keras
import keras.backend as K
from keras.layers import (
    Input,
    Conv2D,
    GaussianNoise,
    SeparableConv2D,
    Add
)

from keras.models import Model

from .layers import _normalization, _activation, _downsizing, _se_block

def Xception(
    input_shape=(256, 256, 1),
    norm='bn',
    activation='relu',
    downsizing='pooling',
    divide=1.,
    is_seblock=False,
    **kwargs):

    img_input = Input(shape=input_shape)
    
    c1 = GaussianNoise(0.1)(img_input)
    c1 = Conv2D(int(32/divide), (3, 3), strides=(2, 2), use_bias=False, padding='same')(c1)
    c1 = _normalization(c1, norm=norm)
    c1 = _activation(c1, activation=activation)
    c1 = Conv2D(int(64/divide), (3, 3), use_bias=False, padding='same')(c1)
    c1 = _normalization(c1, norm=norm)
    c1 = _activation(c1, activation=activation)

    residual = Conv2D(int(128/divide), (1, 1), strides=(1, 1), padding='same', use_bias=False)(c1)
    residual = _normalization(residual, norm=norm)

    c2 = SeparableConv2D(int(128/divide), (3, 3), padding='same', use_bias=False)(c1)
    c2 = _normalization(c2, norm=norm)
    c2 = _activation(c2, activation=activation)
    c2 = SeparableConv2D(int(128/divide), (3, 3), padding='same', use_bias=False)(c2)
    c2 = _normalization(c2, norm=norm)

    if is_seblock:
        c2 = _se_block(c2, filters=int(128/divide))
    c2 = Add()([c2, residual])
    c2 = _activation(c2, activation=activation) #

    c3 = _downsizing(c2, filters=int(128/divide), downsizing=downsizing, norm=norm, activation=activation)
    residual = Conv2D(int(256/divide), (1, 1), strides=(1, 1), padding='same', use_bias=False)(c3)
    residual = _normalization(residual, norm=norm)

    c3 = SeparableConv2D(int(256/divide), (3, 3), padding='same', use_bias=False)(c3)
    c3 = _normalization(c3, norm=norm)
    c3 = _activation(c3, activation=activation)
    c3 = SeparableConv2D(int(256/divide), (3, 3), padding='same', use_bias=False)(c3)
    c3 = _normalization(c3, norm=norm)

    if is_seblock:
        c3 = _se_block(c3, filters=int(256/divide))
    c3 = Add()([c3, residual])
    c3 = _activation(c3, activation=activation) #

    c4 = _downsizing(c3, filters=int(256/divide), downsizing=downsizing, norm=norm, activation=activation)
    residual = Conv2D(int(728/divide), (1, 1), strides=(1, 1), padding='same', use_bias=False)(c4)
    residual = _normalization(residual, norm=norm)

    c4 = SeparableConv2D(int(728/divide), (3, 3), padding='same', use_bias=False)(c4)
    c4 = _normalization(c4, norm=norm)
    c4 = _activation(c4, activation=activation)
    c4 = SeparableConv2D(int(728/divide), (3, 3), padding='same', use_bias=False)(c4)
    c4 = _normalization(c4, norm=norm)

    if is_seblock:
        c4 = _se_block(c4, filters=int(728/divide))
    c4 = Add()([c4, residual])
    c4 = _activation(c4, activation=activation) #

    c5 = _downsizing(c4, filters=int(728/divide), downsizing=downsizing, norm=norm, activation=activation)
    for i in range(8):
        residual = c5

        c5 = SeparableConv2D(int(728/divide), (3, 3), padding='same', use_bias=False)(c5)
        c5 = _normalization(c5, norm=norm)
        c5 = _activation(c5, activation=activation)

        c5 = SeparableConv2D(int(728/divide), (3, 3), padding='same', use_bias=False)(c5)
        c5 = _normalization(c5, norm=norm)
        c5 = _activation(c5, activation=activation)

        c5 = SeparableConv2D(int(728/divide), (3, 3), padding='same', use_bias=False)(c5)
        c5 = _normalization(c5, norm=norm)
        
        if is_seblock:
            c5 = _se_block(c5, filters=int(728/divide))
        c5 = Add()([c5, residual])
        c5 = _activation(c5, activation=activation)

    residual = Conv2D(int(1024/divide), (1, 1), strides=(1, 1), padding='same', use_bias=False)(c5)
    residual = _normalization(residual, norm=norm)

    c5 = SeparableConv2D(int(728/divide), (3, 3), padding='same', use_bias=False)(c5)
    c5 = _normalization(c5, norm=norm)
    c5 = _activation(c5, activation=activation)

    c5 = SeparableConv2D(int(1024/divide), (3, 3), padding='same', use_bias=False)(c5)
    c5 = _normalization(c5, norm=norm)

    if is_seblock:
        c5 = _se_block(c5, filters=int(1024/divide))
    c5 = Add()([c5, residual])
    c5 = _activation(c5, activation=activation) #

    c6 = _downsizing(c5, filters=int(1024/divide), downsizing=downsizing, norm=norm, activation=activation)
    c6 = SeparableConv2D(int(1536/divide), (3, 3), padding='same', use_bias=False)(c6)
    c6 = _normalization(c6, norm=norm)
    c6 = _activation(c6, activation=activation)

    c6 = SeparableConv2D(int(2048/divide), (3, 3), padding='same', use_bias=False)(c6)
    c6 = _normalization(c6, norm=norm)
    c6 = _activation(c6, activation=activation)

    model = Model([img_input], [c2, c3, c4, c5, c6], name='xception')

    return model

if __name__ == "__main__":
    model = Xception(input_shape=(256, 256, 1))
    model.summary()
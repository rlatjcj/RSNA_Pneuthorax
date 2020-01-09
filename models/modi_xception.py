import keras
import keras.backend as K
from keras.layers import (
    Input,
    Conv2D,
    GaussianNoise,
    DepthwiseConv2D,
    ZeroPadding2D,
    Add
)

from keras.models import Model

from .layers import _normalization, _activation, _downsizing, _se_block

def Conv2D_same(x, filters, strides=1, kernel_size=3, rate=1):
    if strides == 1:
        return Conv2D(filters, (kernel_size, kernel_size),
                      strides=(strides, strides),
                      padding='same',
                      use_bias=False,
                      dilation_rate=(rate, rate))(x)

    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters, (kernel_size, kernel_size),
                      strides=(strides, strides),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate))(x)


def SeparableConvBlock(x, filters, kernel_size=3, strides=1, rate=1, activation='relu', norm='bn', last_activation=False):
    '''
    Separable Convolution Block used in MobileNet and Xception
    '''
    if strides == 1:
        padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        padding='valid'

    if not last_activation:
        x = _activation(x, activation=activation)

    # Depwise Convolution
    x = DepthwiseConv2D((3, 3), strides=(strides, strides), use_bias=False, padding=padding, 
                        dilation_rate=(rate, rate))(x)
    x = _normalization(x, norm=norm)
    x = _activation(x, activation=activation)

    # Pointwise Convolution
    x = Conv2D(filters, (1, 1), use_bias=False, padding='same')(x)
    x = _normalization(x, norm=norm)
    if last_activation:
        x = _activation(x, activation=activation)

    return x

def xception_block(inputs, filters, strides, rate=1, mode='conv', activation='relu', norm='bn', is_seblock=False, last_activation=False):
    '''
    Xception Block in DeepLabv3+
    '''

    x = inputs
    for i in range(3):
        x = SeparableConvBlock(x, filters[i],
                               strides=strides if i == 2 else 1,
                               rate=rate,
                               activation=activation,
                               norm=norm,
                               last_activation=last_activation)

        if i == 1:
            skip = x
    
    if is_seblock:
        x = _se_block(x, filters=filters[2])
    if mode == 'conv':
        residual = Conv2D_same(inputs, filters[1], kernel_size=1, strides=strides)
        residual = _normalization(residual, norm=norm)
        x = Add()([x, residual])
    elif mode == 'sum':
        x = Add()([x, inputs])

    return x, skip

def Xception_modified(
    input_shape=(256, 256, 1),
    norm='bn',
    activation='relu',
    downsizing='pooling',
    divide=1.,
    is_seblock=False,
    **kwargs):
    
    img_input = Input(shape=input_shape)

    # Entry flow
    c1 = GaussianNoise(0.1)(img_input)
    c1 = Conv2D(int(32/divide), (3, 3), strides=(2, 2), use_bias=False, padding='same')(c1)
    c1 = _normalization(c1, norm=norm)
    c1 = _activation(c1, activation=activation)
    c1 = Conv2D_same(c1, int(64/divide), kernel_size=3, strides=1)
    c1 = _normalization(c1, norm=norm)
    c1 = _activation(c1, activation=activation)

    c2, c2_out = xception_block(c1, [int(128/divide), int(128/divide), int(128/divide)], 2, 1, mode='conv', activation=activation, norm=norm, is_seblock=is_seblock)
    c3, c3_out = xception_block(c2, [int(256/divide), int(256/divide), int(256/divide)], 2, 1, mode='conv', activation=activation, norm=norm, is_seblock=is_seblock)
    c4, c4_out = xception_block(c3, [int(728/divide), int(728/divide), int(728/divide)], 2, 1, mode='conv', activation=activation, norm=norm, is_seblock=is_seblock)

    # Middle flow
    c5 = c4
    for i in range(8):
        c5, _ = xception_block(c5, [int(728/divide), int(728/divide), int(728/divide)], 1, 1, mode='sum', activation=activation, norm=norm, is_seblock=is_seblock)

    # Exit flow
    c5, c5_out = xception_block(c5, [int(728/divide), int(1024/divide), int(1024/divide)], 2, 1, mode='conv', activation=activation, norm=norm, is_seblock=is_seblock)
    c6, _ = xception_block(c5, [int(1536/divide), int(1536/divide), int(2048/divide)], 1, 2, mode='none', activation=activation, norm=norm, is_seblock=False, last_activation=True)

    model = Model([img_input], [c2_out, c3_out, c4_out, c5_out, c6], name='xception_modified')

    return model

if __name__ == "__main__":
    model = Xception_modified(input_shape=(256, 256, 1))
    model.summary()
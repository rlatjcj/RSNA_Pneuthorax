import keras
from keras.layers import (
    Input,
    Conv2D,
    GaussianNoise,
    Add
)

from keras.models import Model

from .layers import _normalization, _activation, _downsizing, _se_block

def Vgg16(
    input_shape=(256, 256, 1),
    norm='bn',
    activation='relu',
    downsizing='pooling',
    divide=1.,
    classes=2,
    **kwargs):

    img_input = Input(shape=input_shape)
    if K.learning_phase() == 1:
        img_input = GaussianNoise(0.1)(img_input)

    # Block 1
    c1 = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    c1 = _normalization(c1, norm=norm)
    c1 = _activation(c1, activation=activation)
    c1 = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(c1)
    c1 = _normalization(c1, norm=norm)
    c1 = _activation(c1, activation=activation)

    # Block 2
    c2 = _downsizing(c1, filters=64, downsizing=downsizing, norm=norm, activation=activation)
    c2 = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(c2)
    c2 = _normalization(c2, norm=norm)
    c2 = _activation(c2, activation=activation)
    c2 = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(c2)
    c2 = _normalization(c2, norm=norm)
    c2 = _activation(c2, activation=activation)

    # Block 3
    c3 = _downsizing(c2, filters=128, downsizing=downsizing, norm=norm, activation=activation)
    c3 = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(c3)
    c3 = _normalization(c3, norm=norm)
    c3 = _activation(c3, activation=activation)
    c3 = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(c3)
    c3 = _normalization(c3, norm=norm)
    c3 = _activation(c3, activation=activation)
    c3 = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(c3)
    c3 = _normalization(c3, norm=norm)
    c3 = _activation(c3, activation=activation)


    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
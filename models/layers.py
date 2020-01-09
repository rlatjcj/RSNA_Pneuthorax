from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Multiply

from keras_contrib.layers import InstanceNormalization
from keras_contrib.layers import GroupNormalization

def _normalization(inputs, norm='bn'):
    if norm == 'bn':
        return BatchNormalization()(inputs)
    elif norm == 'in':
        return InstanceNormalization()(inputs)
    elif norm == 'gn':
        return GroupNormalization()(inputs)

def _activation(inputs, activation='relu', name=None):
    if activation in ['relu', 'sigmoid', 'softmax']:
        return Activation(activation, name=name)(inputs)
    elif activation == 'leakyrelu':
        return LeakyReLU(alpha=.3, name=name)(inputs)

def _downsizing(inputs, filters, downsizing='pooling', norm='bn', activation='relu'):
    if downsizing == 'pooling':
        return MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inputs)
    elif downsizing == 'conv':
        x = Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(inputs)
        x = _activation(x, activation=activation)
        x = _normalization(x, norm=norm)
        return x

def _se_block(inputs, filters, se_ratio=16):
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(filters//se_ratio, activation='relu')(x)
    x = Dense(filters, activation='sigmoid')(x)
    x = Reshape([1, 1, filters])(x)
    x = Multiply()([inputs, x])
    return x
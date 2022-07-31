from tensorflow.keras.layers import Conv2D, Input, PReLU, Conv2DTranspose, ReLU
from tensorflow.keras.initializers import RandomNormal, HeNormal
from tensorflow.keras.models import Model
import tensorflow as tf

def Residual_block(X_in, filters, kernel_size, strides, padding, scale_factor):
    X = Conv2D(filters, kernel_size, strides, padding)(X_in)
    X = ReLU()(X)
    X = Conv2D(filters, kernel_size, strides, padding)(X)
    X = X * scale_factor
    return (X_in + X)

def Upsample(X_in, scale):
    X = Conv2D(filters=3*scale*scale, kernel_size=3, padding="SAME")(X_in)
    if scale in [2, 3]:
        X = tf.nn.depth_to_space(X, scale)
    elif scale == 4:
        X = tf.nn.depth_to_space(X, 2)
        X = Conv2D(filters=3*scale*scale, kernel_size=3, padding="SAME")(X)
        X = tf.nn.depth_to_space(X, 2)
    return X

def EDSR_network(scale):
    filters = 256
    kernel_size = 3
    strides = 1
    padding = "SAME"
    scale_factor = 0.1

    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters, kernel_size, strides, padding)(X_in)
    for _ in range(32):
        X = Residual_block(X, filters, kernel_size, strides, padding, scale_factor)
    X = Upsample(X, scale)
    X = Conv2D(3, kernel_size, strides, padding)(X)
    X_out = tf.clip_by_value(X_in, 0.0, 1.0)

    return Model(X_in, X_out)


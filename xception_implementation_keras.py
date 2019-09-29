import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ZeroPadding2D, AveragePooling2D, GaussianNoise, SeparableConv2D
config_img_height = 220
config_img_width = 220


def separable_resnet_stack(x_input, n_filters = 728):
    """
    Stack of 3 separable convolutional layers with input added back as in resnet architecture
    Args:
        x: input array (should not have activation already applied to it)
        n_filters: number of filters in each convolutional layer (integer)
    """
    skip_layer = x_input
    
    x = Activation('relu')(x_input)
    
    x = SeparableConv2D(n_filters, (3,3), padding = 'same', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv2D(n_filters, (3,3), padding = 'same', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv2D(n_filters, (3,3), padding = 'same', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Add()([x, skip_layer])
    return x
 


def xception_conv(n_classes = 5, img_h = config_img_height, img_w = config_img_width, n_channels = 3):
    """
    Keras implementation of Xception architecture created by Francois Chollet (https://arxiv.org/abs/1610.02357)
    Args:
        n_classes: number of classes - used in softmax layer
        img_h: input image height
        img_w: input image width
        n_channels: number of channels in input image (3 for rgb)
    Returns:
        Keras Model() object
    """

    input_shape = (img_h, img_w, n_channels)
    x_input = Input(input_shape)
    
    # Two small, initial convolutional layers
    x = Conv2D(32, (7,7), strides = (2,2), use_bias = False) (x_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3,3), use_bias = False) (x_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual layer 1
    skip_layer = Conv2D(64, (1,1), strides = (2,2), padding = 'same', use_bias = False)(x)
    skip_layer = BatchNormalization()(skip_layer)
    
    # Initial separable convolutional layers
    x = SeparableConv2D(64, (3,3), padding = 'same', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(64, (3,3), padding = 'same', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3,3), strides = (2,2), padding = 'same')(x)
    
    # Add back residual layer
    x = Add()([x, skip_layer])
        
    # Second set of separable convolutional layers
    x = SeparableConv2D(128, (3,3), padding = 'same', use_bias = False)(x)
    x = BatchNormalization()(x)
    
    # Several consecutive separable convolutional layers with resnet skip layers
    x = separable_resnet_stack(x, n_filters = 128)
    x = separable_resnet_stack(x, n_filters = 128)
    x = separable_resnet_stack(x, n_filters = 128)
    x = Activation('relu')(x)
    
    # Final separable convolutional layers
    x = SeparableConv2D(256, (3,3), padding = 'same', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SeparableConv2D(512, (3,3), padding = 'same', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Dimension Reduction
    x = GlobalAvgPool2D(name = 'global_avg_pooling')(x) 
    x = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs = x_input, outputs = x, name = 'resnet_28_layer') 
    return model



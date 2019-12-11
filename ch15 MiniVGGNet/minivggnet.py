# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:29:17 2019

This scrip create a miniVGGNet architecture which contains the following layers:
- Two sets of CONV => RELU => CONV => RELU => POOL layers,
- Followed by a set of FC => RELU => FC => SOFTMAX layers.
- Batch normalization will also be applied in between CONV and FC
- Dropout layers are connected between CONV and FC as well.

The VGGNet applies a small filter only 3x3 which makes it so special


@author: jschang
"""

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class miniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initailize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # chanDim=-1 represents channel last order
        chanDim = -1
        
        if K.image_date_format()=="channel_first":
            inputShape = (depth, height, width)
            # chanDim=1 represnets channel first order
            chanDim = 1
        
        # first CONV->RELU ->Conv ->RELU -> POOL
        model.add(Conv2D(32,(3,3), padding="same", 
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        # second CONV->RELU ->Conv ->RELU -> POOL
        model.add(Conv2D(64, (32, 32), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        # FC->RELU
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model
        
        
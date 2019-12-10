# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 21:42:46 2019

This scrip create a shallowNet CNN network with the following structure
input -> CONV -> RELU -> FC

@author: fatyc
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as k


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model with input shape to be 
        # the "channel last"
        model = Sequential()
        inputShape = (height, width, depth)
        
        # if we are using "channel first"
        # modify the input shape
        if k.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            
        # build the shallowNet
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        
        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model
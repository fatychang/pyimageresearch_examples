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

from keras.model import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D

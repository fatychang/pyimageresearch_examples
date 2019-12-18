# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:15:53 2019

This script demonstrates how to visualize the network from Keras
with graphviz and pydot

@author: jschang
"""

from keras.utils import plot_model
import sys
sys.path.append('D:\Jen\_Documents\eLearning\Computer Vision\pyimagesearch\ch15 MiniVGGNet')
from minivggnet import miniVGGNet

# construct the miniVGGNet and visualize it
model = miniVGGNet.build(32,32,3,10)
plot_model(model, to_file="miniVGGNet.png", show_shapes=True)
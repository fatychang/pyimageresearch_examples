# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:56:47 2019

This is an example for runing percenptron structure to predict bitwise dataset
You may use AND, OR and XOR in as the dataset.
A preceptron class is called in this example.

An example from book deep learning for computer vision with Python ch10

@author: fatyc
"""

from perceptron import Perceptron
import numpy as np

# construct the dataset
dataset_OR = np.asanyarray([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
dataset_AND = np.asanyarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
dataset_XOR = np.asanyarray([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

# extract the input and output
dataset = dataset_XOR
x = dataset[:,0:2]
y = dataset[:,-1]

# define the preceptron
print("[INFO] training preceptron...")
p = Perceptron(x.shape[1], alpha = 0.1)
p.fit(x, y, epochs=20)

# test the preceptron
print("[INFO] testing preceptron...")

# loop over the data point
for(x, target) in zip(x, y):
    # make the prediction on the data point
    pred = p.predict(x)
    
    # print out the result
    print("[INFO] data={}, ground-trut={}, pred={}".format(
            x, target, pred))


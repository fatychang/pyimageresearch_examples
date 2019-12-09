# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:20:40 2019

This is an example for runing neural network to predict bitwise dataset
You may use AND, OR and XOR in as the dataset.
A neuralnetwork class is called in this example.

An example from book deep learning for computer vision with Python ch10

@author: fatyc
"""

from neuralnetwork import NeuralNetwork
import numpy as np

# construct the dataset
dataset_OR = np.asanyarray([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
dataset_AND = np.asanyarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
dataset_XOR = np.asanyarray([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

# extract the input and output
dataset = dataset_AND
x = dataset[:,0:2]
y = dataset[:,-1]

# define the 2-2-1 network and train it
nn = NeuralNetwork([2,2,1], alpha=0.01)
nn.fit(x, y, epochs=20000)

# loop over the dataset and see the result
for (x, target) in zip (x, y):
    # make predictions
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format( 
             x, target, pred, step))
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:31:03 2019

This is a class for Perceptron structure in ANN

An example from book deep learning for computer vision with Python ch10

@author: fatyc
"""

import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        #initailize the weights matrix and store the learning rate
        self.W = np.random.randn(N+1)/np.sqrt(N)
        self.alpha = alpha
    
    def step(self, x):
        # apply the step function
        return 1 if x > 0 else 0
    
    def fit(self, X, y, epochs=10):
        # insert a column of 1;s as the last entry in the feature matrix
        # this little trick allows us to treat the bias as a trainable
        # parameter within the weight matrix
        X = np.c_[X, np.ones(X.shape[0])]
        
        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual point
            for (x, target) in zip(X, y):
                # take the dot product between the input features
                # and the weight matrix, then pass the value 
                # through the step function to obtain the predictions
                p = self.step(np.dot(x, self.W))
                
                # only perform a weight update if our prediction
                # does not match the target
                if p != target:
                    # determine the error
                    error = p-target
                    
                    # update the weight
                    self.W += -self.alpha *error * x
                    
    def predict(self, X, addBias=True):
        # ensure the input is a matrix
        X = np.atleast_2d(X)
        
        # check if the bias column should be added
        if addBias:
            # insert a column to the last entry in the feature matrix
            X = np.c_[X, np.ones((X.shape[0]))]
        
        # take the dot product of the input feature and the weight
        # then pass to step function to get the output
        return self.step(np.dot(X, self.W))
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:01:08 2019

This is a class for Neural Network implementation

An example from book deep learning for computer vision with Python ch10

@author: fatyc
"""

import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        #initialize the list of weights matrices, then store the network 
        # architecture and learning rate
        self.W =[]
        self.layers = layers
        self.alpha = alpha
        
        # initailize the weights
        for i in np.arange(0, len(layers)-2): # start looping when reaching the last two layers
            w = np.random.randn(layers[i] +1, layers[i+1]+1)
            self.W.append(w/np.sqrt(layers[i]))
            
        # the last two layers are a special case since there is no need for
        # the bias term in the output layer
        w = np.random.randn(layers[-2] +1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))
        
    def __repr__(self):
        # construct and return a string that represents the network
        return "NeuralNetwork: {}".format(
                "-".join(str(l) for l in self.layers))
        
    def sigmoid(self, x):
        # compute and return the signmoid activation value
        return 1.0/(1+np.exp(-x))
    
    def sigmoid_deriv(self, x):
        # compute and return the derivative of sigmoind function
        return x*(1-x)
    
    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # insert the bias term for the input
        X = np.c_[X, np.ones((X.shape[0]))]
        
        # loop over the desired number of epochs
        for epoch in range(0, epochs):
            # loop over each individual data point
            for (x, target) in zip(X,y):
                self.fit_partial(x, target)
                
                # check if we should display a training update
                if epoch==0 or (epoch +1)% displayUpdate == 0:
                    loss = self.calculate_loss(X, y)
                    print("[INFO] epoch={}, loss={:.7f}".format(epoch+1, loss))
    
    
    def fit_partial(self, x, y):
        
        # construct our list of output activations for each layer  
        # as our data point flows through the network; the first 
        # activation is a special case -- it’s just the input 
        # feature vector itself 
        A = [np.atleast_2d(x)]
         
        # feedforward
        # loop over the layers in the network
        for layer in range(0, len(self.W)):
            # calculate the forward value
            net = A[layer].dot(self.W[layer])
             
            # pass the activation function
            out = self.sigmoid(net)
             
            # add the net, output to the list
            A.append(out)
             
        # backward propagation
        # compute the error
        error = A[-1] - y
         
        # build the list of delta, D
        D = [error*self.sigmoid_deriv(A[-1])]
         
        # apply chain rules
        for layer in range(len(A) -2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta*self.sigmoid_deriv(A[layer])
            D.append(delta)
         
        # reverse the delta list since we did it in a reversed order
        D = D[::-1]
        
        # weight update phase
        # loop over thelayers
        for layer in range(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
       
    
    def predict(self, X, addBias=True):
        # initialize the output as the input features
        p = np.atleast_2d(X)
        
        # check if bias is required
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        
        # loop over the layers in the network
        for layer in range(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        
        return p
    
    def calculate_loss(self, X, target):
        targets = np.atleast_2d(target)
        predictions = self.predict(X, addBias=False)
        loss = 0.5* np.sum((predictions-targets)**2)
        
        return loss
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
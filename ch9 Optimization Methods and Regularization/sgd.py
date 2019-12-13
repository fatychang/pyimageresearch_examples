# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:25:21 2019

This sciprt demonstrates how stochastic gradient descent works.

@author: jschang
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import numpy as np

def sigmoid_activation(x):
    # compute the sigmoid activation function
    return 1.0/(1+np.exp(-x))

def predict(X, W):
    # take the dot product of the features and weights
    preds = sigmoid_activation(X.dot(W))
    
    # apply step function to threshold the output
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    
    return preds

def next_batch(X, y, batch_size):
    # loop over the dataset X in mini-batches
    # yielding a tunple of the current batch data
    for i in np.arange(0, X.shape[0], batch_size):
        yield(X[i:i+batch_size], y[i:i+batch_size])
        
        
# generate the dataset
(X, y ) = make_blobs(n_samples=1000, n_features=2, centers=2, 
                    cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
        
# insert the bias term in the feature matrix
X = np.c_[X, np.ones((X.shape[0]))]

# split dataset to training and testing set
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=1)

# initialize the weight matrix and list for loss
W = np.random.rand(X.shape[1], 1)
losses=[]

# loop over the desired number of epochs
epochs=100
alpha=0.01
batch_size=32
for epoch in range(0, epochs):
    # initialize the total loss of each epoch
    epochLoss=[]
    
    # loop over the data in the batch
    for (batchX, batchY) in next_batch(X,y, batch_size=batch_size):
        # predict the result
        preds = sigmoid_activation(batchX.dot(W))
        
        # calculate the errors based on the predictions
        errors = preds - batchY
        epochLoss.append(np.sum(errors**2))
        
        # calculate the gradient
        gradient = batchX.T.dot(errors)
        
        # update weights
        W += -alpha * gradient
     
    # update loss history of each epoch
    loss = np.average(epochLoss)
    losses.append(loss)
    
    # display the result every 5 epochs
    if epoch%5 == 0:  
        print("[INFO] epoch={}, loss={:.7f}".format(
                int(epoch+1), loss))

# evaluate the result
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the classification result
plt.style.use("ggplot")
plt.figure()
plt.scatter(testX[:,0], testX[:,1], marker="o")

# plot the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), losses)
plt.show()
    



        
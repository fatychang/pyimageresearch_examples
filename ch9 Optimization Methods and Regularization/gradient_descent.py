# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:06:11 2019

This sciprt demonstrates how gradient descent works.

@author: jschang
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import numpy as np

def sigmoid_activation(x):
    # compute the sigmoid activation function
    return 1.0/ (1+np.exp(-x))

def predict(X, W):
    # take dot product of the features(X) and weights(W)
    preds = sigmoid_activation(X.dot(W))
    
    # apply step function to threshold the output
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    
    return preds

# generate the dataset with n=1000 and each data point is 2D
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert the bias term in the feature matrix
X = np.c_[X, np.ones((X.shape[0]))]

# split the data to training and test set as 50-50
(trainX, testX, trainY, testY) = train_test_split(X, y,
                                    test_size=0.5, random_state=1)
    
# initialize the weight matrix and list of loss
print("[INFO] training...")
W = np.random.rand(X.shape[1],1)
losses= []

# loop over the desired number of epochs
epochs = 100
alpha = 0.01
for epoch in np.arange(0, epochs):
    # compute the score function
    preds = sigmoid_activation(trainX.dot(W))

    # calculate the errors and loss
    errors = preds-trainY
    loss = np.sum(errors**2)
    losses.append(loss)
    
    # calcuate the gradient
    gradient = trainX.T.dot(errors)
    
    # update the weights
    W += -alpha*gradient
    
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
plt.title("Data visualization")
plt.scatter(testX[:, 0], testX[:, 1], marker="o")

# plot the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), losses)
plt.show()
        
        
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:11:55 2019

This script follows the example from the book
Deep learning for computer vision with python by Dr. Rosebroke, 2017

Ch12 Training Your first CNN (shallowNet)
In this project, it aims to train the CNN with CIFAR-10 dataset with Keras

@author: jschang
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np

# load the dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()

# scale the raw pixel intensities to range [0, 1.0]
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

# conver the labels from integers to binary vectors (one-hot encoding)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initailize the label names for cifar10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

# initialize the optimizer and model
print("[INFO] compiling shallownet...")
sgd = SGD(learning_rate=0.01)






# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
            predictions.argmax(axis=1),target_names=labelNames))


# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:37:28 2019

his script follows the example from the book
Deep learning for computer vision with python by Dr. Rosebroke, 2017

Ch15 MiniVGGNet
In this project, it aims to train the MiniVGGNet with CIFAR-10 dataset with Keras

@author: jschang
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras.optimizers import SGD
from minivggnet import miniVGGNet

import matplotlib.pyplot as plt

# load the dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()

# scale the raw data to range [0, 1.0]
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

# convert the label from integers to binary vectors (one-hot encoding)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initializae the label name for cifar10
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]



# initializae the model and optimizer
model = miniVGGNet.build(32, 32, 3, 10)


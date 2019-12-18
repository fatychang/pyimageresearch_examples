# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:44:49 2019

This script implements the technique of checkpointing the model
whenever there is an improvement happened to the model, it will serialize it.

In this script, we use the miniVGGNet on cifar10 dataset.

@author: jschang
"""

from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10

import sys
sys.path.append('D:\Jen\_Documents\eLearning\Computer Vision\pyimagesearch\ch15 MiniVGGNet')
from minivggnet import miniVGGNet

import os

output_path = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\ch18 Checkpointing Models\\serialized output\\"

# load the dataset and scale the data
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

# convert the label to binary format
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)


# construct the miniVGGNet
model = miniVGGNet.build(width=32, height=32, depth=3, classes=10)
sgd = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])


# construct the callback to save only the *best* model
# to the disk based on the validation loss
fname = os.path.sep.join([output_path, 
                         "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
                             save_best_only=True, verbose=1)
callbacks=[checkpoint]

# train the network
H=model.fit(trainX, trainY, validation_data=(testX, testY),
            batch_size=64, epochs=40, callbacks=callbacks, verbose=2)


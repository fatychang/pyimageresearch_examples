# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:54:31 2019

This script follows the example from the book
Deep learning for computer vision with python by Dr. Rosebroke, 2017

Ch10 Neural Network Fundamentals - Multi-layer Networks with Keras
In this project, it aims to train the network with MNIST dataset with Keras
And visualizae the training process (loss and accuracy) along the training

The accuracy reachs 92% but with ANN it may reach 99%.

@author: jschang
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets

import matplotlib.pyplot as plt
import numpy as np

# get the MNIST dataset (if it is your first time running this script,
# the download may take a minute - 55MB MNIST dataset will be download)
print("[INFO] loading MNIST (full) dataset...")
dataset = datasets.fetch_mldata("MNIST Original")

# scale the raw pixel intensities to the range [0, 1.0] then construct
# the training and testing splits
data = dataset.data.astype("float")/ 255.0
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target,
                                    test_size=0.25)

# convert the labels from integers to vectors (one-hot encoding)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# define the NN model
model = Sequential()
model.add(Dense(256, input_shape=(784,),activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training the network...")
sgd = SGD(learning_rate=0.01)
model.compile(sgd, loss="categorical_crossentropy", metrics=["accuracy"])
H = model.fit(trainX, trainY, batch_size=128, epochs=100,
              validation_data=(testX, testY))

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("loss/accuracy")           
plt.legend()
plt.show()










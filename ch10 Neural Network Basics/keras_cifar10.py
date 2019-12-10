# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:47:16 2019

This script follows the example from the book
Deep learning for computer vision with python by Dr. Rosebroke, 2017

Ch10 Neural Network Fundamentals - Multi-layer Networks with Keras
In this project, it aims to train the network with CIFAR-10 dataset with Keras

@author: jschang
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# get the dataset (~170MB)
(trainX, trainY), (testX, testY) = cifar10.load_data()

# scale the raw pixel intensities to range [0, 1.0]
# and reshape the design matrix (32*32*3)
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0
trainX = trainX.reshape(-1, 3072)
testX = testX.reshape(-1, 3072)

# conver the labels from integers to binary vectors (one-hot encoding)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initailize the label names for cifar10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

# define the NN model [3072-1024-512-10]
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training the network...")
sgd = SGD(learning_rate=0.01)
model.compile(sgd, loss="categorical_crossentropy", metrics=["accuracy"])
H = model.fit(trainX, trainY, batch_size=32, epochs=100,
              validation_data=(testX, testY))

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
            predictions.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))

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

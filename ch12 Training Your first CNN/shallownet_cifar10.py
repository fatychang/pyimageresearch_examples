# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:11:55 2019

This script follows the example from the book
Deep learning for computer vision with python by Dr. Rosebroke, 2017

Ch12 Training Your first CNN (shallowNet)
In this project, it aims to train the CNN with CIFAR-10 dataset with Keras

@author: jschang
"""


import sys
sys.path.append('D:\Jen\_Documents\eLearning\Computer Vision\pyimagesearch\ch12 Training Your first CNN')

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras.optimizers import SGD
from shallownet import ShallowNet
import matplotlib.pyplot as plt
import numpy as np
import pickle

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
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])

# train the network
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=40, verbose=1)





# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
            predictions.argmax(axis=1),target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("loss/accuracy")           
plt.legend()
plt.show()

#####################
## save the model and its history to the disk if necessary
#model.save("shallownet_cifar10_model.h5")
#
## save the history
#with open('shallownet_cifar10_model_history.pickle','wb') as file:
#    pickle.dump(H.history, file)
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:34:19 2019

@author: jschang
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import pickle


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


# load the model from the disk
from keras.models import load_model
model = load_model("nn_cifar10_model.h5")

# load the training history from the disk
pickle_in = open('nn_cifar10_model_history.pickle','rb')
H = pickle.load(pickle_in)


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
            predictions.argmax(axis=1),target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("loss/accuracy")           
plt.legend()
plt.show()
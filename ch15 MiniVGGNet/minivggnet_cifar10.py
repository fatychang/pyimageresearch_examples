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
import numpy as np
#import pickle

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
opt = SGD(learning_rate=0.01, decay = 0.01/40,
          momentum = 0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, 
              metrics=["accuracy"])

# train the network
H = model.fit(trainX, trainY, validation_data = (testX, testY),
              batch_size=64, epochs=40, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
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


## save the model and training history
#model.save("minivggnet_cifar10_model.h5")
#with open("minivggnet_cifar10_mode_history.pickle", "wb") as file:
#    pickle.dump(H.history, file)

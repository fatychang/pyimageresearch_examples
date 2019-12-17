# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:22:33 2019

This script creates a class that is responsible to monitor the training
during the trainig instead of after the whole process.

It served as a callback function that will be activate after each Epoch.


@author: jschang
"""

from keras.callbacks import BaseLogger

import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # store the output path for the figure, the path to json
        # serialized file and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
        self.H={}
        self.ctr=0
        
    def on_training_begin(self, logs={}):
        # initailize the history dictionary
        self.H={}
        
        # if the json history exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                
                # check whether the starting epoch is supplied
                if self.startAt > 0:
                    # loop over the entires in the history log 
                    # and trim any entries that past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]
    
    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k,[])
            l.append(v)
            self.H[k] = l
        
        # check if the training history should be serialized
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()
        
        
        # construct the actual plot at least every two epochs have passed
        if len(self.H["loss"]) > 1:
            self.ctr+=1
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="accur")
            plt.plot(N, self.H["val_accuracy"], label="val_accur")
            plt.title("Training Loss and Accuracy [Epoch {}]".format
                      (len(self.H["loss"])))
            plt.xlabel("#Epochs")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            
            # save the figure
            figname = self.figPath + "_{}".format(self.ctr)
            plt.savefig(figname)
            plt.close()
                    
        
            
            
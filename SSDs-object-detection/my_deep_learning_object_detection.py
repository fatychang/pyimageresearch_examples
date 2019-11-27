# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:20:01 2019

Object detection with deep learning and Opencv
Project guilde provided by pyimagesearch

This project implements an MobileNetSSD architecture to do object detection
The run for a single image is 0.06 seconds.

NOTED:
    the color in cv.rectangle() function requires a list input containing 
    "int" data type. Therefore, a conversion is required to convert the 
    COLOR[idx] (this is a numpy array) to the required format.

@author: jschang
"""

import numpy as np
import time
import cv2

dir_input_image = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\SSDs-object-detection\\images\\example_01.jpg"
dir_coffe_prototxt = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\SSDs-object-detection\\MobileNetSSD_deploy.prototxt.txt"
dir_coffe_model = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\SSDs-object-detection\\MobileNetSSD_deploy.caffemodel"
confidence_thresh = 0.5

# initialize the list of class labels MobileNet SSD was trained to detect
# and generate a set of bonding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.randint(0,255, size=(len(CLASSES), 3), dtype="uint8")


# load the serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(dir_coffe_prototxt, dir_coffe_model)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(dir_input_image)
(h, w) = image.shape[0:2]
blob = cv2.dnn.blobFromImage(image, 1/255.0, (300, 300), 127.5)


# pass the blob through the network and obtain the detections and predictions
start = time.time()
net.setInput(blob)
detections = net.forward()
end = time.time()
print("[INFO] classification took {:.6f} seconds".format(end-start))

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence associated witht the prediction
    confidence = detections[0, 0, i, 2]
    
    # filter out weak detections
    if confidence > confidence_thresh:
        # extract the index of the class label
        idx = int(detections[0, 0, i, 1])
        
        # extract and compute the bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # display the prediction
        text = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
        print("[INFO] {}".format(text))
        
        # draw the bounging box
        color = [int(c) for c in COLORS[idx]]
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        text_y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, text, (startX, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

# show the image
cv2.imshow("Image", image)
cv2.waitKey(0)



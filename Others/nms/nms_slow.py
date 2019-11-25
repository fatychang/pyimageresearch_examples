# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:43:26 2019

This script follows the sample from Pyimagesearch
the objective is to show the non-maxima-suppression (NMS) technique
which deals with multiple detections with a single object.

The post can be found in the following link.
https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

@author: jschang
"""

import nms
import numpy as np
import cv2

# construct a list containing the image along with their respective boundary
images = [
        ("audrey H.jpg", np.array([
        (67,10,809,815),
        (6,14,885,820),
        (32,35,846,787)])),
        ("two_people.jpg", np.array([
        (58,5,118,64),
        (144,7,219,71),
        (152,11,211,66)]))]

# Loop over the images
for (imagePath, boundingBoxes) in images:
    # load the image and clone it
    print("[x] %d initail boundary boxes" % (len(boundingBoxes)))
    image = cv2.imread(imagePath)
    orig = image.copy()

    # draw the boundary boxes
    for(startX, startY, endX, endY) in boundingBoxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0,0,255), 2)
    
    # apply nms on the bounding boxes
    pick = nms.non_max_suppression_slow(boundingBoxes, 0.3)
    print("[x] after applying nms, %d bounding boxes" % (len(pick)))
    
    # loop over and print the selected bounding box and draw them
    for(startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0), 2)
    
    # display images
    cv2.imshow("Original", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)



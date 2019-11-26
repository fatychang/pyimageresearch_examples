# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:33:06 2019


YOLO object detection with OpenCV
Project guilde provided by pyimagesearch

The script requires input like path for your target image, 
path for the pre-trainned YOLO model, 
confidence level (threshold) which used to filter out object that are unsured,
nms_threshold which removes multiple detection in a single object.

In this project the YOLO network is trainned on a COCO dataset, 
which contains 80 categories. The class name can be found in the LABELS

YOLO model performs poorly when two objects are too close to each other.
You may try the image named dinning_table to see the result.
Only one wine glass is able to be detected.

The average run-time for my pc is around 0.7 second.

@author: jschang
"""

#import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os


# Input arguments
dir_input_image = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\yolo-object-detection\\images\\dining_table.jpg"
dir_yoloModel = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\yolo-object-detection\\yolo-coco\\"
confidence_thresh = 0.2
nms_thresh=0.2

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([dir_yoloModel, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0,255, size=(len(LABELS), 3), dtype="uint8")

# load the weights and model config path
weightPath = os.path.sep.join([dir_yoloModel, "yolov3.weights"])
configPath = os.path.sep.join([dir_yoloModel, "yolov3.cfg"])

# load YOLO object detector trained on COCO dataset (80 class)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(dir_input_image)
(h, w) = image.shape[:2]


# determine only the output layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] -1] for i in net.getUnconnectedOutLayers()]

# construct a blob form the input image and then perform a forward 
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True, crop=False)
net.setInput(blob)

start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

#show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end-start))



#######################
## Visualization

# initialize lists of detected bounding boxes, confidence, and classID
boxes=[]
confidences=[]
classIDs=[]

# loop over each of the layer outputs
for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID confidence (i.e. probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        
        # filter out weak predictions by ensuring the detected 
        # probabiilty is greater than the minimum probability
        if confidence > confidence_thresh:
            # scale the bounding box corrdinates back relatvie to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y) coordinates of the bounding 
            # box followed by the boxes 'width and height
            box = detection[0:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            
            # use the center (x, y) coordinates to derive the top and 
            # left corner of the bounding box
            x = int(centerX-(width/2))
            y = int(centerY-(height/2))
            
            # update the list of bounding box coordinates, confidence, and
            # classID
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlap bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, nms_thresh)

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the idxs we are keeping
    for i in idxs.flatten():
        # extract the bounding box corrdinate
        (x, y) = (boxes[i][0], boxes[i][1])
        (wid, hei) = (boxes[i][2], boxes[i][3])
        
        # draw the box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x,y), (x+wid, y+hei), color, 2)
        text ="{}:{:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # print out the detected object and its confidence
        print(text)
        
# show the output
cv2.imshow("Image", image)
cv2.waitKey(0)


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:45:08 2019

A gentle guide to deep learning object detection
Project guilde provided by pyimagesearch.

Similar to the real-time object detection project, 
but in this script, a filter is added to filter out the unwanted object.

@author: jschang
"""


import numpy as np
import cv2 
import time
import imutils
from imutils.video import VideoStream
from imutils.video import FPS


dir_coffe_prototxt = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\filter-object-detection\\MobileNetSSD_deploy.prototxt.txt"
dir_coffe_model = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\filter-object-detection\\MobileNetSSD_deploy.caffemodel"
confidence_thresh = 0.5


# load the labels and generate color list for each label class
LABELS =  ["background", "aeroplane", "bicycle", "bird", "boat",
            	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            	"sofa", "train", "tvmonitor"]

IGNORE = ["background", "aeroplane", "bicycle", "bird", "boat",
            	"bus", "car", "cat", "chair", "cow", "diningtable",
            	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            	"sofa", "train", "tvmonitor"]q

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# read the network from local disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(dir_coffe_prototxt, dir_coffe_model)

# initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(scr=0).start()
time.sleep(2)
fps = FPS().start()

while(True):
    # grab the frame from the webcam and resize
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    # pre-process the frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                             0.007843, (300, 300), 127.5)
    # pass the blob to the net
    net.setInput(blob)
    detections = net.forward()
    
    # loop over the predictions
    for i in range(0, detections.shape[2]):
        # extract the confidence
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_thresh:
            # extract the class ID
            classid = int(detections[0, 0, i, 1])
            
            # if the predicted class id is in the set of the IGNORE
            # skip the detection
            if LABELS[classid] in IGNORE:
                continue
            
            # extract the (x, y) coordination
            box = detections[0, 0, i, 3:7] * np.asarray([w, h, w, h])
            (startX,startY, endX, endY) = box.astype("int")
            
            # localize the object
            text = "{}: {:2f}%".format(LABELS[classid], confidence * 100)
            color = [int(c) for c in COLORS[classid]]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            text_y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, text, (startX, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # quit when pressing 'q'
    if key == ord("q"):
        break
    
    # update FPS
    fps.update()


# stop the timer and display FPS information
fps.stop()
vs.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS:{:.2f}".format(fps.fps()))     
cv2.destroyAllWindows()

    
    
    
    
    
    
    
    
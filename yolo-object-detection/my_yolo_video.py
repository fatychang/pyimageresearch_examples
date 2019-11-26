# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:47:30 2019

YOLO object detection with OpenCV
Project guilde provided by pyimagesearch

This project is similar to my_yolo which forward a single image to the network
instead of a video. The concept is pretty much the same, the only differnce is
how you hanld a image or a video which consists a series of image (frame).

The input argument for this project includes the path of the video, YOLO model, 
and the directory for the output model. Becareful to add the file name as well
in the output directory.




@author: jschang
"""


import numpy as np
import imutils
import time
import cv2
import os


path_input_video = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\yolo-object-detection\\videos\\airport.mp4"
path_output_video = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\yolo-object-detection\\output\\out.avi"
path_YOLO_model = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\yolo-object-detection\\yolo-coco\\"
confidence_thresh = 0.5
nms_thresh=0.2

# load the COCO database categories
labelsPath = os.path.sep.join([path_YOLO_model, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([path_YOLO_model, "yolov3.weights"])
configPath = os.path.sep.join([path_YOLO_model, "yolov3.cfg"])

# load the YOLO model trained on COCO dataset (80 classes)
# and determin only the *output* layer names that we need from YOLO
print("[INOF] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] -1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(path_input_video)
writer = None
(W, H) = (None, None)


# try to determine the total munber of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in the video".format(total))

# an error occured while trying to determine the total number 
# of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
    
        
# loop over the frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    
    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break
    
    # if the frame dimension are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB = True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] the total runtime for one frame is {:.6f}".format(end-start))

    # initialize the lists of detected bounding boxes, confidence, and 
    # class IDs
    boxes=[]
    confidences=[]
    classIDs=[]
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            # extract the class ID and confidence (i.e probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            # filter out weak predictions by ensuring the detected 
            # probabilities is greater than the minimum probability
            if confidence > confidence_thresh:
                # scale the bounding box coordinate back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y) coordinates of the
                # bounding box followed by the boses' width and height
                box = detection[0:4]*np.array([W,H,W,H])
                (centerX, centerY, width, height) = box.astype("int")
                
                # use the center (x, y) coordinates to derive the top
                # and left corner of the bounding box
                x = int(centerX - width/2)
                y = int(centerY - height/2)
                
                # update the list of bounding box coordinates, 
                # confidences, and classIDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    # apply nms to supress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, nms_thresh)
            
    # ensure at least one detection exist
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extrach the bounding box coordinates
            (x, y) = boxes[i][0], boxes[i][1]
            (w, h) = boxes[i][2], boxes[i][3]
            
            # draw the bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2. rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = "{}:{:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2)
    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path_output_video, fourcc, 30, 
                                 (frame.shape[1], frame.shape[0]), True)
        

        
        # some information on processing single frame
        if total > 0:
            elap = (end-start)
            print("[INFO] the runtime for one frame is {:.6f}".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap*total))
    # write the output frame to disk
    writer.write(frame)
    

# release the file pointer
print("[INFO] cleaning up...")
writer.release()
vs.release()
                
            
                
            
        

        
        
        
        
        
        
        
        
        
        
        
        
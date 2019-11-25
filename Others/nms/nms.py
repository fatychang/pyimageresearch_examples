# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:24:37 2019

This script follows the sample from Pyimagesearch
the objective is to show the non-maxima-suppression (NMS) technique
which deals with multiple detections with a single object.

The post can be found in the following link.
https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

@author: jschang
"""

import numpy as np

# Felzenszwald method for nms
def non_max_suppression_slow(boxes, overlapThresh):
    # return empty list if there are no boxes
    if len(boxes)==0:
        return []
    
    # initialize the list of picked indexes
    pick=[]
    
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    print(y2)
    
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2-x1+1)*(y2-y1+1)
    idxs = np.argsort(y2)
    
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs)-1
        i = idxs[last]
        pick.append(i)
        
        suppress = [last] # suppress is the list which we want to ignore
        
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j=idxs[pos]
            
            # find the larget (x, y) coordinates for the start of
            # the boundary box and the smallest (x, y) coordinates
            # for the end of the bonding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            
            # compute the width and height of the bounding box
            w = max(0,xx2-xx1+1)
            h = max(0,yy2-yy1+1)
            
            # compute the ratio of the overlap between the computed 
            # bounding box and the bounding box in the area list
            overlap = float(w*h)/area[j]
#            print("%d, overlap=%f" %(j, overlap))
            
            # if there is sufficient overlap, suppress the current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
        
        # delete all indexes from the index list that are in the suprression list
        idxs = np.delete(idxs, suppress)
  
    return boxes[pick]
        




##### Testing code
## construct a list containing the image along with their respective boundary
#images = [("audrey H.jpg", np.array([
#        (67,10,809,815),
#        (6,14,885,820),
#        (32,35,846,787)]))]
#pick=[]

#overlapThresh = 0.3
#boxes = images[0][1]
#x1 = boxes[:, 0]
#y1 = boxes[:, 1]
#x2 = boxes[:, 2]
#y2 = boxes[:, 3]
#
#area = (x2-x1+1)*(y2-y1+1)
#idxs = np.argsort(y2)
#
#while len(idxs) > 0:
#    last = len(idxs)-1
#    i = idxs[last]
#    pick.append(i)
#    suppress = [last] # suppress is the list which we want to ignore
#    
#    for pos in range(0, last):
#        # grab the current index
#        print(pos)
#        j=idxs[pos]
#        
#    #    print("i = %d, j=%d, xi=%d, xj=%d" %(i, j, x1[i], x1[j]))
#        print("i = %d, j=%d, yi=%d, yj=%d" %(i, j, y1[i], y1[j]))
#    #    print("i = %d, j=%d, xi=%d, xj=%d" %(i, j, x2[i], x2[j]))
#    #    print("i = %d, j=%d, xi=%d, xj=%d" %(i, j, y2[i], y2[j]))
#        
#        # find the larget (x, y) coordinates for the start of
#        # the boundary box and the smallest (x, y) coordinates
#        # for the end of the bonding box
#        xx1 = max(x1[i], x1[j])
#        yy1 = max(y1[i], y1[j])
#        xx2 = min(x2[i], x2[j])
#        yy2 = min(y2[i], y2[j])
#        
#         # compute the width and height of the bounding box
#        w = max(0,xx2-xx1+1)
#        h = max(0,yy2-yy1+1)
#        
#        # compute the ratio of the overlap between the computed 
#        # bounding box and the bounding box in the area list
#        overlap = float(w*h)/area[j]
#        print("%d, overlap=%f" %(j, overlap))
#        
#        # if there is sufficient overlap, suppress the current bounding box
#        if overlap > overlapThresh:
#            suppress.append(pos)
#    
#    # delete all indexes from the index list that are in the suprression list
#    idxs = np.delete(idxs, suppress)
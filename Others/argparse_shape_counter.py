# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:07:46 2019

This script follows the example from pyimagesearch.com
The aim of this project

@author: jschang
"""

import argparse
import imutils
import cv2

# construct the argment parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the input image")
ap.add_argument("-o", "--output", required=True, 
                help="path to the output image")
args=vars(ap.parse_args())

# load the input image from disk
image = cv2.imread(args["input"])

# convert the image to grey scale, blur it and threshold it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# extract contours from the image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                      cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours and draw them on the input image
for c in cnts:
    cv2.drawContours(image, [c], -1, (0,0,255), 2)
    
# display the total number of shapes on the image
text="{} total shapes are found in the image".format(len(cnts))
cv2.putText(image, text, (10, 20), cv2.FONT_GERSHEY_SIMPLEX, 0.5,
            (0,0,255), 2)

# write the output image to disk
cv2.imwrite(args["output"],image)

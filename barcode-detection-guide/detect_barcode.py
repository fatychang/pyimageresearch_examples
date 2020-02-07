# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:04:46 2020

This script demonstrates the technique to detect the barcode 
using computer vision and image processing technique

A edge detection approach is implemented in this script with opencv

This sample is inspired by the post from pyinagesearch

@author: jschang
"""

#import the packages
import numpy as np
import cv2


#input arguments
dir_input_image = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\barcode-detection-guide\\detecting-barcodes-in-images\\images\\barcode_01.jpg"


#load the image via cv2
image = cv2.imread(dir_input_image)

#convert the image to gray scale
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#construct the gradient magnitude representation in the horizontal and vertical direction
#compute the Scharr gradient magnitude representation of the image in both x, y direction
gradX = cv2.Sobel(gray_img, ddepth= cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray_img, ddepth= cv2.CV_32F, dx=0, dy=1, ksize=-1)

#substract the y-gradient from the x-gradient
#left the region that have high horizontal and low vertical gradient
gradient = cv2.subtract(gradX, gradY)
#calculate absoulute values and converts the result to 8bit
gradient = cv2.convertScaleAbs(gradient)

#blur and threshold the image
#smooth out the high frequency noise in the gradient image
blurred = cv2.blur(gradient, (9,9))
#threshold the image to ignore the gradient less than 225
#set the ignored pixel to 0 (black), others to 255 (black)
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)


#morphological operation to close the gaps by constructing a closing kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#remove the small blobs by perfoming a series of erosions and dilations
closed_erosion = cv2.erode(closed, None, iterations=4)
closed_dilation = cv2.dilate(closed_erosion, None, iterations=4)

#find the contours in the thresholded image, then sort the contours
#by their area, keeping only the largest
(cnts, _) = cv2.findContours(closed_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse = True)[0]

#compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

#draw a bounding box around the detected barcode
cv2.drawContours(image, [box], -1, (0, 255,0), 3)


#display the images
cv2.imshow("image", image)
cv2.imshow("gray", gray_img)
cv2.imshow("gradient", gradient)
cv2.imshow("blurred", blurred)
cv2.imshow("threshold", thresh)
cv2.imshow("closed", closed)
cv2.imshow("erosion", closed_erosion)
cv2.imshow("dilation", closed_dilation)


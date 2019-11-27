# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:34:36 2019

Deep Learning with OpenCV
Project guilde provided by pyimagesearch

In this project, caffe is implemented to achieve object classification task.
It can identify the class of the object but it cannot be used to detect the
position of the object in the image (bounding box).

Since, caffe is only responsible to classify the object, the run time is much 
fast which usually takes 0.03 senconds to classify a single image.

@author: jschang
"""

import numpy as np
import cv2
import time

dir_input_image = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\caffe-object-detection\\images\\eagle.png"
dir_coffe_prototxt = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\caffe-object-detection\\bvlc_googlenet.prototxt"
dir_coffe_model = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\caffe-object-detection\\bvlc_googlenet.caffemodel"
dir_imageNet_labels = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\caffe-object-detection\\synset_words.txt"

# load the input image
image = cv2.imread(dir_input_image)

# load the class labels from disk
rows = open(dir_imageNet_labels).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]


# Our CNN model requires fixed spatial dimensions for our image images
# so we need to ensure it is resize to 224x224 pixels while performing
# mean subtraction (104, 117, 123) to normalize the input.
blob = cv2.dnn.blobFromImage(image, 1, (244, 244), (104, 117, 123))


# load serizlized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(dir_coffe_prototxt, dir_coffe_model)

# set the blob as input to the network and perform a forward-pass to obtain 
# the output classification
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.6f} seconds".format(end-start))

# sort the indexes of the probabilities in descending order and grab
# the top 5 predictions
idxs = np.argsort(preds[0])[::-1][:5]   #[::-1] means all item in the array, reverse

# loop over the top 5 predictions and display
for (i, idx) in enumerate(idxs):
    # draw the top prediction on the input image
    if i==0:
        text = "Label:{}, {:.2f}%".format(classes[idx], preds[0][idx]*100)
        cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    # display the prediction label and associated probability to the console
    print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
		classes[idx], preds[0][idx]))

# display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
    

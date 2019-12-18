# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:05:51 2019

This script implements a pre-trained model on Keras library

@author: jschang
"""

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import numpy as np
import cv2

input_image = 'D:\Jen\_Documents\eLearning\Computer Vision\pyimagesearch\images\dog1.jpg'
model_name = "resnet"


# define a dictionary that maps model names to their classes
MODELS={
        "vgg16":VGG16,
        "vgg19":VGG19,
        "inception":InceptionV3,
        "xception":Xception,
        "resnet":ResNet50        
}

# ensure a valid model name was supplied
if model_name not in MODELS:
    raise AssertionError("The model you want is invalid")


# initialize the input shape and the preprocessor
# (this might need to change based on the network)
inputShape=(224, 224)
preprocess = imagenet_utils.preprocess_input

# load the pre-trained network
# it may takes some time to download the files if it's
# your first time runing the script
print("[INFO] loading {} network...".format(model_name))
Network=MODELS[model_name]
model = Network(weights="imagenet")

# load the image with load_image
print("[INFO] loading and pre-processing image...")
image = load_img(input_image, target_size=inputShape)
image = img_to_array(image)

# expend the image array dimension from (inputShape[0], inputShape[1], 3)
# to (1, inputShape[0], inputShape[1], 3) so we can pass it through thenetwork
image = np.expand_dims(image, axis=0)

# pre-process the image using the appropriate function based on the model
image = preprocess(image)


# classify the image
print("[INFO] Classifying image with {}...".format(model_name))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)


# loop over the predictions and display the rank 5 predictions 
# and their probabilities to the terminal
for (i, (imageID, label, prob)) in enumerate (P[0]):
    print("{}. {}: {:.2f}".format(i+1, label, prob*100))
    
    
# display the image and draw the top prediction on the image
orig = cv2.imread(input_image)
(imagenetID, label, prob) = P[0][0]
text = "Label: {} {:.2f}".format(label, prob*100)
cv2.putText(orig, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imshow("result", orig)
cv2.waitKey(0)
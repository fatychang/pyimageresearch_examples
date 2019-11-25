# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:39:27 2019

This script follows the example from pyimagesearch.com
which talks about the argparse package.

'argparse' package allows the user to provide different inputs without changing the code.
In the case of computer vision, the inputs are mostly directories.
In the case of deep learning, the inputs are mosly model path or epoch counts.


The link to the post is as below:
https://www.pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/

@author: jschang
"""

import argparse

# construct the arguemnt parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, 
                 help="name of the user")
args = vars(ap.parse_args())

# display a friendly message to the user
print("Hi there {}, it's nice to meet you".format(args["name"]))

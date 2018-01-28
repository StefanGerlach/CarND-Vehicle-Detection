# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./output_images/main_image.png "Splash"

[![Youtube Link][image7]](https://youtu.be/iSw3WAGySTk "Udacity Self Driving Car ND Project 4 - Advanced Lane Finding")

In this repository I describe my approach to write a software pipeline that identifies other vehicles in a video file. The precise requirements of this project are:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier "Linear SVM classifier"
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Normalizing the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

In the following writeup I describe how I managed to solve the requirements.


##  Code description

This overview describes the project structure and modules:

* packages/batchgenerator.py containing a class for generating batches of images - from behaviorial project 
* packages/class_equalizer.py for collecting image paths of different classes + random sample to equal frequency + train test split
* packages/imageaugmentation.py class ImageAugmenter using aleju/imgaug
* packages/image_sliding_windows.py.py a class for splitting the image in rectangles
* packages/sliding_window_filter.py a function for filtering overlapping rectangles with heatmap
* packages/training.py.py complete script for grid search feature extractor parameter and training classifier 
* packages/classify.py containing a file loader that is dumped in training.py + classifier wrapper class
* main.py the main pipeline putting it all together for car identification in video stream


### Dependencies
This lab requires:

* [Image Augmentation by aleju/imgaug](https://github.com/aleju/imgaug)
* Labeled dataset for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)



```python

```

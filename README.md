# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./output_images/main_image.png "Splash"
[image2]: ./output_images/original_frame.png "Frame"
[image3]: ./output_images/frame_ROI.png "Frame ROI"
[image4]: ./output_images/frame_ROI_scale_0.5.png "Frame ROI Scaled 1"
[image5]: ./output_images/frame_ROI_scale_0.25.png "Frame ROI Scaled 2"
[image6]: ./output_images/frame_ROI_scale_0.1875.png "Frame ROI Scaled 3"
[image7]: ./output_images/grid_no_overlap.png "Grid"
[image8]: ./output_images/features_unscaled.png "Feature Vector unscaled"
[image9]: ./output_images/features_scaled.png "Feature Vector scaled"
[image10]: ./output_images/raw_detections.png "Grid detections"
[image11]: ./output_images/heatmap_detections.png "Grid heatmap"
[image12]: ./output_images/binarized_heatmap_detections.png "Grid heatmap binarization"
[image13]: ./output_images/filtered_detections.png "Filtered Grid detections"
[image14]: ./output_images/original_frame_detections.png "Frame with car detection"


[![Youtube Link][image15]](https://youtu.be/iSw3WAGySTk "Udacity Self Driving Car ND Project 4 - Advanced Lane Finding")

In this repository I describe my approach to write a software pipeline that identifies other vehicles in a video file. The precise requirements of this project are:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier "Linear SVM classifier".
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector.
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


## Training Pipeline

In the following section I want to explain how I trained a Support Vector Machine classifier to distinguish between vehicles and non-vehicles images. To implement the pipeline, I oriented myself on code-examples of the Udacity course material. 

My [training pipeline](packages/training.py) consists of the following steps:


### Data Reading
* Read directories with images, ordered by classes.
* Randomly dublicate images until both classes are equally distributed.
* Random split the dataset into training and validation images.

I created the class **ClassEqualizer** to do this. Within this class, glob is used for directory reading and Scikit-learn for the train-test-split with 5 % for validation. After splitting, I arrange the datasets as tuples of lists: (train_x, train_y) and (val_x, val_y), where the labels have been translated into integers by Scikit-learn **LabelEncoder**.


### Data Preprocessing and Feeding
* Use Batchgenerator to:
  * Shuffle images
  * Preprocess and augment the images

For feeding the images to the feature extractor in the next step, I recylced my [**BatchGenerator**](packages/batchgenerator.py) class from Behavioral Cloning project. This enables the usage of [**ImageAugmenter**](packages/imageaugmentation.py) class to augment the images. This will help to compensate that the training images are from time series and this may cause overfitting during training.

I enabled the following augmentation functions for this project:

| Function | Probability to occur | Parameter |
| :-------- | :-------------------- | :--------- |
| Gaussian Noise | 0.2 | / |
| Simplex Noise for local intensity shift | 0.2 | multiply pixel values with 0.7 |
| Average Blur | 0.2 | Kernel size 3 |
| Contrast Normalization | 0.2 | Alpha = 0.75 |
| Contrast Normalization | 0.2 | Alpha = 1.25 |
| Flip Left-Right | 0.5 | / |
| Random Cropping | 0.2 | Max border cut (4, 4, 4, 4) |
| Multiply image | 0.2 | Factor 0.75 |
| Multiply image | 0.2 | Factor 1.25 |


### Feature Extraction

The next step includes the extraction of features from the training and validation images. To find the best features I created the class [**SlidingWindowFeatureExtractor**](packages/feature_extraction.py) which is instantiated with the complete set of parameters as arguments:

| Argument | Description |
| :------- | :---------- |
| bin_spatial_size | To use raw pixel data with binning. |
| bin_spatial_color_cvt | E.g. cv2.COLOR_RGB2HSV to use different color spaces for spatial features. |
| color_channel_hist_bins | If not None, it's the number of color histogram bins. |
| color_hist_color_cvt | E.g. cv2.COLOR_RGB2HSV to use different color space for color histogram features |
| hog_compute | Bool: True for computing Histogram of Oriented Gradients |
| hog_color_cvt | E.g. cv2.COLOR_RGB2HSV to use different color space for HOG features |
| hog_channels | List of channels where HOG features should be computed |
| hog_orient | The number of HOG orientations |
| hog_px_per_cell | The number of pixels per HOG cell |
| hog_cells_per_blk | The number of cells per HOG blocks | 
| hog_norm | To use normalization (hog transform_sqrt) |

So I created a list for every possible parameter and created all possible permutations in a big, nested for-loop. For every parameter-set that is created this way (**Grid-search**), an instance of the **SlidingWindowFeatureExtractor** is created. 

With the help of the **BatchGenerator**, I iterated n-times (n=2) over the complete training-set and invoked every created instance of the SlidingWindowFeatureExtractor class to compute the features.

After this process, for every parameter-set I have a list of training features, training labels, validation features and validation labels. These information are stored in a dictionary-structure and is stored on the filesystem with the help of the **pickle**-API (*feature_permutation_checkpoint.picklefile*).


### Feature Scaling

Since the features have different value range, a preprocessing step on the features is needed. **This step is essential to train a classifier on them**. With the help of the Scikit-learn class **StandardScaler**, all features can be zero-centered and scaled to have unit variance. 

![Unscaled features][image8] ![Scaled features][image9]

So, for every parameter-set I called the **fit()**-function on the training features and created a fitted StandardScaler-object this way. The training features and validation features are transformed with this StandardScaler, for every parameter-set-features individually.


### Training a classifier

So at this point of the training pipeline there are different parameter-sets, each coupled with a list of training and validation features. With a simple iteration over these parameter-sets I created the following classifiers:
* Logistic Regression Classifier
* DecisionTree Classifier
* AdaBoost Classifier
* SVM

For every classifier I called the **fit()**-function on the current training features plus labels and **score()** on the validation features plus labels. This way, I could evaluate the parameter-set and the classifiers, since **score()** returns the mean accuracy over the validation set.


### Best combination of features and classifier

I ended up that the following combination of feature extraction parameters and classifer with mean accuracy of 99.44 %:

| Parameter | Value |
| :------- | :---------- |
| classifier | SVM |
| bin_spatial_size | (32, 32) |
| bin_spatial_color_cvt | Keep RGB |
| color_channel_hist_bins | 32 |
| color_hist_color_cvt | RGB2HSV |
| hog_compute | True |
| hog_color_cvt | Keep RGB |
| hog_channels | All |
| hog_orient | 8 |
| hog_px_per_cell | (8, 8) |
| hog_cells_per_blk | (2, 2) | 
| hog_norm | True |


All tested combinations are dumped to filesystem, for later usage during video frame analysis. Every combination consists of the following objects that can be read from filesystem:

* StandardScaler (fitted on the respective training features)
* LabelEncoder
* FeatureExtractor (SlidingWindowFeatureExtractor) object
* Classifier objects (as list of tuples: classifier object, mean accuracy)

To read the file I use my class [**CompetitionClassifierLoader**](packages/classify.py)



## Video Frame Processing Pipeline

To use the trained classifier



```python

```




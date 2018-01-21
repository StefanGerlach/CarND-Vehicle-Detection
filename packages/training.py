import numpy as np
import cv2

from sklearn import tree, ensemble, svm
from sklearn.preprocessing import LabelEncoder

from packages.class_equalizer import ClassEqualizer
from packages.batchgenerator import BatchGenerator
from packages.imageaugmentation import ImageAugmenter

"""
Training script
===========================================================
for testing several different classifiers from Scikit-learn
and convolutional neuronal networks with keras.
"""

class_eq = ClassEqualizer()

# Create random sampled sets for training and validation with
# same class occurrence of 'non-vehicle' and 'vehicle' within each individual set.
training_dict, validation_dict = class_eq.get_splitted_and_classnormed_filelist()

# Create LabelEncoder for numbers instead of class names
label_enc = LabelEncoder()
label_enc.fit([k for k in training_dict])

# Re-arrange the dictionaries for better usability in the next section.
train_x = []
train_y = []

val_x = []
val_y = []
for i, c_set in enumerate([training_dict, validation_dict]):
    for k in c_set:
        if i == 0:
            train_x += c_set[k]
            train_y += np.repeat(label_enc.transform([k])[0], len(c_set[k]), axis=0).tolist()
        else:
            val_x += c_set[k]
            val_y += np.repeat(label_enc.transform([k])[0], len(c_set[k]), axis=0).tolist()


# Define the function to load the images from file system
def extract_xy_fn(x):
    img = cv2.imread(x[0])
    return img, x[1], False  # the third return value means the 'mirrorable' flag from Behavioral Cloning Project


# Define the preprocessing function
def preprocessing(x):
    img = cv2.resize(x, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    return img


# Define the Image Augmentation methods
train_augmenter = ImageAugmenter()

train_augmenter.add_gaussian_noise(prob=0.1)
train_augmenter.add_simplex_noise(prob=0.1, multiplicator=0.7)
train_augmenter.add_average_blur(prob=0.1, kernel=7)
train_augmenter.add_contrast_normalization(prob=0.1, alpha=0.5)
train_augmenter.add_contrast_normalization(prob=0.1, alpha=1.5)
train_augmenter.add_flip_left_right(prob=0.5)
train_augmenter.add_random_crop(prob=0.1)
train_augmenter.add_multiply(prob=0.1, mul=0.5)
train_augmenter.add_multiply(prob=0.1, mul=1.5)

# Create a Batch-Generator to have the possibility to augment and preprocess the images
training_batchgen = BatchGenerator(batch_size=1, n_classes=len(label_enc.classes_), dataset=list(zip(train_x, train_y)),
                                   augmentation_fn=train_augmenter.augment,
                                   preprocessing_fn=preprocessing,
                                   extract_xy_fn=extract_xy_fn,
                                   shuffle=True, seed=1337)

for i in range(training_batchgen.n):
    x, y = training_batchgen.custom_next()
    cv2.imwrite('test_'+str(i)+'.png', x[0, :, :, :])
    if i > 100:
        break

# Create HOG Descriptors for each set with some different parameters to
# 'grid-search' the best parameters for the HOG Descriptor.


i = 100

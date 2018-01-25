import numpy as np
import cv2
import pickle
import os

from tqdm import tqdm
from sklearn import tree, ensemble, svm, linear_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.feature import hog

from packages.class_equalizer import ClassEqualizer
from packages.batchgenerator import BatchGenerator
from packages.imageaugmentation import ImageAugmenter

"""
Training script
===========================================================
for testing several different classifiers from Scikit-learn
and convolutional neuronal networks with keras.
"""

feature_file_name = 'feature_permutation_checkpoint.picklefile'
score_file_name = 'complete_classifier_competition.picklefile'

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

# convert to uint8
val_y = np.uint8(val_y)

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

# And the Batch-Generator for the validation data
val_batchgen = BatchGenerator(batch_size=1, n_classes=len(label_enc.classes_), dataset=list(zip(val_x, val_y)),
                              augmentation_fn=None,
                              preprocessing_fn=preprocessing,
                              extract_xy_fn=extract_xy_fn,
                              shuffle=False)

# Debug writing of augmented images
if False:
    for i in range(training_batchgen.n):
        x, y = training_batchgen.custom_next()
        cv2.imwrite('test_'+str(i)+'_C'+str(y)+'.png', x[0, :, :, :])
        if i > 100:
            break

# Create HOG Descriptors for each set with some different parameters to 'grid-search' the best parameters
# for the HOG Descriptor. So lets create some ranges for num orientations, pixels_per_cell, cells_per_block and norm:
normalize = [True, False]
orientations = [8, 12, 16]
pixels_per_cell = [(8, 8), (16, 16), (32, 32)]
cells_per_block = [(1, 1), (2, 2)]

# Lets push this into an list of dictionaries, so each permutation has its own dictionary
permutations = [dict({'orientation': orient, 'pixels_per_cell': px_per_cell,
                      'cells_per_block': cells_pb, 'normalize': norm,
                      'train_features': [], 'val_features': [],
                      'train_labels': [], 'val_labels': []})
                for orient in orientations
                for px_per_cell in pixels_per_cell
                for cells_pb in cells_per_block
                for norm in normalize]

# Now we have to compute the HOG Features for train_x and val_x for every parameter set!
# Check if we already have computed them:
if os.path.isfile(feature_file_name):
    # Yes, load them !
    permutations = pickle.load(open(feature_file_name, 'rb'))
    print('Loaded ' + str(len(permutations)) + ' parameter sets with training and val feature vectors.')
else:
    # Take all the training images and create the features:
    print('Computing feature vectors..')
    for i in tqdm(range(training_batchgen.n)):
        x, y = training_batchgen.custom_next()
        # Go over parameter sets
        for parameter_set in permutations:
            feature_descriptor = hog(image=cv2.cvtColor(np.uint8(x[0]), cv2.COLOR_RGB2GRAY),
                                     orientations=parameter_set['orientation'],
                                     pixels_per_cell=parameter_set['pixels_per_cell'],
                                     cells_per_block=parameter_set['cells_per_block'],
                                     transform_sqrt=parameter_set['normalize'],
                                     visualise=False)

            parameter_set['train_features'].append(feature_descriptor)
            parameter_set['train_labels'].append(y)

    for i in tqdm(range(val_batchgen.n)):
        x, y = val_batchgen.custom_next()
        # Go over parameter sets
        for parameter_set in permutations:
            feature_descriptor = hog(image=cv2.cvtColor(np.uint8(x[0]), cv2.COLOR_RGB2GRAY),
                                     orientations=parameter_set['orientation'],
                                     pixels_per_cell=parameter_set['pixels_per_cell'],
                                     cells_per_block=parameter_set['cells_per_block'],
                                     transform_sqrt=parameter_set['normalize'],
                                     visualise=False)

            parameter_set['val_features'].append(feature_descriptor)
            parameter_set['val_labels'].append(y)

    print('Dumping feature vectors..')
    pickle.dump(permutations, open(feature_file_name, 'wb'))

# We have the train and the validation features computed by now.
# But one step is missing: Scaling ! Lets use the sklearn StandardScaler to
# center the features around zero and a way, so they have unit variance !

# This is done for each setup (permutation) individually.
print('Scaling feature vectors..')
for parameter_set in tqdm(permutations):
    scaler = StandardScaler()
    scaler.fit_transform(parameter_set['train_features'])
    parameter_set['val_features'] = scaler.transform(parameter_set['val_features'])

# Now we want to know how these combinations of parameters of the HOG Descriptor performs
# for every tested classifier method: tree, ensemble, svm
print('Testing classifier..')
for parameter_set in permutations:
    for classifier in [(linear_model.LogisticRegression(), 'LogisticRegression'),
                       (tree.DecisionTreeClassifier(), 'DecisionTree'),
                       (ensemble.AdaBoostClassifier(), 'AdaBoost'),
                       (svm.SVC(), 'SVM')]:
        # Fit the classifier on the training-data
        classifier[0].fit(np.array(parameter_set['train_features']),
                          np.array(parameter_set['train_labels']).ravel())

        # Then evaluate the classifier by calling score()
        mean_acc = classifier[0].score(np.array(parameter_set['val_features']),
                                       np.array(parameter_set['val_labels']).ravel())

        print(classifier[1]+' scored: '+str(mean_acc))

        # The competition-key of classifiers into container
        results_key = 'competition'
        if results_key not in parameter_set:
            parameter_set[results_key] = {}

        # Push the result and the classifier object into the competition-container
        if classifier[1] not in parameter_set[results_key]:
            parameter_set[results_key][classifier[1]] = (classifier, mean_acc)

print('Dumping classifiers competition..')
pickle.dump(permutations, open(score_file_name, 'wb'))


from keras.preprocessing.image import ImageDataGenerator
import imgaug.augmenters as iaa
import imgaug.parameters as iap

import random as rnd
import numpy as np


class ImageAugmenter(object):
    def __init__(self):
        self._keras_augmenter = None
        self._sequential_augmentation = None

    def add_keras_augmenter(self, data_generator: ImageDataGenerator):
        self._keras_augmenter = data_generator

    def add_gaussian_noise(self, prob=0.5, scale=25.5):
        if self._sequential_augmentation is None:
            self._sequential_augmentation = iaa.Sequential()

        self._sequential_augmentation.add(iaa.Sometimes(prob,
                                                        iaa.AdditiveGaussianNoise(scale=scale)))

    def add_median(self, prob=0.5, k=5):
        if self._sequential_augmentation is None:
            self._sequential_augmentation = iaa.Sequential()

        self._sequential_augmentation.add(iaa.Sometimes(prob, iaa.MedianBlur(k=k)))

    def add_invert(self, prob=0.5):
        if self._sequential_augmentation is None:
            self._sequential_augmentation = iaa.Sequential()

        self._sequential_augmentation.add(iaa.Sometimes(prob, iaa.Invert(p=1.0)))

    def add_simplex_noise(self, prob=0.5, multiplicator=0.7):
        if self._sequential_augmentation is None:
            self._sequential_augmentation = iaa.Sequential()

        self._sequential_augmentation.add(iaa.Sometimes(prob,
                                                        iaa.SimplexNoiseAlpha(iaa.Multiply(multiplicator),
                                                                              upscale_method='linear')))

    def add_random_crop(self, prob=0.5, max_cropping=(32, 32, 32, 32)):
        if self._sequential_augmentation is None:
            self._sequential_augmentation = iaa.Sequential()

        self._sequential_augmentation.add(iaa.Sometimes(prob, iaa.Crop(px=(
            iap.DiscreteUniform(0, max_cropping[0]),
            iap.DiscreteUniform(0, max_cropping[1]),
            iap.DiscreteUniform(0, max_cropping[2]),
            iap.DiscreteUniform(0, max_cropping[3])
        ))))

    def add_flip_left_right(self, prob=0.5):
        if self._sequential_augmentation is None:
            self._sequential_augmentation = iaa.Sequential()

        self._sequential_augmentation.add(iaa.Sometimes(prob, iaa.Fliplr(p=1.0)))

    def add_multiply(self, prob=0.5, mul=0.5):
        if self._sequential_augmentation is None:
            self._sequential_augmentation = iaa.Sequential()

        self._sequential_augmentation.add(iaa.Sometimes(prob, iaa.Multiply(mul=mul)))

    def add_average_blur(self, prob=0.5, kernel=3):
        if self._sequential_augmentation is None:
            self._sequential_augmentation = iaa.Sequential()

        self._sequential_augmentation.add(iaa.Sometimes(prob, iaa.AverageBlur(k=kernel)))

    def add_coarse_dropout(self, prob=0.5, size_percentage=0.2):
        if self._sequential_augmentation is None:
            self._sequential_augmentation = iaa.Sequential()

        self._sequential_augmentation.add(iaa.Sometimes(prob, iaa.CoarseDropout(p=0.1, size_percent=size_percentage)))

    def add_contrast_normalization(self, prob=0.5, alpha=1.5):
        if self._sequential_augmentation is None:
            self._sequential_augmentation = iaa.Sequential()

        self._sequential_augmentation.add(iaa.Sometimes(prob, iaa.ContrastNormalization(alpha=alpha)))

    def augment(self, x, y, mirrorable):
        # flipping is done here with resp flipped y
        rand_num = rnd.randint(0, 9)
        if rand_num >= 5 and mirrorable:
            x = np.flip(x, axis=1)
            y = -y

        if self._keras_augmenter is not None:
            x = self._keras_augmenter.random_transform(x)

        if self._sequential_augmentation is not None:
            x = self._sequential_augmentation.augment_image(x)

        return x, y

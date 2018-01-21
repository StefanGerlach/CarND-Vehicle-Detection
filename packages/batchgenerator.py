import numpy as np
import keras.preprocessing.image as kgenerators
import random


class BatchGenerator(kgenerators.Iterator):
    """ This class implements a simple batch generator. """

    def __init__(self,
                 batch_size,
                 n_classes,
                 dataset,
                 augmentation_fn=None,
                 preprocessing_fn=None,
                 extract_xy_fn=None,
                 shuffle=True,
                 seed=1337):

        self._x = dataset
        self._augmentation_fn = augmentation_fn
        self._preprocessing = preprocessing_fn

        if extract_xy_fn is None:
            extract_xy_fn = (lambda e: e.load_image_and_label())

        self._extract_xy_fn = extract_xy_fn

        self._batch_size = batch_size
        self._num_classes = n_classes

        super().__init__(n=len(dataset), batch_size=batch_size, shuffle=shuffle, seed=seed)

    def custom_next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):

        elements = np.take(self._x, index_array, axis=0)
        batch_x = []
        batch_y = []

        # Load images, preprocess and augment
        for i in range(len(elements)):

            # Extract function may be a list. Choose 1 randomly
            if isinstance(self._extract_xy_fn, list):
                extract_fn = random.choice(self._extract_xy_fn)
            else:
                extract_fn = self._extract_xy_fn

            image, target, mirrorable = extract_fn(elements[i])

            # Do preprocessing if function is set
            if self._preprocessing is not None:
                image = self._preprocessing(image)

            # Do augmentation if function is set
            if self._augmentation_fn is not None:
                image, target = self._augmentation_fn(image, target, mirrorable)

            batch_x.append(image)
            batch_y.append(target)

        assert len(batch_x) == len(batch_y)
        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

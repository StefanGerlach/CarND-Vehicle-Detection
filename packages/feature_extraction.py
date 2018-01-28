import cv2
import numpy as np
from skimage.feature import hog as HogFeatures


class SlidingWindowFeatureExtractor(object):

    def __init__(self,
                 bin_spatial_size=(32, 32),
                 bin_spatial_color_cvt=None,
                 color_channel_hist_bins=32,
                 color_hist_color_cvt=None,
                 hog_compute=True,
                 hog_color_cvt=None,
                 hog_channels=None,
                 hog_orient=8,
                 hog_px_per_cell=(32, 32),
                 hog_cells_per_blk=(2, 2),
                 hog_norm=True
                 ):
        self._bin_spatial_size = bin_spatial_size
        self._bin_spatial_color_space = bin_spatial_color_cvt

        self._color_hist_bins = color_channel_hist_bins
        self._color_hist_color_space = color_hist_color_cvt

        self._hog_compute = hog_compute
        self._hog_color_cvt = hog_color_cvt
        self._hog_channels = hog_channels
        self._hog_orient = hog_orient
        self._hog_px_per_cell = hog_px_per_cell
        self._hog_cell_per_blk = hog_cells_per_blk
        self._hog_norm = hog_norm

    def compute_features(self, img):
        feature_list = []
        # Binning the raw pixel data
        if self._bin_spatial_size is not None:
            feature_list.append(SlidingWindowFeatureExtractor.bin_spatial(img,
                                                                          self._bin_spatial_color_space,
                                                                          self._bin_spatial_size))
        # Histogram of colors
        if self._color_hist_bins is not None:
            feature_list.append(SlidingWindowFeatureExtractor.color_hist(img,
                                                                         self._color_hist_color_space,
                                                                         self._color_hist_bins))
        # Histogram of oriented gradients
        if self._hog_compute is True:
            feature_list.append(SlidingWindowFeatureExtractor.hog_features(img,
                                                                           self._hog_color_cvt,
                                                                           self._hog_orient,
                                                                           self._hog_px_per_cell,
                                                                           self._hog_cell_per_blk,
                                                                           self._hog_norm,
                                                                           self._hog_channels))
        # Return the features
        return np.concatenate(feature_list).astype(np.float64)

    @staticmethod
    def _check_dims(img, num_dimensions=3):
        if len(img.shape) != num_dimensions:
            raise TypeError("Image tensor has not " + str(num_dimensions) + "dimensions!")

    @staticmethod
    def _convert_color_space(img, color_space):
        if color_space is not None:
            return cv2.cvtColor(img, color_space)
        return img

    @staticmethod
    def bin_spatial(img, color_space=None, size=(32, 32)):
        img = SlidingWindowFeatureExtractor._convert_color_space(img, color_space)
        img = cv2.resize(img, size)
        return img.ravel()

    @staticmethod
    def color_hist(img, color_space=None, n_bins=32, bins_range=(0, 256)):
        # Check if input is 3-Dimensional
        SlidingWindowFeatureExtractor._check_dims(img)
        img = SlidingWindowFeatureExtractor._convert_color_space(img, color_space)

        channel_hists = []
        for i in range(int(img.shape[2])):
            channel_hists.append(np.histogram(img[:, :, i], bins=n_bins, range=bins_range)[0])

        return np.concatenate(channel_hists)

    @staticmethod
    def hog_features(img, color_space=None, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(2, 2), normalize=True, hog_channels=None):
        # Check if input is 3-Dimensional
        SlidingWindowFeatureExtractor._check_dims(img)
        img = SlidingWindowFeatureExtractor._convert_color_space(img, color_space)

        hog_features = []
        for channel in range(int(img.shape[2])):
            if hog_channels is None or channel in hog_channels:
                hog_features.append(HogFeatures(img[:, :, channel],
                                                orientations=orientations,
                                                pixels_per_cell=pixels_per_cell,
                                                cells_per_block=cells_per_block,
                                                visualise=False,
                                                transform_sqrt=normalize,
                                                feature_vector=True).ravel())
        return np.concatenate(hog_features)

    def compute_on_windows_naiv(self, img, sliding_windows, feature_scaler, preproc_func=None):
        """
        This function extracts features over several sliding windows of the input image.
        :param img: The image.
        :param feature_scaler: Optional Scaler for computed feature vector.
        :param preproc_func: Optional Preprocessing function for ROI-images.
        :param sliding_windows: A dictionary with rects e.g. { shape: [rects], shape2: [rects], ..}
        :return:
        """
        for rect_shape in sliding_windows:
            rect_list = sliding_windows[rect_shape]
            rect_features = []

            for rect in rect_list:
                # Extract roi from image
                sub_img = img[rect[0][1]: rect[1][1], rect[0][0]: rect[1][0]]
                if sub_img.shape[0] <= 0 or sub_img.shape[1] <= 0:
                    rect_features.append([])
                    continue

                # Preprocess the image
                if preproc_func is not None:
                    sub_img = preproc_func(sub_img)

                # Generate features
                features = self.compute_features(sub_img)

                # Scale the features
                if feature_scaler is not None:
                    features = feature_scaler.transform([features.ravel()])

                rect_features.append(features)

            # Push that to the destination container
            sliding_windows[rect_shape] = (rect_list, rect_features)

        return sliding_windows

from skimage.feature import hog


class SlidingWindowHOGDescriptor(object):
    """
    This class wraps the sk-learn HOG interface for efficient
    computation over sliding windows.
    """
    def compute_on_windows(self, img, normalize, orientations, pix_per_cell, cells_per_block, sliding_windows):
        """
        This function extracts HOG features over several sliding windows.
        :param img:
        :param pix_per_cell:
        :param cells_per_block:
        :param normalize:
        :param orientations:
        :param sliding_windows: A dictionary with rects e.g. { shape: [rects], shape2: [rects], ..}
        :return:
        """
        feature_array = hog(img,
                            orientations=orientations,
                            pixels_per_cell=(pix_per_cell, pix_per_cell),
                            cells_per_block=(cells_per_block, cells_per_block),
                            visualise=False,
                            feature_vector=False,
                            transform_sqrt=normalize)




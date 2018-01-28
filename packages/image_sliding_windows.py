"""
This file includes the class to create subimages out of one big image.
"""
import numpy as np
import random as rnd
import cv2


class SlidingWindows(object):
    def __init__(self, image_shape):
        self._image_shape = image_shape

    def create_sliding_window_positions(self, min_shape, max_shape, num_scales, overlap_perc=0.5):
        """
        This function creates a nice centered grid of slided windows.
        :param min_shape: Min window shape size (with, height).
        :param max_shape: Max window shape size (width, height).
        :param num_scales: How many zooms there should be (>= 2).
        :param overlap_perc: Overlap. 1.0 is no overlap. 0.5 is half overlap.
        :return: A dictionary of shapes and rects { shape_0: [rects], shape_1: [rects], ..}
        """
        debug_paint = False

        # List with all shapes
        shape_sizes = [min_shape, max_shape] if min_shape != max_shape else [min_shape]

        # Dictionary with shapes and respective rectangles in grid
        grids = {}

        # Check for steps in between
        num_scales = np.max([0, num_scales])
        if num_scales > 2:
            step_x = (max_shape[0] - min_shape[0]) // (num_scales - 1)
            step_y = (max_shape[1] - min_shape[1]) // (num_scales - 1)
            for i in range(num_scales - 2):
                shape_sizes.append((min_shape[0] + ((i + 1) * step_x), min_shape[1] + ((i + 1) * step_y)))

        # Refresh the result structure
        for shape in shape_sizes:
            grids[shape] = []

        # Go over every shape_size
        for c, win_size in enumerate(shape_sizes):
            # Check how often and in what distance to put that size in the image.
            fits = []
            offs = []
            # The following values are calculated to center the 'grid' in the image.
            # I do this for a nice looking and well sampled grid!

            # Overlaps is (for each dimension) the little part of rectangles, that overlap
            overlaps = [int(win_size[0] * (1.0 - overlap_perc)), int(win_size[1] * (1.0 - overlap_perc))]
            # Overlapped_dims is (for each dimension) the larger part of the rectangles, that are not overlapping
            overlapped_dims = [int(win_size[0] * overlap_perc), int(win_size[1] * overlap_perc)]
            for i in range(len(win_size)):
                fits.append(self._image_shape[abs(i-1)] // overlapped_dims[i])
                offs.append(((self._image_shape[abs(i-1)] - fits[-1] * (overlapped_dims[i]))-overlaps[i]) // 2)
                # offs.append(0)

            # DEBUG paint it into such an image !
            if debug_paint:
                img = np.zeros(shape=(self._image_shape[0], self._image_shape[1], 1), dtype=np.uint8)

            # Create the actual rectangles !
            for fx in range(fits[0]):
                for fy in range(fits[1]):
                    cx = min(max(int(offs[0] + (fx * int(win_size[0]) * overlap_perc)), 0), self._image_shape[1] - 1)
                    cy = min(max(int(offs[1] + (fy * int(win_size[1]) * overlap_perc)), 0), self._image_shape[0] - 1)
                    nx = min(max(int(cx + win_size[0]), 0), self._image_shape[1] - 1)
                    ny = min(max(int(cy + win_size[1]), 0), self._image_shape[0] - 1)

                    # Push to result container
                    grids[win_size].append(((cx, cy), (nx, ny)))

                    # DEBUG paint it into such an image !
                    if debug_paint:
                        # Random color and thickness
                        thickness = int(rnd.uniform(2, 2))
                        color = int(rnd.uniform(255, 255))
                        cv2.rectangle(img, (cx, cy), (nx, ny), color, thickness)

            # DEBUG paint it into such an image !
            if debug_paint:
                cv2.imwrite('test_' + str(c) + '.png', img)

        return grids




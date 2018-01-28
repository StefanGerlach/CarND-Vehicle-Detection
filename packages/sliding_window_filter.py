import numpy as np
import cv2


def filter_detections(rectangles, img_size, heatmap_threshold=3, rect_overlap_thesh=0.75):
    # Create heatmap
    heatmap = np.zeros(shape=(img_size[0], img_size[1], 1), dtype=np.uint8)
    heatmap_bin = np.zeros_like(heatmap)
    for rect in rectangles:
        heatmap_bin.fill(0)
        cv2.rectangle(heatmap_bin, (rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), 1, -1)
        heatmap = heatmap + heatmap_bin

    # Binarize the heatmap by thresholding
    heatmap_bin[heatmap > heatmap_threshold] = 255

    # Finding the contours to separate the heatmap-detections
    _, contours, _ = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect all rectangles that overlap with a shared contour
    rect_collections = []
    contour_mask = np.zeros(shape=(img_size[0], img_size[1], 1), dtype=np.uint8)
    rectang_mask = contour_mask.copy()
    for contour in contours:
        contour_mask.fill(0)
        contour_mask = cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        collection = []
        for rect in rectangles:
            rectang_mask.fill(0)
            cv2.rectangle(rectang_mask, rect[0], rect[1], 255, -1)
            if np.count_nonzero(rectang_mask & contour_mask) > (np.count_nonzero(rectang_mask)*rect_overlap_thesh):
                collection.append(rect)
        if len(collection) > 0:
            rect_collections.append(collection)

    result_rects = []
    for rect_col in rect_collections:
        mean_pt1 = np.min(np.array(rect_col), axis=0)[0]
        mean_pt2 = np.max(np.array(rect_col), axis=0)[1]
        result_rects.append(((int(mean_pt1[0]), int(mean_pt1[1])), (int(mean_pt2[0]), int(mean_pt2[1]))))

    # Post-Filtering
    if heatmap_threshold > 0:
        result_rects = filter_detections(result_rects, img_size, 0, 0.0)
    return result_rects

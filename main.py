"""
In main.py the complete pipeline will be included.
Here the single modules will be called that do:

 - Camera Calibration
 - Perspective Image Transformation for Bird-Eye-View
 - Image Color Conversion and Channel Extraction
 - Image Gradient Calculation
 - Combined Thresholding techniques
 - Lane Detection
 - Filtering
 - Visualisation

"""

import os
import cv2
import random as rnd
import numpy as np

from moviepy.editor import ImageSequenceClip
from packages.image_sliding_windows import SlidingWindows as SlidingWin
from packages.classify import GridClassifier, CompetitionClassifierLoader
from packages.sliding_window_filter import filter_detections

""" Definition of some globals """

# Competition Filename
competition_filename = 'packages/complete_classifier_competition.picklefile'

# For intermediate outputs
output_path = 'output_images'
output_video_frames = 'video_frames'

# For Video output
output_video_file = 'output_project_video.mp4'
save_output_video = True

# Video files
video_file = 'project_video.mp4'

""" Definition of ROI """
image_height_roi = [300, -100]
image_scale_factor = 0.5
image_sliding_window_scales = [1.0, 0.5, 0.25]


""" Define the preprocessing function """
def preprocessing(x):
    x = cv2.resize(x, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
    if x.shape[2] > 2:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x


""" Create Grid-Generator, HOG Extractor and Classifier """
sliding_win_gen = None

loader = CompetitionClassifierLoader(competition_filename)
sliding_cls = GridClassifier(loader.classifier)

# Open video file
if not os.path.isfile(video_file):
    raise FileNotFoundError("Video file not found.")

clip = cv2.VideoCapture(video_file)

# Iterate all frames of the video
frame_id = 0
car_detections = []

while clip.isOpened():
    _, frame = clip.read()
    if frame is None:
        break

    detect_vehicles = frame_id % 10 == 0
    frame_id += 1

    # if frame_id < 305:
    #     continue

    # Remember the original image
    frame_src = frame.copy()

    if detect_vehicles:
        # Apply ROI
        frame = frame[image_height_roi[0]: image_height_roi[1], :, :]

        # Apply image scaling
        frame = cv2.resize(frame, (int(frame.shape[1] * image_scale_factor), int(frame.shape[0] * image_scale_factor)))

        # Get the Feature Extractor !
        feature_extractor = loader.feature_extractor

        # Get the Scaler !
        feature_scaler = loader.scaler

        # Classify ROI images on different scales
        car_rectangles = []
        for scale in image_sliding_window_scales:
            scaled_frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

            # Create Grid-Generator
            sliding_win_gen = SlidingWin(scaled_frame.shape)

            # Create the grid rectangles
            grid = sliding_win_gen.create_sliding_window_positions(min_shape=(48, 48),
                                                                   max_shape=(48, 48),
                                                                   num_scales=1,
                                                                   overlap_perc=0.5)

            # Compute the Features
            grid_with_features = feature_extractor.compute_on_windows_naiv(scaled_frame, grid, feature_scaler, preprocessing)

            # Classify each rect into car or not car
            grid_classes = sliding_cls.classify_grid(grid_with_features)

            # Collect car rectangles and scale
            if 1 in grid_classes:
                for rect in grid_classes[1]:
                    car_rectangles.append(((int(rect[0][0]//scale),
                                            int(rect[0][1]//scale)),
                                           (int(rect[1][0]//scale),
                                            int(rect[1][1]//scale))))

        if False:
            orig_car_rects = car_rectangles
            frame_orig = frame.copy()
            for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                frame_t = frame_orig.copy()
                car_rectangles = filter_detections(orig_car_rects, (frame.shape[0], frame.shape[1]), rect_overlap_thesh=t)

                # Paint raw detections in image
                for rect in car_rectangles:
                    # Random color and thickness
                    thickness = int(rnd.uniform(1, 5))
                    color = int(rnd.uniform(125, 255))
                    cv2.rectangle(frame_t, (rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), color, thickness)

                frame = np.concatenate([frame, frame_t])

        if False:
            # Draw raw detections into image
            debug_draw = frame.copy()
            for rect in car_rectangles:
                thickness = 2
                color = 255
                cv2.rectangle(debug_draw, (rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), color, thickness)

        # Filter car detection rectangles by using a heatmap and thresholding it
        car_rectangles = filter_detections(car_rectangles, (frame.shape[0], frame.shape[1]),
                                           heatmap_threshold=3,
                                           rect_overlap_thesh=0.25)

        if False:
            # Draw raw detections into image
            debug_draw = frame.copy()
            for rect in car_rectangles:
                thickness = 2
                color = 255
                cv2.rectangle(debug_draw, (rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), color, thickness)

        # Re-project the rectangles into original frame
        reproj_rects = []
        for rect in car_rectangles:
            # Rescale and Correct offset
            rect = ((int(rect[0][0] // image_scale_factor), int(rect[0][1] // image_scale_factor) + int(image_height_roi[0])),
                    (int(rect[1][0] // image_scale_factor), int(rect[1][1] // image_scale_factor) + int(image_height_roi[0])))
            reproj_rects.append(rect)

    else:
        reproj_rects = car_detections

    # Paint raw detections in image
    for rect in reproj_rects:
        thickness = 3
        color = 255
        cv2.rectangle(frame_src, (rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), color, thickness)

    car_detections = reproj_rects

    if save_output_video:
        if os.path.exists(output_video_frames) is False:
            os.makedirs(output_video_frames)
        cv2.imwrite(os.path.join(output_video_frames, 'frame_' + str(frame_id).zfill(4)+'.png'), frame_src)

clip.release()

if save_output_video:
    # Create output video
    print('Creating Video.')

    video_file = output_video_file
    clip = ImageSequenceClip(output_video_frames, fps=25)
    clip.write_videofile(video_file)

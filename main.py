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
import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import ImageSequenceClip


""" Definition of some globals """

# For intermediate outputs
output_path = 'output_images'
output_video_frames = 'video_frames'

# For Video output
output_video_file = 'output_video.mp4'
save_output_video = False

# Video files
video_file = 'project_video.mp4'
#video_file = 'challenge_video.mp4'

# Open video file
clip = cv2.VideoCapture(video_file)

# Iterate all frames of the video
frame_id = 0

while clip.isOpened():
    _, frame = clip.read()
    if frame is None:
        break

    cv2.imshow('Frames', frame)
    cv2.waitKey(25)

    if save_output_video:
        if os.path.exists(output_video_frames) is False:
            os.makedirs(output_video_frames)
        cv2.imwrite(os.path.join(output_video_frames, 'frame_' + str(frame_id).zfill(4)+'.png'), frame)

clip.release()

if save_output_video:
    # Create output video
    print('Creating Video.')

    video_file = output_video_file
    clip = ImageSequenceClip(output_video_frames, fps=25)
    clip.write_videofile(video_file)

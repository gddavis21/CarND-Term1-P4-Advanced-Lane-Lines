import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from moviepy.editor import VideoFileClip
import lanelines as LL


def calibration_progress(msg):
    print(msg)

camera_cal = LL.CameraCal()
camera_cal.calibrate_from_chessboard(
    chessboard_files=glob.glob('./camera_cal/calibration*.jpg'), 
    chessboard_size=(9,6), 
    save_diags=False,
    progress=calibration_progress)

rect_prsp = LL.PerspectiveTransform.make_top_down()

process_image = LL.LaneDetectionPipeline(camera_cal, rect_prsp)

clip = VideoFileClip('./project_video.mp4')
lanes_clip = clip.fl_image(process_image)
lanes_clip.write_videofile('./output_videos/project_video_out.mp4', audio=False)
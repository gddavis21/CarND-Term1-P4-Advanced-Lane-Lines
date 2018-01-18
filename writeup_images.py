import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
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

print('undistorted calibration image')
cal_bgr = cv2.imread('./camera_cal/calibration2.jpg')
cv2.imwrite('./output_images/cal_raw.jpg', cal_bgr)
cal_bgr = camera_cal.undistort_image(cal_bgr)
cv2.imwrite('./output_images/cal_undist.jpg', cal_bgr)

print('undistorted test image')
test_bgr = cv2.imread('./test_images/test4.jpg')
cv2.imwrite('./output_images/test_raw.jpg', test_bgr)
test_bgr = camera_cal.undistort_image(test_bgr)
cv2.imwrite('./output_images/test_undist.jpg', test_bgr)

print('thresholded images')
test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
test_hsv = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2HSV)
test_thresh_yellow = LL.thresh_yellow(test_hsv)
cv2.imwrite('./output_images/test_thresh_yellow.jpg', test_thresh_yellow)
test_thresh_white = LL.thresh_white(test_rgb)
cv2.imwrite('./output_images/test_thresh_white.jpg', test_thresh_white)
test_thresh_comb = LL.thresh_lane_lines(test_rgb)
cv2.imwrite('./output_images/test_thresh_combined.jpg', test_thresh_comb)

print('perspective images')
straight_lines_bgr = cv2.imread('./test_images/straight_lines2.jpg')
straight_lines_bgr = camera_cal.undistort_image(straight_lines_bgr)
cv2.imwrite('./output_images/straight_lines_undist.jpg', straight_lines_bgr)
img_size = (straight_lines_bgr.shape[1], straight_lines_bgr.shape[0])
straight_lines_warp = rect_prsp.warp_image(straight_lines_bgr, img_size)
cv2.imwrite('./output_images/straight_lines_warped.jpg', straight_lines_warp)
test_thresh_warp = rect_prsp.warp_image(test_thresh_comb, img_size)
cv2.imwrite('./output_images/test_thresh_warped.jpg', test_thresh_warp)

print('full pipeline image')
process_image = LL.LaneDetectionPipeline(camera_cal, rect_prsp)
test_pipe = process_image(test_rgb)
cv2.imwrite('./output_images/pipeline_result.jpg', cv2.cvtColor(test_pipe, cv2.COLOR_RGB2BGR))

print('lane-lines detection image')
lane = LL.Lane(test_thresh_comb, rect_prsp, (LL.PIXEL_SIZE_X, LL.PIXEL_SIZE_Y))
lane_det_rgb = lane.lane_lines_diagnostic()
lane_det_bgr = cv2.cvtColor(lane_det_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite('./output_images/lane_detection.jpg', lane_det_bgr)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import camera

def calibration_progress(msg):
    print(msg)

cal_files = glob.glob('./camera_cal/calibration*.jpg')

camera_matrix, dist_coeffs = camera.calibrate_from_chessboard(
    cal_files, 
    pattern_size=(9,6), 
    save_diags=True,
    diag_prefix='corners-',
    progress=calibration_progress)
    
example_img = cv2.imread('./camera_cal/calibration2.jpg')
undist_img = camera.undistort_image(example_img, camera_matrix, dist_coeffs)
cv2.imwrite('./camera_cal/undist-calibration2.jpg', undist_img)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
fig.tight_layout()
ax1.imshow(example_img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undist_img)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

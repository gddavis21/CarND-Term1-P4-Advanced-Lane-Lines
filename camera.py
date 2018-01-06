import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def make_nominal_corners(pattern_size):
    ''' create nominal "object points" for use with cv2.calibrateCamera '''
    px = pattern_size[0]
    py = pattern_size[1]
    nom = np.zeros((px*py,3), np.float32)

    for y in range(py):
        for x in range(px):
            nom[px*y + x, :] = (x,y,0)
    return nom


def calibrate_from_chessboard(
    image_files,
    pattern_size, 
    save_diags, 
    diag_prefix=None,
    progress=None):
    '''
    compute camera calibration parameters from chessboard images
    inputs:
      - image_files: list of chessboard image file paths
      - pattern_size: chessboard dimensions
      - save_diags: enable saving diagnostic images
      - diag_prefix: file name prefix for diagnostic images
    outputs:
      - camera_matrix: computed camera matrix
      - dist_coeffs: compute distortion coefficients
    '''
    nom_corners = make_nominal_corners(pattern_size)
    obj_points = []
    img_points = []
    img_size = None
    
    if progress:
        progress('locating chessboard corners...')
    
    for file_path in image_files:
        if progress:
            progress(file_path)
        # load image from disk, convert to grayscale
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find chessboard pattern in image
        found_corners, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found_corners:
            # add points correspondences for this image
            obj_points.append(nom_corners)
            img_points.append(corners)
            if not img_size:
                # get image size from 1st image
                img_size = (gray.shape[1], gray.shape[0])
            if save_diags and diag_prefix:
                # save diagnostic image with chessboard annotations
                cv2.drawChessboardCorners(img, pattern_size, corners, found_corners)
                dir, fname = os.path.split(file_path)
                cv2.imwrite(os.path.join(dir, diag_prefix+fname), img)
            
    camera_matrix = None
    dist_coeffs = None
    
    if img_points:
        if progress:
            progress('calibrating camera...')
        repr_err, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None)
        if progress:
            progress('finished camera calibration')
        
    return camera_matrix, dist_coeffs

def undistort_image(img, camera_matrix, dist_coeffs):
    return cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
    

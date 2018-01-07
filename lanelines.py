import os
import cv2
import numpy as np

class Camera:
    def __init__(self):
        '''
        '''
        self._camera_matrix = None
        self._dist_coeffs = None
        self._prsp_forward = None
        self._prsp_backward = None
            

    def calibrate_from_chessboard(
        self,
        chessboard_files,
        chessboard_size, 
        save_diags=False, 
        diag_prefix=None,
        progress=None):
        '''
        compute camera calibration parameters from chessboard images
        inputs:
          - chessboard_files: list of image file paths
          - chessboard_size: board dimensions
          - save_diags: enable saving diagnostic images
          - diag_prefix: file name prefix for diagnostic images
          - progress: progress message callback function
        outputs:
          - camera_matrix: OpenCV camera matrix
          - dist_coeffs: OpenCV distortion coefficients
        '''
        # reset camera cal data
        self._camera_matrix = None
        self._dist_coeffs = None
        
        # make nominal corners
        px = chessboard_size[0]
        py = chessboard_size[1]
        nom_corners = np.zeros((px*py,3), np.float32)

        for y in range(py):
            for x in range(px):
                nom_corners[px*y + x, :] = (x,y,0)
        
        # find corner points in chessboard images
        if progress:
            progress('locating chessboard corners...')
        
        obj_points = []
        img_points = []
        img_size = None
        
        for file_path in chessboard_files:
            if progress:
                progress(file_path)
            # load image from disk, convert to grayscale
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find chessboard pattern in image
            found_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if found_corners:
                # add points correspondences for this image
                obj_points.append(nom_corners)
                img_points.append(corners)
                if not img_size:
                    # get image size from 1st image
                    img_size = (gray.shape[1], gray.shape[0])
                if save_diags and diag_prefix:
                    # save diagnostic image with chessboard annotations
                    cv2.drawChessboardCorners(img, chessboard_size, corners, found_corners)
                    dir, fname = os.path.split(file_path)
                    cv2.imwrite(os.path.join(dir, diag_prefix+fname), img)
                
        if not img_points:
            if progress:
                progress('calibration failed: not enough corners found')
            return False
            
        if progress:
            progress('calibrating camera...')

        # compute camera calibration
        repr_err, C, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None)

        self._camera_matrix = C
        self._dist_coeffs = D

        if progress:
            progress('finished camera calibration')
            
        return True

        
    def undistort_image(self, image):
        return cv2.undistort(
            image, 
            self._camera_matrix, 
            self._dist_coeffs, 
            None, 
            self._camera_matrix)
            
            
    def set_perspective(self, src_quad, dst_quad):
        self._prsp_forward = cv2.getPerspectiveTransform(src_quad, dst_quad)
        self._prsp_backward = cv2.getPerspectiveTransform(dst_quad, src_quad)
        
        
    def warp_image(self, image, dst_size):
        return cv2.warpPerspective(
            image, 
            self._prsp_forward, 
            dst_size, 
            flags=cv2.INTER_LINEAR)
            
            
    def unwarp_image(self, image, dst_size):
        return cv2.warpPerspective(
            image, 
            self._prsp_backward, 
            dst_size, 
            flags=cv2.INTER_LINEAR)
        
        
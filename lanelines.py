import os
import cv2
import numpy as np

class CameraCal:
    def __init__(self):
        '''
        '''
        self._camera_matrix = None
        self._dist_coeffs = None
            
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
        if progress:
            progress('calibrating camera...')
        
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
            progress('locating calibration landmarks...')
        
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
                progress('camera calibration failed: not enough landmarks found')
            return False
            
        if progress:
            progress('computing calibration data...')

        # compute camera calibration
        repr_err, C, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None)

        self._camera_matrix = C
        self._dist_coeffs = D

        if progress:
            progress('camera calibration finished')
            
        return True

    def undistort_image(self, image):
        return cv2.undistort(
            image, 
            self._camera_matrix, 
            self._dist_coeffs, 
            None, 
            self._camera_matrix)
            
                
class PerspectiveTransform:
    def __init__(self, src_quad, dst_quad):
        self._forward_transform = cv2.getPerspectiveTransform(src_quad, dst_quad)
        self._backward_transform = cv2.getPerspectiveTransform(dst_quad, src_quad)
        
    def warp_image(self, image, dst_size):
        return cv2.warpPerspective(
            image, 
            self._forward_transform, 
            dst_size, 
            flags=cv2.INTER_LINEAR)
            
    def unwarp_image(self, image, dst_size):
        return cv2.warpPerspective(
            image, 
            self._backward_transform, 
            dst_size, 
            flags=cv2.INTER_LINEAR)
            
    def warp_points(self, x, y):
        wrp = cv2.perspectiveTransform(
            np.array([np.column_stack((x,y))]), 
            self._forward_transform)
        return np.hsplit(wrp[0], 2)

    def unwarp_points(self, x, y):
        unw = cv2.perspectiveTransform(
            np.array([np.column_stack((x,y))]), 
            self._backward_transform)
        return np.hsplit(unw[0], 2)
        
    @staticmethod
    def make_top_down():
        return PerspectiveTransform(
            src_quad=np.float32([[192,719],[579,459],[701,459],[1122,719]]), 
            dst_quad=np.float32([[240,719],[240,74],[1040,74],[1040,719]]))
        

# class LaneLine():
    # def __init__(self):
        # # was the line detected in the last iteration?
        # self.detected = False  
        # # x values of the last n fits of the line
        # self.recent_xfitted = [] 
        # #average x values of the fitted line over the last n iterations
        # self.bestx = None     
        # #polynomial coefficients averaged over the last n iterations
        # self.best_fit = None  
        # #polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]  
        # #radius of curvature of the line in some units
        # self.radius_of_curvature = None 
        # #distance in meters of vehicle center from the line
        # self.line_base_pos = None 
        # #difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float') 
        # #x values for detected line pixels
        # self.allx = None  
        # #y values for detected line pixels
        # self.ally = None
        
class LaneLine:
    def __init__(self, xpix, ypix, image_size, pixel_size):
        self.xpix = xpix
        self.ypix = ypix
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.coeffs_pix = np.polyfit(ypix, xpix, 2)
        self.coeffs_world = np.polyfit(ypix*pixel_size[1], xpix*pixel_size[0], 2)
        
    def source_pixels(self):
        return self.xpix, self.ypix
        
    def contour(self, yvals):
        return np.polyval(self.coeffs_pix, yvals)
    
    def position(self):
        # position = evaluate lane-line polynomial at bottom of image
        y = (self.image_size[1]-1)*self.pixel_size[1]
        return np.polyval(self.coeffs_world, y)
        
    def radius_of_curvature(self):
        # evaluate radius at bottom of image
        C = self.coeffs_world
        y = (self.image_size[1]-1)*self.pixel_size[1]
        return ((1 + (2*C[0]*y + C[1])**2)**1.5) / np.absolute(2*C[0])
        
class Lane:
    def __init__(self, poi_mask, rect_prsp, rect_pixel_size, prior=None):
        self.poi_mask = poi_mask
        self.image_size = (poi_mask.shape[1], poi_mask.shape[0])
        self.pixel_size = rect_pixel_size
        self.prsp = rect_prsp
        self.prior = prior
        self.rect_poi_mask = rect_prsp.warp_image(poi_mask, self.image_size)
        self.left_line = None
        self.right_line = None
        if prior:
            self.detect_lane_lines_with_prior(prior)
        else:
            self.detect_lane_lines_from_scratch()
        
    def detect_lane_lines_from_scratch(self):
        '''
        '''
        # start with rectified binary
        rect_binary = self.rect_poi_mask // 255
        # Take a histogram of the bottom half of the image
        histogram = np.sum(rect_binary[rect_binary.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(rect_binary.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = rect_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = rect_binary.shape[0] - (window+1)*window_height
            win_y_high = rect_binary.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        self.left_line = LaneLine(leftx, lefty, self.image_size, self.pixel_size)
        self.right_line = LaneLine(rightx, righty, self.image_size, self.pixel_size) 
        
    def detect_lane_lines_with_prior(self, prior):
        '''
        '''
        # start with rectified binary
        rect_binary = self.rect_poi_mask // 255

        nonzero = rect_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100

        prior_left = prior.left_line.contour(nonzeroy)
        left_bound = prior_left - margin
        right_bound = prior_left + margin
        left_lane_inds = ((nonzerox > left_bound) & (nonzerox < right_bound))
        # left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        # left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        # left_fit[1]*nonzeroy + left_fit[2] + margin))) 

        prior_right = prior.right_line.contour(nonzeroy)
        left_bound = prior_right - margin
        right_bound = prior_right + margin
        right_lane_inds = ((nonzerox > left_bound) & (nonzerox < right_bound))
        # right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        # right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        # right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.left_line = LaneLine(leftx, lefty, self.image_size, self.pixel_size)
        self.right_line = LaneLine(rightx, righty, self.image_size, self.pixel_size) 

        # # Fit a second order polynomial to each
        # left_fit = np.polyfit(lefty, leftx, 2)
        # right_fit = np.polyfit(righty, rightx, 2)
        # # Generate x and y values for plotting
        # ploty = np.linspace(0, rect_binary.shape[0]-1, rect_binary.shape[0] )
        # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # # Create an image to draw on and an image to show the selection window
        # out_img = np.dstack((rect_binary, rect_binary, rect_binary))*255
        # window_img = np.zeros_like(out_img)
        # # Color in left and right line pixels
        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # # Generate a polygon to illustrate the search window area
        # # And recast the x and y points into usable format for cv2.fillPoly()
        # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                      # ploty])))])
        # left_line_pts = np.hstack((left_line_window1, left_line_window2))
        # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                      # ploty])))])
        # right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # # Draw the lane onto the warped blank image
        # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        # plt.imshow(result)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)        
        
    def is_valid(self, nom_lane_width, lane_width_tol, parallel_lines_tol):
        '''
        '''
        # compute lane-line contours & differences
        yvals = np.linspace(0, img_height-1, img_height)
        xfit_L = self.left_line.contour(yvals)
        xfit_R = self.right_line.contour(yvals)
        diff = np.absolute(xfit_R - xfit_L) * self.pixel_size[0]
        diff_bar = np.mean(diff)
        
        # check lane for correct width
        if np.absolute(diff_bar - nom_lane_width) > lane_width_tol:
            return False
            
        # check lane-lines for approximate parallelism
        if np.any(np.absolute(diff - diff_bar) > parallel_lines_tol):
            return False
        
        return True
        
    def position(self):
        pos_L = self.left_line.position()
        pos_R = self.right_line.position()
        return (pos_L + pos_R) / 2
        
    def radius_of_curvature(self):
        rad_L = self.left_line.radius_of_curvature()
        rad_R = self.right_line.radius_of_curvature()
        return (rad_L + rad_R) / 2
        
    def overlay_image(self, lane_color):
        '''
        '''
        # Create empty image to draw the overlay on
        img_width = self.image_size[0]
        img_height = self.image_size[1]
        empty_channel = np.zeros((img_height, img_width), dtype=np.uint8)
        overlay = np.dstack((empty_channel, empty_channel, empty_channel))
        
        # compute left/right lane contours
        yvals = np.linspace(0, img_height-1, img_height)
        xfit_L = self.left_line.contour(yvals)
        xfit_R = self.right_line.contour(yvals)

        # Recast x and y points into usable format for cv2.fillPoly()
        pts_L = np.array([np.transpose(np.vstack([xfit_L, yvals]))])
        pts_R = np.array([np.flipud(np.transpose(np.vstack([xfit_R, yvals])))])
        pts = np.hstack((pts_L, pts_R))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, np.int_([pts]), lane_color)
        
        # return unwarped overlay image
        return self.prsp.unwarp_image(overlay, self.image_size)
        
class LaneDetectionPipeline:
    '''
    '''
    def __init__(self, camera_cal, rect_prsp):
        self.camera_cal = camera_cal
        self.rect_prsp = rect_prsp
        
    def __call__(self, raw_rgb):
        '''
        '''
        pix_size_x = 3.7 / 720
        pix_size_y = 3.0 / 70
        pix_size = (pix_size_x, pix_size_y)
        
        img_width = raw_rgb.shape[1]
        img_height = raw_rgb.shape[0]
        image_size = (img_width, img_height)

        # undistorted image (remove lens distortion)
        rgb = self.camera_cal.undistort_image(raw_rgb)

        # pixels-of-interest mask
        poi_mask = thresh_lane_lines(rgb)
        
        # detect lane lines
        lane = Lane(poi_mask, self.rect_prsp, pix_size)

        # make overlay image of lane bounded by left/right contours
        overlay = lane.overlay_image(lane_color=(0,255,0))

        # blend undistorted image with overlay
        result = cv2.addWeighted(rgb, 1, overlay, 0.3, 0)
        
        # und_fx_L, und_fy_L = prsp.unwarp_points(td_fx_L, td_fy)
        # und_fx_R, und_fy_R = prsp.unwarp_points(td_fx_R, td_fy)
        # result[np.int_(und_fy_L), np.int_(und_fx_L)] = [255,0,0]
        # result[np.int_(und_fy_R), np.int_(und_fx_R)] = [255,0,0]
        
        # annotate radius of curvature
        cv2.putText(
            result,
            'Radius of Curvature = %d m' % int(lane.radius_of_curvature()),
            org=(20,50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255,255,255),
            thickness=1,
            lineType=cv2.LINE_AA)

        # annotate vehicle offset from lane center
        td_img_ctr_x, td_img_bot_y = self.rect_prsp.warp_points(img_width/2, img_height-1)
        veh_offs = lane.position() - (td_img_ctr_x * pix_size_x)
        
        if veh_offs < 0.0:
            offs_side = 'left'
        else:
            offs_side = 'right'
        
        cv2.putText(
            result,
            'Vehicle is %.2fm %s of center' % (abs(veh_offs), offs_side),
            org=(20,100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255,255,255),
            thickness=1,
            lineType=cv2.LINE_AA)

        return result
        
def sobel_xy(gray, kernel_size=3):
    scale = 1.0 / 2.0**(kernel_size*2 - 4)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=scale)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=scale)
    return sobel_x, sobel_y
    
def scharr_xy(gray):
    scale = 0.0625  # 1/16
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale)
    return scharr_x, scharr_y
    
def polar_gradient(grad_x, grad_y):
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(np.absolute(grad_x), np.absolute(grad_y))
    return grad_mag, grad_dir
    
def image_gradient(gray, kernel_size=3):
    scale = 1.0 / 2.0**(kernel_size*2 - 4)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=scale)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=scale)
    return cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)
    
def thresh_yellow(hsv):
    HSV_YELLOW_LOWER = np.array([15, 70, 130])
    HSV_YELLOW_UPPER = np.array([25, 255, 255])
    return cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
    
# def thresh_white(gray):
    # return cv2.adaptiveThreshold(
        # gray, 
        # maxValue=255, 
        # adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
        # thresholdType=cv2.THRESH_BINARY,
        # blockSize=33,
        # C=-60)
        
# def thresh_white_proj(gray_proj):
    # return cv2.adaptiveThreshold(
        # gray_proj, 
        # maxValue=255, 
        # adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
        # thresholdType=cv2.THRESH_BINARY,
        # blockSize=53,
        # C=-20)

def thresh_white(rgb):
    RGB_WHITE_LOWER = np.array([190, 190, 190])
    RGB_WHITE_UPPER = np.array([255, 255, 255])
    return cv2.inRange(rgb, RGB_WHITE_LOWER, RGB_WHITE_UPPER)
    
def thresh_lane_edges(gray):
    grad_mag, grad_dir = image_gradient(gray, kernel_size=13)
    strong_edges = cv2.inRange(grad_mag, 25, 255)
    pos_horz_edges = cv2.inRange(grad_dir, 80, 100)
    neg_horz_edges = cv2.inRange(grad_dir, 260, 280)
    horz_edges = cv2.bitwise_or(pos_horz_edges, neg_horz_edges)
    return cv2.bitwise_and(strong_edges, cv2.bitwise_not(horz_edges)) 

# def thresh_lane_lines(rgb, prsp):
    # hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # image_size = (gray.shape[1], gray.shape[0])
    # gray_proj = prsp.warp_image(gray, image_size)
    # yellow = thresh_yellow(hsv)
    # white_proj = thresh_white_proj(gray_proj)
    # white = prsp.unwarp_image(white_proj, image_size)
    # result = cv2.bitwise_or(yellow, white)
    # # edges = thresh_lane_edges(gray)
    # # result = cv2.bitwise_or(result, edges)
    # return result
    
def thresh_lane_lines(rgb):
    white = thresh_white(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = thresh_yellow(hsv)
    result = cv2.bitwise_or(yellow, white)
    # edges = thresh_lane_edges(gray)
    # result = cv2.bitwise_or(result, edges)
    return result
    
# def detect_lanes(binary_warped):
    # # Assuming you have created a warped binary image called "binary_warped"
    # # Take a histogram of the bottom half of the image
    # histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # # # Create an output image to draw on and  visualize the result
    # # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # # Find the peak of the left and right halves of the histogram
    # # These will be the starting point for the left and right lines
    # midpoint = np.int(histogram.shape[0]/2)
    # leftx_base = np.argmax(histogram[:midpoint])
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # # Choose the number of sliding windows
    # nwindows = 9
    # # Set height of windows
    # window_height = np.int(binary_warped.shape[0]/nwindows)
    # # Identify the x and y positions of all nonzero pixels in the image
    # nonzero = binary_warped.nonzero()
    # nonzeroy = np.array(nonzero[0])
    # nonzerox = np.array(nonzero[1])
    # # Current positions to be updated for each window
    # leftx_current = leftx_base
    # rightx_current = rightx_base
    # # Set the width of the windows +/- margin
    # margin = 100
    # # Set minimum number of pixels found to recenter window
    # minpix = 50
    # # Create empty lists to receive left and right lane pixel indices
    # left_lane_inds = []
    # right_lane_inds = []

    # # Step through the windows one by one
    # for window in range(nwindows):
        # # Identify window boundaries in x and y (and right and left)
        # win_y_low = binary_warped.shape[0] - (window+1)*window_height
        # win_y_high = binary_warped.shape[0] - window*window_height
        # win_xleft_low = leftx_current - margin
        # win_xleft_high = leftx_current + margin
        # win_xright_low = rightx_current - margin
        # win_xright_high = rightx_current + margin
        # # # Draw the windows on the visualization image
        # # cv2.rectangle(
            # # out_img,
            # # (win_xleft_low,win_y_low),
            # # (win_xleft_high,win_y_high),
            # # color=(0,255,0), 
            # # thickness=2) 
        # # cv2.rectangle(
            # # out_img,
            # # (win_xright_low,win_y_low),
            # # (win_xright_high,win_y_high),
            # # color=(0,255,0), 
            # # thickness=2) 
        # # Identify the nonzero pixels in x and y within the window
        # good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        # (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        # good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        # (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # # Append these indices to the lists
        # left_lane_inds.append(good_left_inds)
        # right_lane_inds.append(good_right_inds)
        # # If you found > minpix pixels, recenter next window on their mean position
        # if len(good_left_inds) > minpix:
            # leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        # if len(good_right_inds) > minpix:        
            # rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # # Concatenate the arrays of indices
    # left_lane_inds = np.concatenate(left_lane_inds)
    # right_lane_inds = np.concatenate(right_lane_inds)

    # # Extract left and right line pixel positions
    # leftx = nonzerox[left_lane_inds]
    # lefty = nonzeroy[left_lane_inds] 
    # rightx = nonzerox[right_lane_inds]
    # righty = nonzeroy[right_lane_inds] 
    
    # return leftx, lefty, rightx, righty

    # # # Fit a second order polynomial to each
    # # left_fit = np.polyfit(lefty, leftx, 2)
    # # right_fit = np.polyfit(righty, rightx, 2)

    # # # Generate x and y values for plotting
    # # ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    # # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # # out_img[np.round(ploty), np.round(left_fitx)] = [255, 255, 0]
    # # out_img[np.round(ploty), np.round(right_fitx)] = [255, 255, 0]
    # # return out_img
    
# def filled_lane_overlay(image_size, fx_L, fx_R, fy, lane_color):
    # '''
    # '''
    # # Create empty image to draw the overlay on
    # empty = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
    # overlay = np.dstack((empty, empty, empty))

    # # Recast x and y points into usable format for cv2.fillPoly()
    # pts_L = np.array([np.transpose(np.vstack([fx_L, fy]))])
    # pts_R = np.array([np.flipud(np.transpose(np.vstack([fx_R, fy])))])
    # pts = np.hstack((pts_L, pts_R))

    # # Draw the lane onto the warped blank image
    # cv2.fillPoly(overlay, np.int_([pts]), lane_color)
    # return overlay
    
# def radius_of_curvature(coeff, x):
    # return ((1 + (2*coeff[0]*x + coeff[1])**2)**1.5) / np.absolute(2*coeff[0])

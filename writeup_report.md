## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[img_cal_raw]: ./output_images/cal_raw.jpg "Raw Calibration Image"
[img_cal_corners]: ./output_images/cal_corners.jpg "Calibration Image with Detected Corners"
[img_cal_undist]: ./output_images/cal_undist.jpg "Undistorted Calibration Image"
[img_test_raw]: ./output_images/test_raw.jpg "Raw Road Image"
[img_test_undist]: ./output_images/test_undist.jpg "Undistorted Road Image"
[img_thresh_yellow]: ./output_images/test_thresh_yellow.jpg "Threshold Yellow Pixels"
[img_thresh_white]: ./output_images/test_thresh_white.jpg "Threshold White Pixels"
[img_thresh_comb]: ./output_images/test_thresh_combined.jpg "Combined Threshold Image"
[img_lines_prsp]: ./output_images/straight_lines_prsp.jpg "Lane Lines in Perspective"
[img_lines_undist]: ./output_images/straight_lines_undist.jpg "Straight Lane Lines"
[img_lines_warp]: ./output_images/straight_lines_warped.jpg "Rectified Lane Lines (bird's eye view)"
[img_thresh_warp]: ./output_images/test_thresh_warped.jpg "Rectified Threshold Image"
[img_lane_det]: ./output_images/lane_detection.jpg "Lane Detection & Curve Fitting"
[img_pipeline_result]: ./output_images/pipeline_result.jpg "Pipeline Result Image"

[video1]: ./project_video.mp4 "Video"
[video_pipeline_result]: ./output_videos/project_video_out.mp4 "Project Video Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I implemented camera calibration and distortion correction in class 'lanelines.CameraCal' (lanelines.py 155-252). The calibration code is contained in class method `CameraCal.calibrate_from_chessboard()` (lanelines.py line 162).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `nom_corners` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  

The next step is detection of the chessboard corners pattern in each calibration image, using the `cv2.findChessboardCorners()` function. With each successful chessboard detection, `img_points` is appended with the (x, y) pixel position of each of the corners in the image plane.

`CameraCal.calibrate_from_chessboard()` provides optional parameters that enable the caller to save diagnostic images of corner detections, using the 'cv2.drawChessboardCorners()` function. This was useful during development as a way to verify the functionality of corner detection.

I then use `obj_points` and `img_points` to compute the camera calibration and distortion coefficients with `cv2.calibrateCamera()` and store these results in member variables of the CameraCal instance. After calibrating, the CameraCal client can apply distortion correction to any number of images with method `CameraCal.undistort_image()` (lanelines.py line 246) which applies the `cv2.undistort()` function (using the stored calibration data), and returns the undistorted image. 

Below is an example of one of the chessboard calibration images, the diagnostic image showing detected corners, and the same image with distortion correction applied:

![Calibration Image][img_cal_raw] 
![Corner Detection][img_cal_corners] 
![Distortion Corrected][img_cal_undist]

### Pipeline (single images)

My high-level image processing 'pipeline' is implemented via the 'callable' class 'lanelines.LaneDetectionPipeline' (lanelines.py 16-152). The class instance stores required camera calibration and perspective transform objects, and provides the `__call__()` mechanism which python uses to invoke the class instance like a function call. This technique enables my pipeline to conveniently encapsulate persistent state data like camera calibration and perspective transform objects, to provide a reusable interface for processing single images or video streams, and to cache 'prior state' data that's useful for efficient video stream processing.

The 'pipeline' algorithm consists of the following steps:
1. Apply distortion correction to incoming image
2. Generate threshold image of lane-line candidate pixels
3. Create an instance of `lanelines.Lane` which encapsulates the following:
    * Segment left and right lane-line regions in the threshold image
    * Compute functional approximations for left & right lane-line curves
    * Compute lane radius of curvature
    * Compute offset of vehicle from lane center
4. Annotate the (undistorted) image with the following:
    * Draw graphic representation of detected lane
    * Print computed radius of curvature of the detected lane
    * Print vehicle offset from center of detected lane
5. Return annotated image

#### 1. Provide an example of a distortion-corrected image.

The following is an example of a raw image captured by the camera, and the same image with distortion correction applied via `CameraCal.undistort_image()`:

![Test Image][img_test_raw] 
![Distortion Corrected][img_test_undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

One of the main steps in the lane detection pipeline is creation of a binary image of candidate lane-line pixels. This "pixels of interest" image is then passed as input to the lane-line detection/measurement algorithm. In my code, the "pixels of interest" binary image is created by the function `lanelines.thresh_lane_lines()` (lanelines.py line 612).

`thresh_lane_lines()` does the following:

1. Creates a threshold image of 'white' pixels using function `lanelines.thresh_white()` (lanelines.py line 586).
    * `thresh_white()` uses `cv2.inRange()` to threshold the RGB image for high values (190-255) in all 3 color channels.
    * I implemented a script (thresh_white.py) that creates a simple interactive GUI for efficiently optimizing these threshold values.

2. Creates a threshold image of 'yellow' pixels using function `lanelines.thresh_yellow()` (lanelines.py line 563):
    * `thresh_yellow()` operates on an HSV color image, so `thresh_lane_lines()` uses `cv2.cvtColor()` to create an HSV image from the RGB input image.
    * `thresh_yellow()` uses `cv2.inRange()` to threshold the HSV image within the hue range 15-25, where Saturation >= 70 and Value >= 130.
    * I implemented a script (thresh_yellow.py) that creates a simple interactive GUI for efficiently optimizing these threshold values.
    
3. Uses `cv2.bitwise_or()` to combine the white and yellow threshold images into the result binary image.

The following is an example of a source RGB image, the white threshold image, the yellow threshold image, and the combined threshold image:

![RGB image][img_test_undist]
![White threshold][img_thresh_white]
![Yellow threshold][img_thresh_yellow]
![Combined result][img_thresh_comb]

NOTE: I also implemented and experimented with thresholding gradient magnitude/direction images, using the Sobel filter with a variety of kernel sizes and threshold values. This method is effective for marking lane-line pixels, but because it identifies contrast instead of color I found that it also marked too many other pixels unrelated to the lane lines, resulting in difficulty in later stages of lane-line detection.

I do believe edge detection could be a beneficial contributor to the "pixels of interest" stage, but not simply as a straight addition to the white/yellow pixels. During the time I was developing this project, I made an effort to observe my own perception of lane lines while driving, and it's clear to me that contrast by itself is not a primary visual cue for my own lane-line perception. There's too many other high-contrast visual features on the road--color has to be the primary cue.

My idea, though I did not have time to implement it, would be to use contrast to enhance the threshold image as follows:
* Threshold white & yellow pixels as described above.
* Combine white & yellow into combined threshold image.
* Use morphological dilation on combined threshold image to create a mask.
* Threshold gradient magnitude.
* Add gradient threshold to combined threshold, masked by dilated combined threshold image.
* This means pixels with strong contrast would be added to the result, as long as they're sufficiently close to pixels already identified as white/yellow. In other words, use contrast to enhance detected color instead of blindly adding to it.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The lane-line detection logic uses a perspective transform to rectify the threshold image into the (top-down) plane of the road, which greatly simplifies the lane-line geometry to be analyzed (reducing a 3D problem to 2D). The code for my perspective transform is implemented in the class `lanelines.PerspectiveTransform` (lanelines.py 255-297). The class constructor takes as inputs source and destination quadrilaterals, which are used with the function `cv2.getPerspectiveTransform()` to compute both forward and backward transformation matrices. The class instance can then be used to perform perspective projection between the source and destination image planes.

`PerspectiveTransform` provides the following instance methods:
* `warp_image()` transforms a given image from source plane into destination plane, using the function `cv2.warpPerspective()` with the stored forward transform matrix.
* `unwarp_image()` transforms a given (rectified) image from destination plane back to source plane, using the function `cv2.warpPerspective()` with the stored backward transform matrix.
* `warp_point()` and `warp_points()` transform single and multiple x/y points from source plane into destination plane, using the function `cv2.perspectiveTransform()` with the stored forward transform matrix.
* `unwarp_point()` and `unwarp_points()` transform single and multiple (rectified) x/y points from destination plane back to source plane, using the function `cv2.perspectiveTransform()` with the stored backward transform matrix.

To arrive at the source and destination coordinates, I selected an (undistorted) road image where the lane-lines are very straight, sketched out a trapezoid shape of the lane lines in perspective, observed the trapezoid corner points in the image, and then mapped these corner points to rectangular corners in an idealized top-down image plane. 

![Perspective source corners][img_lines_prsp]

The destination corner coordinates were carefully chosen as a 'masking' technique for the rectified image, balancing inclusion of potential lane pixels with exclusion of noisy pixels thresholded from the surrounding areas. This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 456      | 320, 0        | 
| 208, 720      | 320, 720      |
| 1122, 720     | 960, 720      |
| 701, 456      | 960, 0        |

The class static method `PerspectiveTransform.make_top_down()` (lanelines.py line 292) constructs a `PerspectiveTransform` instance with these hard-coded values. I verified that my perspective transform was working as expected by applying `PerspectiveTransform.warp_image()` on the 'straight-lines' road image and observing that the lane lines appear parallel in the warped image. 

![Straight lines in perspective][img_lines_undist]
![Rectified straight lines][img_lines_warp]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

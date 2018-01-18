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

The code for this step is contained in class method `CameraCal.calibrate_from_chessboard()` (lanelines.py 162-244).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `nom_corners` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  

The next step is detection of the pattern of chessboard corners in each supplied calibration image. A 9x6 pattern of corner points is detected in each image using the `cv2.findChessboardCorners` function. With each successful chessboard detection, `img_points` is appended with the (x, y) pixel position of each of the corners in the image plane.

`CameraCal.calibrate_from_chessboard()` provides some optional parameters that enable the caller to save diagnostic images of the chessboard detection, using the 'cv2.drawChessboardCorners` function. This was useful during development as a way to verify that the detection was working as expected.

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function, and store these as member data in the CameraCal instance for later use. After calibrating, the CameraCal client is able to apply distortion correction to any number of images with class method `CameraCal.undistort_image()` which applies the `cv2.undistort()` function (using the previously computed calibration data), and returns the undistorted image. 

Below is an example of one of the chessboard calibration images, the diagnostic image showing detected corners, and the same image with distortion correction applied:

![Calibration Image][img_cal_raw] 
![Corner Detection][img_cal_corners] 
![Distortion Corrected][img_cal_undist]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The following is an example of an image as captured from the camera, and the same image with distortion correction applied via CameraCal.undistort_image():

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

My thought, though I did not have time to implement it, would be to use contrast to enhance the threshold image as follows:
* Threshold white & yellow pixels as described above.
* Combine white & yellow into combined threshold image.
* Use morphological dilation on combined threshold image to create a mask.
* Threshold gradient magnitude.
* Add gradient threshold to combined threshold, using dilated mask.
* This means pixels with strong contrast would be added to the result, as long as they're sufficiently close to pixels already identified as white/yellow. In other words, use contrast to enhance color instead of just adding to it.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

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

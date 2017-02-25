##Vehicle Detection

[//]: # (Image References)
[image1]: ./output_images/car_notcar.jpg
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/optimization_init.jpg
[image4]: ./output_images/sliding_windows.jpg
[image5]: ./output_images/boxes.jpg
[image6]: ./output_images/heat.jpg
[image7]: ./output_images/bboxes_and_heat_1.jpg
[video1]: ./project_video.mp4

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cells 2-15 of the IPython notebook and in lines 12 through 91 of the file called `detect_utils.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images from GTI and KITTI data sets.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then divided data data into train set and test sets. Because GTI data contains time series of car images,  I included the GTI data in the train set only. This is to avoid high correlation between the train set and the test images. A better, though more laborious approach would be to manually separate out time series and divide them between the train and the test sets.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes in the train set and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I performed feature selection with Bayesian optimization using Gaussian Processes implemented in `skopt` package (the code cells 8-12 of the IPython notebook).
My objective function calculates accuracy (negative value) of car vs notcar classification with linear SVM using features scaled to zero mean and unit variance before training the classifier.
I initially searched through the full parameter space: `color_space`, `orient`, `hog_channel`, `cell_per_block`, `pix_per_cell`, `spatial_size`, `hist_bins`, `spatial_feat`, `hist_feat`. This initial optimization run showed that `YCrCb` color space leads to highest accuracy and that including spatial and histogram features provides only minor improvement. 
For this reason, my final features are HOG features using the `YCrCb` color space.

Here is the objective function for the final optimization run with optimal HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

![alt text][image3]

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After feature selection I trained a linear SVM using full data set and HOG features scaled to zero mean and unit variance.
The code for this step is contained in the code cell 21 of the IPython notebook.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used smaller scales for boxes in the upper part and larger scales for boxes in the lower part of the bottom half of the image.
To decide scale sizes, I plotted boxes on test images and make sure that cars fit into them.  I initially used overlap of 0.5, but then the resulting box density was too small for effective false positive removal so so I increased the overlap to 0.75.  I also adjusted sliding windows scales and positions to maximize the performance of the vehicle detection pipeline on test images.

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some example images:

![alt text][image5]

To optimize performance of my classifier I varied scales of search windows and value of thresholding function. I also played with regularization parameter `C`  of SVM but default value of 1.0 worked well.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video file](./project_video.mp4)

[![IMAGE ALT TEXT](http://img.youtube.com/vi/xjOohXwxud0/0.jpg)](http://www.youtube.com/watch?v=xjOohXwxud0 "Video Title")

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

### Here is an example of a complete vehicle detection pipeline:

![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Challenges faced during implementation of this project:
- Finding a good balance between density of boxes (both location, scales, overlaps) and thresholding false positives was somewhat challenging. 

Potential problems with the algorithm:
- Boxes wobble and may disappear  
- Clearly, false negatives and false positives are still present
- The algorithm is likely to have worse performance under poor visibility conditions e.g. under dark, bad weather conditions

Possible ways to make the algorithm more robust:
- Incorporate tracking by remembering positions and velocities of cars detected in previous frames. This could improve detections and reduce wobbling.
- Improve accuracy of vehicle detection by testing more machine learning algorithms. This could reduce number of False positives and negatives.

##Car Detection

[//]: # (Image References)
[image1]: ./output_images/car_notcar.jpg
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/optimization_init.jpg
[image4]: ./output_images/sliding_windows.jpg
[image5]: ./output_images/bboxes.jpg
[image6]: ./output_images/heat.jpg
[video1]: ./project_video.mp4

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell 1-10 of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images from GTI and KITTI data sets.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then divided data data into the train set and the test set. GTI data contains time series of car images so avoid having highly correlated images in test and train set I excluded GTI data from the test set. A better, though laborers,  approach would be to manually separate out GTI images into train and test sets.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes in the train set and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I performed feature selection with Bayesian optimization using Gaussian Processes implemented in `skopt` package.
Objective function calculates accuracy (negative value) of car/notcar classification with linear SVM and features scaled to zero mean and unit variance before training the classifier.
I initially searched through the full parameter space including `color_space, orient, hog_channel, cell_per_block, pix_per_cell, spatial_size, hist_bins, spatial_feat, hist_feat'. This initial optimization run showed that `YCrCb` color space leads to best results and including spatial and histogram features leads only to minor improvement. 

After this initial optimization I varied `orient` and `hog\_channel` only for `YCrCb` channel and HOG fores.
Here is the objective function:

![alt text][image3]

The final features are HOG features using the `YCrCb` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After feature selection I trained a linear SVM using full data set. 
The code for this step is contained in the code cell 1-10 of the IPython notebook

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used smaller scales for boxes in the upper and larger scales for boxes in the lower part of the bottom half of the image.
To decide scale sizes, I plotted boxes on test images and make sure that cars fit into them. 
I initially used overlap of 0.5, but the resulting box density was too small for effective false positive removal so so I increased it to 0.75.

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on four scales using YCrCb 3-channel HOG features, which provided a nice result.  
Here are some example images:

![alt text][image5]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


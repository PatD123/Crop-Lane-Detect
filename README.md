# Lane Detection w/ DBSCAN Clustering
This is a lane detection challenge to preprocess live video feed and detect rows of crops as
lanes that a precision Agribot can follow. 

This project heavily relies on several image processing techniques such as Gaussian Blur and Erosion/Dilation, the classic Hough Lines
detector, and an inverse perspective mapping algorithm.

In further pruning, realized that Hough lines when predicting lines can be very erratic. So, I needed a some kind of smoothing or 
clustering technique as Hough Lines often predicted clusters of lines near the left and right lanes. This way, I could cluster but 
also get rid of rather annoying outliers. Herein lies DBSCAN. I used DBSCAN midpoint clustering to average lane detections.

Actually in a Computer Vision class, "Temporal Smoothing" was mentioned and decided to give it a look. Idea is that at the start
of the algorithm, we keep a line buffer, that keeps track of a window of N past, averaged lines (line calculated from DBSCAN).
Now, everytime I generate a new average line from DBSCAN, we slide this window (or rather we update this window so still N lines).
This is done for both the left and right lanes so we keep priors for both. To display the detected lines, we take the mean of the slope
and intercept of the lines in the buffer, allowing for temporal smoothing. Works very well!

The below video is one of the results that I have achieved, with others located in `/agri_videos`. Further, 
overall pipeline is able to localize camera (where the footage is being taken), allowing a future robot
to know it's own position and navigate autonomously.

**If there is anything I could use to improve algorithm, tell me!! Much appreciation!**

Uses OpenCV and Python for the following.

- warp2.py and live_warp.py work with car lanes.
- **ONLY** agri_warp.py and live_agri_warp.py work for agricultural rows of crops

Example of **"NARROW"** crop lane detection.

https://github.com/PatD123/lane_detec/assets/76749942/dda1d859-2e88-408e-854d-e4534c311b26

Example of **"WIDE"** crop lane detection.

https://github.com/PatD123/Crop-Lane-Detect/assets/76749942/01f73df8-684d-44a4-a952-c2f0388b8949

The black text in the upper right corner is dervived from IPM. Idea is that when we project to
birds-eye-view, the pixel-to-meter scaling factor should be the same (directly looking down). With
some handy-dandy linear algebra, we are able to calculate the distance between the current vehicle
trajectory and the reference line (the midline of the detected left and right lanes). 

The purpose of the pixel-2-meter scaling is for input into a Model Predictive Architecture (cool, right?!)
# How to use?
1. Make python virtual env and activate venv.
2. Install requirements.
3. Change directory into /lane_detec/src
4. Run the desired .py file.

For individual frames...
```
py agri_warp.py
```
For full videos...
```
py live_agri_warp.py
```

# New Features
One of the issues that is currently being faced is that sometimes image preprocessing is
just not enough and HoughLines refuses to get the correct line or rather, it gets more
than it should. So in several feature images of the _h_channel, it'll show several clusters
of lanes lines (should be 2, left and right lane line) but also some outlier lines that 
currently affect the depiction of the average line. To fix this, DBSCAN is implemented
to bolster lane detection and become more robust in the face of outliers!!!

- If you wanna argue about K-means or DBSCAN, I'll be happy to shift perspective :)

Another new feature, now we got TEMPORAL SMOOTHING. Basically, we just keep a buffer of 
previous lines (priors). Every frame that we process, we add the current predicted avg line,
add it to the buffer, get rid of a buffered line, then calculate the avg within that buffer.
This achieves very smooth lane movements in a crop environment.

# Goals!!!!
1. Kalman Filtering to track lines more smoothly
2. Find a way to determine when to switch ROIs
   - Partially being solved by DBSCAN - Might have to do better though?

> [!NOTE] 
If anybody can tell me about 2, that'd be cool. But no one looks
at this repo, unfortunado.... 

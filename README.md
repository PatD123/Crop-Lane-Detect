# Lane Detection w/ DBSCAN Clustering
This is a lane detection challenge to preprocess live video feed and detect rows of crops as
lanes that a precision Agribot can follow. This project heavily relies on several image
processing techniques such as Gaussian Blur and Erosion/Dilation, the classic Hough Lines
detector, an inverse perspective mapping algorithm, and a DBSCAN midpoint clustering technique to average lane detections.
The below video is one of the results that I have achieved, with others located in `/agri_videos`. Further, 
overall pipeline is able to localize camera (where the footage is being taken), allowing a future robot
to know it's own position and navigate autonomously.

Uses OpenCV and Python for the following.

- warp2.py and live_warp.py work with car lanes.
- **ONLY** agri_warp.py and live_agri_warp.py work for agricultural rows of crops

https://github.com/PatD123/lane_detec/assets/76749942/dda1d859-2e88-408e-854d-e4534c311b26

(The above however does not include DBSCAN. If you download the current output_video.mp4, that includes the DBSCAN clustering algorithm)
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

# Goals!!!!
1. Kalman Filtering to track lines more smoothly
2. Find a way to determine when to switch ROIs
   - Partially being solved by DBSCAN - Might have to do better though?

> [!NOTE] 
If anybody can tell me about 2, that'd be cool. But no one looks
at this repo, unfortunado.... 

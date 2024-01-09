import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import cv2
import numpy as np
import pickle as pickle
import glob
import math

ORIGINAL_SIZE = 1280, 720
UNWARPED_SIZE = 500, 600

imgs = ["test_images2/straight_lines1.jpg"]
img = mpimg.imread(imgs[0])

# Get a new ROI for image, on which we apply Hough Transform.
# y=425 the upper bound.
# y=665 the lower bound.
# Make a triangle shape to identify lines that go off into vanishing point.
# MAKE NOTE THAT YOU ALWAYS DO WIDTH (X) THEN HEIGHT (Y).
roi_points = np.array([[0, ORIGINAL_SIZE[1] - 55],
                       [ORIGINAL_SIZE[0], ORIGINAL_SIZE[1] - 55],
                       [ORIGINAL_SIZE[0] // 2, ORIGINAL_SIZE[1] - 295]])
roi = np.zeros((720, 1280), np.uint8) # uint8 good for 0-255 so good for small numbers like colors
cv2.fillPoly(roi, [roi_points], 1)

# Employing Gaussian Blur
img = cv2.GaussianBlur(img,(3,3),2)

# Canny + Hough Lines
img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
_h_channel = img_HLS[:, :, 0]
_l_channel  = img_HLS[:, :, 1]
_s_channel = img_HLS[:, :, 2]

low_thresh = 100
high_thresh = 200
# Better to do Canny on lightness channel
edges = cv2.Canny(_l_channel, low_thresh, high_thresh)
new_img = cv2.bitwise_and(edges, edges, mask=roi)
lines = cv2.HoughLinesP(new_img, 2, np.pi/180, 30, None, 180, 120)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness = 1)

plt.imshow(img)
plt.show()






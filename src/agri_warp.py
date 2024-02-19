import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import cv2
import numpy as np
import pickle as pickle
import glob
import math

from DBSCAN import *

ORIGINAL_SIZE = 1280, 720
WARPED_SIZE = 500, 600

imgs = ["../agri_images/0021.jpg"]
img = mpimg.imread(imgs[0])

# Get a new ROI for image, on which we apply Hough Transform.
# y=425 the upper bound (original_size[0] - 295).
# y=665 the lower bound (original_size[1] - 55).
# Make a triangle shape to identify lines that go off into vanishing point.
# MAKE NOTE THAT YOU ALWAYS DO WIDTH (X) THEN HEIGHT (Y).
roi_points = np.array([[0, ORIGINAL_SIZE[1] - 25],
                       [ORIGINAL_SIZE[0], ORIGINAL_SIZE[1] - 25],
                       [ORIGINAL_SIZE[0] // 2 + 10, ORIGINAL_SIZE[1] - 540]])
roi_points = np.array([[0, 360],
                       [1280, 360],
                       [1280, 665],
                       [0, 665]])
roi = np.zeros((720, 1280), np.uint8) # uint8 good for 0-255 so good for small numbers like colors
cv2.fillPoly(roi, [roi_points], 1)

# Employing Gaussian Blur
kernel = np.ones((3,3),np.uint8)
img = cv2.GaussianBlur(img,(3,3),2)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#img = cv2.addWeighted(img, 2.3, np.zeros(img.shape, img.dtype), 0, 4)

# Might need to skip horizontal lines when doing HoughLine

# Canny + Hough Lines
img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
_h_channel = img_HLS[:, :, 0]
_l_channel  = img_HLS[:, :, 1]
_s_channel = img_HLS[:, :, 2]

# print(_l_channel[558][412])

_h_channel = cv2.equalizeHist(_h_channel)

low_thresh = 100
high_thresh = 200
# Better to do Canny on lightness channel
#_h_channel = cv2.erode(_h_channel,kernel,iterations = 1)
_h_channel = cv2.morphologyEx(_h_channel, cv2.MORPH_CLOSE, kernel)
_h_channel = cv2.GaussianBlur(_h_channel,(3,3),2)
_h_channel = cv2.GaussianBlur(_h_channel,(3,3),2)

kernel = np.ones((5,5),np.uint8)
edges = cv2.Canny(_h_channel, high_thresh, low_thresh)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
new_img = cv2.bitwise_and(edges, edges, mask=roi)
plt.imshow(edges)
plt.show()
lines = cv2.HoughLinesP(new_img, 2, np.pi/180, 60, None, 60, 100)

Lhs = np.zeros((2, 2), dtype = np.float32)
Rhs = np.zeros((2, 1), dtype = np.float32)
x_max = 0
x_min = 2555    
left_av = []
right_av = []
dbscan_left = DBSCAN(50, 2)
dbscan_right = DBSCAN(50, 2)
for line in lines:
    for x1, y1, x2, y2 in line:
        # Average out the lines
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - (slope * x1)
        if abs(slope) > 5 or slope == 0 or abs(slope) < 0.1:
            pass
        else:
            if slope > 0:
                dbscan_right.update(line)
            else:
                dbscan_left.update(line)

def fillAvgs(lines):
    l = []
    for i in range(len(lines)):
        line = lines[i]
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - (slope * x1)
        l.append([slope, intercept])
        # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness = 2)
        return l
    
# l has (SLOPE, INTERCEPT) tuples

left_classes = dbscan_left.scan()
left_lines = dbscan_left.return_max(left_classes)
left_av = fillAvgs(left_lines)

right_classes = dbscan_right.scan()
right_lines = dbscan_right.return_max(right_classes)
right_av = fillAvgs(right_lines)

plt.imshow(img)
plt.show()

# Cont. Averaging Lines
left_fitted_av = np.average(left_av, axis=0)
right_fitted_av = np.average(right_av, axis=0)
print(left_fitted_av, right_fitted_av)

# Cont. Averaging Lines
top = ORIGINAL_SIZE[1] - 700
bot = ORIGINAL_SIZE[1] - 55
y1 = ORIGINAL_SIZE[1] - 55
y2 = ORIGINAL_SIZE[1] - 700
left_x1 = int((y1 - left_fitted_av[1]) / left_fitted_av[0])
left_x2 = int((y2 - left_fitted_av[1]) / left_fitted_av[0])
right_x1 = int((y1 - right_fitted_av[1]) / right_fitted_av[0])
right_x2 = int((y2 - right_fitted_av[1]) / right_fitted_av[0])
cv2.line(img, (left_x1, y1), (left_x2, int(y2)), (255, 0, 0), thickness = 2)
cv2.line(img, (right_x1, y1), (right_x2, int(y2)), (255, 0, 0), thickness = 2)
src_pts = np.float32([[0, 360],
                      [1280, 360],
                      [1280, 665.      ],
                      [0, 665.      ]])

dst_pts = np.float32([[0, 0], [WARPED_SIZE[0], 0],
                       [WARPED_SIZE[0], WARPED_SIZE[1]],
                       [0, WARPED_SIZE[1]]])

# Draw Trapezoid
cv2.polylines(img, [src_pts.astype(np.int32)],True, (0,200,100), thickness=5)

plt.imshow(img)
plt.show()

src_pts[0] += [-1, 1]
src_pts[1] += [1, 1]
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped_img = cv2.warpPerspective(img, M, WARPED_SIZE)
f = warped_img
# M is what you will be using from now on.

# Next steps ...
# 1) Find mid-line between the two lane lines, serving as our reference line.
# 2) Warp and then hough for the reference line.
# 3) We already know the position of the current trajectory in warp (still just middle
#    of the image). Do some linear algebra to get distance in pixels.
# 4) Also still need to figure out pixel2meter.
#   a) Split project into 2: 1 for pixel2meter and 2 for running livestream

# Transform averaged points into warped coordinates.
# Endpoints for averaged lines
bot_left = np.array([left_x1, y1], dtype="float32")
top_left = np.array([left_x2, int(y2)], dtype="float32")
bot_right = np.array([right_x1, y1], dtype="float32")
top_right = np.array([right_x2, int(y2)], dtype="float32")

# Transforming above endpoints
bot_left = cv2.perspectiveTransform(np.array([[bot_left]]), M, WARPED_SIZE).squeeze()
top_left = cv2.perspectiveTransform(np.array([[top_left]]), M, WARPED_SIZE).squeeze()
bot_right = cv2.perspectiveTransform(np.array([[bot_right]]), M, WARPED_SIZE).squeeze()
top_right = cv2.perspectiveTransform(np.array([[top_right]]), M, WARPED_SIZE).squeeze()
cv2.line(f, bot_left.astype("int"), top_left.astype("int"), (0, 255, 0), 3)
cv2.line(f, bot_right.astype("int"), top_right.astype("int"), (0, 255, 0), 3)

mid_top = [int((top_left[0] + top_right[0]) / 2),
         int((top_left[1] + top_right[1]) / 2)]
mid_bot = [int((bot_left[0] + bot_right[0]) / 2),
         int((bot_left[1] + bot_right[1]) / 2)]

# Drawing mid-line
cv2.line(f, mid_top, mid_bot, (0, 0, 255), 3)
# Add current car trajectory
traj_bot = [f.shape[1] // 2, 600]
traj_top = [f.shape[1] // 2, 0]
cv2.line(f,traj_bot, traj_top, (0, 0, 255), 3)
x = traj_bot[0]
mid_slope = 0
if mid_top[0] - mid_bot[0] != 0:
    mid_slope = (mid_top[1] - mid_bot[1]) / (mid_top[0] - mid_bot[0])
else:
    mid_slope = (mid_top[1] - mid_bot[1]) * 1000
mid_int = mid_top[1] - mid_top[0] * mid_slope
y = x * mid_slope + mid_int
P = np.array([x, y])

# Calculating pixel distance between averaged line and trajectory line.
PA = np.array(traj_bot) - P
PB = np.array(mid_bot) - P

PB_mag = np.linalg.norm(PB)
PB_unit = PB / PB_mag
A_parallel = np.dot(PA, PB_unit) * PB_unit
A_parallel_pt = A_parallel + P

# Find Intercept

cv2.line(f,traj_bot, A_parallel_pt.astype("int"), (0, 0, 255), 3)

plt.imshow(f)
plt.show()






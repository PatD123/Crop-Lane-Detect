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
WARPED_SIZE = 500, 600

imgs = ["test_images2/frame0312.jpg"]
#imgs = ["test_images2/straight_lines1.jpg"]
img = mpimg.imread(imgs[0])

# Get a new ROI for image, on which we apply Hough Transform.
# y=425 the upper bound (original_size[0] - 295).
# y=665 the lower bound (original_size[1] - 55).
# Make a triangle shape to identify lines that go off into vanishing point.
# MAKE NOTE THAT YOU ALWAYS DO WIDTH (X) THEN HEIGHT (Y).
roi_points = np.array([[300, ORIGINAL_SIZE[1] - 55],
                       [ORIGINAL_SIZE[0] - 300, ORIGINAL_SIZE[1] - 55],
                       [ORIGINAL_SIZE[0] // 2, ORIGINAL_SIZE[1] - 295]])
roi = np.zeros((720, 1280), np.uint8) # uint8 good for 0-255 so good for small numbers like colors
cv2.fillPoly(roi, [roi_points], 1)

# Employing Gaussian Blur
img = cv2.GaussianBlur(img,(3,3),2)

#img = cv2.addWeighted(img, 2.3, np.zeros(img.shape, img.dtype), 0, 4)
#plt.imshow(img)
#plt.show() 

# Might need to skip horizontal lines when doing HoughLine

# Canny + Hough Lines
img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
_h_channel = img_HLS[:, :, 0]
_l_channel  = img_HLS[:, :, 1]
_s_channel = img_HLS[:, :, 2]

# print(_l_channel[558][412])

#ret, p = cv2.threshold(_l_channel,140, 255,cv2.THRESH_BINARY)
#ret, q = cv2.threshold(_l_channel,160,255,cv2.THRESH_BINARY)
#_l_channel = cv2.bitwise_xor(p, q)

_l_channel = cv2.equalizeHist(_l_channel)

low_thresh = 100
high_thresh = 200
# Better to do Canny on lightness channel
edges = cv2.Canny(_l_channel, high_thresh, low_thresh)
new_img = cv2.bitwise_and(edges, edges, mask=roi)
plt.imshow(new_img)
plt.show()
lines = cv2.HoughLinesP(new_img, 2, np.pi/180, 30, None, 180, 120)

Lhs = np.zeros((2, 2), dtype = np.float32)
Rhs = np.zeros((2, 1), dtype = np.float32)
x_max = 0
x_min = 2555    
left_av = []
right_av = []
for line in lines:
    for x1, y1, x2, y2 in line:
        # Find the norm (the distances between the two points)
        normal = np.array([[-(y2-y1)], [x2-x1]], dtype = np.float32) # question about this implementation
        normal = normal / np.linalg.norm(normal)
        pt = np.array([[x1], [y1]], dtype = np.float32)
        outer = np.matmul(normal, normal.T)
        
        Lhs += outer
        Rhs += np.matmul(outer, pt) #use matmul for matrix multiply and not dot product

        # Average out the lines
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - (slope * x1)
        if abs(slope) > 5:
            pass
        elif slope > 0:
            right_av.append([slope, intercept])
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness = 2)
        else:
            left_av.append([slope, intercept])
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness = 2)

        x_iter_max = max(x1, x2)
        x_iter_min = min(x1, x2)
        x_max = max(x_max, x_iter_max)
        x_min = min(x_min, x_iter_min)
width = x_max - x_min
print('width : ', width)
# Calculate Vanishing Point
vp = np.matmul(np.linalg.inv(Lhs), Rhs)
vp = vp.flatten()

print('vp is : ', vp)
plt.plot(vp[0], vp[1], 'c^')

plt.imshow(img)
plt.show()

# Cont. Averaging Lines
left_fitted_av = np.average(left_av, axis=0)
right_fitted_av = np.average(right_av, axis=0)

# Drawing up source points for perspective warps
def find_pt_inline(p1, p2, y):
    """
    Here we use point-slope formula in order to find a point that is present on the line
    that passes through our vanishing point (vp). 
    input: points p1, p2, and y. They come is as tuples [x, y]
    We then use the point-slope formula: y - b = m(x - a)
    y: y-coordinate of desired point on the line
    x: x-coordinate of desired point on the line
    m: slope
    b: y-coordinate of p1
    a: x-coodrinate of p1
    x = p1x + (1/m)(y - p1y)
    """
    m_inv = (p2[0] - p1[0]) / float(p2[1] - p1[1])
    Δy = (y - p1[1])
    x = p1[0] + m_inv * Δy
    return [x, y]

top = vp[1] + 65
bot = ORIGINAL_SIZE[1] - 55

#print(900*right_av[0][0] + right_av[0][1])
#cv2.line(img, (0, -38), (900, 1188), (0, 0, 255), 3)


# Cont. Averaging Lines
y1 = bot
y2 = top
left_x1 = int((y1 - left_fitted_av[1]) / left_fitted_av[0])
left_x2 = int((y2 - left_fitted_av[1]) / left_fitted_av[0])
right_x1 = int((y1 - right_fitted_av[1]) / right_fitted_av[0])
right_x2 = int((y2 - right_fitted_av[1]) / right_fitted_av[0])
cv2.line(img, (left_x1, y1), (left_x2, int(y2)), (255, 0, 0), thickness = 2)
cv2.line(img, (right_x1, y1), (right_x2, int(y2)), (255, 0, 0), thickness = 2)

# Make a large width so that you can grab the lines on the challenge video
width = 300

p1 = [vp[0] - width/2, top]
p2 = [vp[0] + width/2, top]
p3 = find_pt_inline(p2, vp, bot)
p4 = find_pt_inline(p1, vp, bot)

src_pts = np.float32([p1, p2, p3, p4])
src_pts = np.float32([[ 462.2556, 487.81726 ],
                      [ 812.2556, 487.81726 ],
                      [1289.286, 665.      ],
                      [ -14.774837, 665.      ]])

dst_pts = np.float32([[0, 0], [WARPED_SIZE[0], 0],
                       [WARPED_SIZE[0], WARPED_SIZE[1]],
                       [0, WARPED_SIZE[1]]])

# Draw Trapezoid
cv2.polylines(img, [src_pts.astype(np.int32)],True, (0,200,100), thickness=5)
plt.plot(p1[0], p1[1])
plt.plot(p2[0], p2[1])
plt.plot(p3[0], p3[1])
plt.plot(p4[0], p4[1])
plt.title('Trapezoid For Perspective Transform')

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





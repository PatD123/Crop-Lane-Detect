import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import cv2
import numpy as np
import pickle as pickle
import glob
import math

ORIGINAL_SIZE = 640, 480
UNWARPED_SIZE = 500, 600

imgs = ["test_images1/1.jpg"]
img = mpimg.imread(imgs[0])

#plt.imshow(img)
#plt.show()

#################################################################

Lhs = np.zeros((2, 2), dtype = np.float32)
Rhs = np.zeros((2, 1), dtype = np.float32)
x_max = 0
x_min = 2555
# The 2 lines come from Agronav. You do the vp calcs and the warps
# with the two lines. The Agronav ref line is in the image. The two lines
# are not and just used to warp.
lines = [[0, 170, 630, 310],
         [0, 300, 630, 170]]
a = 0

for line in lines:
  x1 = line[0]
  y1 = line[1]
  x2 = line[2]
  y2 = line[3]

  a+=1
  # Find the norm (the distances between the two points)
  normal = np.array([[-(y2-y1)], [x2-x1]], dtype = np.float32) # question about this implementation [a, b] . [-b, a] = 0
  normal = normal / np.linalg.norm(normal)  # = sqrt(a1^2 + a2^2 + a3^2 + . . . )

  # Normal is 2x1 and is a vector v . w = vT . w

  pt = np.array([[x1], [y1]], dtype = np.float32)

  outer = np.matmul(normal, normal.T)

  Lhs += outer
  Rhs += np.matmul(outer, pt) #use matmul for matrix multiply and not dot product
  # Just keep summing these guys up like how you and they derived it.

  #cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness = 30)

  x_iter_max = max(x1, x2)
  x_iter_min = min(x1, x2)
  x_max = max(x_max, x_iter_max)
  x_min = min(x_min, x_iter_min)

# Calculate Vanishing Point
vp = np.matmul(np.linalg.inv(Lhs), Rhs)
#plt.plot(vp[0], vp[1], 'c^')
#plt.imshow(img)
#plt.title('Vanishing Point visualization')
#plt.show()
print("Done!")

###############################################################
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
    return [np.array(x), np.array(np.float32([y]))]


top = vp[1] + 65
bot = ORIGINAL_SIZE[1] - 40
print(bot)

# Make a large width so that you can grab the lines on the challenge video
width = 500

p1 = [vp[0] - width/2, top]
p2 = [vp[0] + width/2, top]
p3 = find_pt_inline(p2, vp, bot)
p4 = find_pt_inline(p1, vp, bot)

print(p1, p3)
src_pts = np.float32([p1, p2, p3, p4])

# Mapping one corner to 0,0 so the warped image is the entire image now.
dst_pts = np.float32([[0, 0], [UNWARPED_SIZE[0], 0],
                       [UNWARPED_SIZE[0], UNWARPED_SIZE[1]],
                       [0, UNWARPED_SIZE[1]]])

# Draw Trapezoid
cv2.polylines(img, [src_pts.astype(np.int32)],True, (0,200,100), thickness=5)
#plt.plot(p1[0], p1[1], 'r+')
#plt.plot(p2[0], p2[1], 'c^')
#plt.plot(p3[0], p3[1], 'r^')
#plt.plot(p4[0], p4[1], 'g^')
#plt.title('Trapezoid For Perspective Transform')
#plt.imshow(img)
#plt.show()

############################################################################################

# H is the homography matrix
M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, UNWARPED_SIZE)
plt.imshow(warped)
plt.show()

############################################################################################

reg, mask = cv2.threshold(warped[:,:,0], 245, 255, cv2.THRESH_BINARY)
plt.imshow(mask)
plt.show()

lines = cv2.HoughLinesP(mask, 0.5, np.pi/180, 100, 50, 50)
line = lines[0].flatten()

# Transform the center bottom of the screen using the transfom matrix
pixel_coordinates = np.array([[ORIGINAL_SIZE[0] / 2, bot]], dtype=np.float32)
pixel_coordinates = pixel_coordinates.reshape(-1, 1, 2)
coords = cv2.perspectiveTransform(pixel_coordinates, M, UNWARPED_SIZE).squeeze()
#print(coords)

A = line[2:]
B = line[:2]
AB = np.subtract(B, A)
AC = np.subtract(coords, A)
#print(AB, AC)

mag = np.linalg.norm(AB)
tmp = np.dot(AB, AC) / (mag**2)
AB_parallel = tmp * AB
inter = AB_parallel + A
print("Distance in pixels:", math.dist(coords, inter))
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

imgs = ["test_images2/straight_lines1.jpg"]
img = mpimg.imread(imgs[0])

# Getting a new ROI for image
print(img.shape)

# Canny + Hough Lines
img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
_h_channel = img_HLS[:, :, 0]
_l_channel  = img_HLS[:, :, 1]
_s_channel = img_HLS[:, :, 2]

low_thresh = 100
high_thresh = 200
# Better to do Canny on lightness channel
edges = cv2.Canny(_l_channel, low_thresh, high_thresh)

plt.imshow(edges)
plt.show()







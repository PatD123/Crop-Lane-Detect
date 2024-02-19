import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import cv2
import numpy as np
import pickle as pickle
import glob
import math
from moviepy.editor import *
from DBSCAN import *

ORIGINAL_SIZE = 1280, 720
WARPED_SIZE = 500, 600

left_buffer = []
right_buffer = []
buffer_size = 5

def getROI():
    #roi_points = np.array([[0, ORIGINAL_SIZE[1] - 25],
    #                   [ORIGINAL_SIZE[0], ORIGINAL_SIZE[1] - 25],
    #                   [ORIGINAL_SIZE[0] // 2 + 10, ORIGINAL_SIZE[1] - 540]])
    roi_points = np.array([[0, 360],
                       [1280, 360],
                       [1280, 665],
                       [0, 665]])
    roi = np.zeros((720, 1280), np.uint8) # uint8 good for 0-255 so good for small numbers like colors
    cv2.fillPoly(roi, [roi_points], 1)
    return roi

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

def getLines(img):
    roi = getROI()

    # Employing Gaussian Blur
    img = cv2.GaussianBlur(img,(3,3),2)
    kernel = np.ones((3,3),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # Might need to skip horizontal lines when doing HoughLine

    # Canny + Hough Lines
    img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    _h_channel = img_HLS[:, :, 0]
    _l_channel  = img_HLS[:, :, 1]
    _s_channel = img_HLS[:, :, 2]

    #ret, p = cv2.threshold(_l_channel,140, 255,cv2.THRESH_BINARY)
    #ret, q = cv2.threshold(_l_channel,160,255,cv2.THRESH_BINARY)
    #_l_channel = cv2.bitwise_xor(p, q)

    _h_channel = cv2.equalizeHist(_h_channel)

    low_thresh = 100
    high_thresh = 200
    # Better to do Canny on lightness channel
    _h_channel = cv2.morphologyEx(_h_channel, cv2.MORPH_CLOSE, kernel)
    _h_channel = cv2.GaussianBlur(_h_channel,(3,3),2)
    _h_channel = cv2.GaussianBlur(_h_channel,(3,3),2)

    kernel = np.ones((5,5),np.uint8)
    edges = cv2.Canny(_h_channel, low_thresh, high_thresh)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    new_img = cv2.bitwise_and(edges, edges, mask=roi)
    lines = cv2.HoughLinesP(new_img, 2, np.pi/180, 60, None, 60, 100)
    return lines

def main(img):
    global left_buffer, right_buffer, buffer_size

    lines = getLines(img)

    if lines is None:
        return img

    left_av = []
    right_av = []
    dbscan_left = DBSCAN(50, 2)
    dbscan_right = DBSCAN(50, 2)

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Average out the lines
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) > 5 or slope == 0 or abs(slope) < 0.1:
                pass
            else:
                if slope > 0:
                    dbscan_right.update(line)
                else:
                    dbscan_left.update(line)

    left_classes = dbscan_left.scan()
    left_lines = dbscan_left.return_max(left_classes)
    left_av = fillAvgs(left_lines)

    right_classes = dbscan_right.scan()
    right_lines = dbscan_right.return_max(right_classes)
    right_av = fillAvgs(right_lines)

    # Cont. Averaging Lines
    left_fitted_av = []
    right_fitted_av = []
    if left_av:
        left_fitted_av = np.average(left_av, axis=0)
    else:
        return img
    if right_av:
        right_fitted_av = np.average(right_av, axis=0)
    else:
        return img
    
    # Adding average line to buffer
    def add_line_to_buffer(line_buffer, line):
        line_buffer.append(line)
        return line_buffer[-buffer_size:]
    left_buffer = add_line_to_buffer(left_buffer, left_fitted_av)
    right_buffer = add_line_to_buffer(right_buffer, right_fitted_av)

    # Get mean of buffered lines
    left_fitted_av = np.mean(left_buffer, axis=0)
    right_fitted_av = np.mean(right_buffer, axis=0)

    top = ORIGINAL_SIZE[1] - 700
    bot = ORIGINAL_SIZE[1] - 55
    y1 = ORIGINAL_SIZE[1] - 55
    y2 = ORIGINAL_SIZE[1] - 500
    try:
        left_x1 = int((y1 - left_fitted_av[1]) / left_fitted_av[0])
        left_x2 = int((y2 - left_fitted_av[1]) / left_fitted_av[0])
        right_x1 = int((y1 - right_fitted_av[1]) / right_fitted_av[0])
        right_x2 = int((y2 - right_fitted_av[1]) / right_fitted_av[0])
        cv2.line(img, (left_x1, y1), (left_x2, int(y2)), (255, 0, 0), thickness = 2)
        cv2.line(img, (right_x1, y1), (right_x2, int(y2)), (255, 0, 0), thickness = 2)
    except:
        return img

    # Hard-coded src and dest pts
    src_pts = np.float32([[0, 360],
                      [1280, 360],
                      [1280, 665.      ],
                      [0, 665.      ]])
    dst_pts = np.float32([[0, 0], [WARPED_SIZE[0], 0],
                       [WARPED_SIZE[0], WARPED_SIZE[1]],
                       [0, WARPED_SIZE[1]]])
    # Draw ROI
    cv2.polylines(img, [src_pts.astype(np.int32)],True, (0,200,100), thickness=2)

    src_pts[0] += [-1, 1]
    src_pts[1] += [1, 1]
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(img, M, WARPED_SIZE)
    
    f = warped_img

    # Transform averaged points into warped coordinates.
    # Endpoints for averaged lines
    bot_left = np.array([left_x1, y1], dtype="float32")
    top_left = np.array([left_x2, int(y2)], dtype="float32")
    bot_right = np.array([right_x1, y1], dtype="float32")
    top_right = np.array([right_x2, int(y2)], dtype="float32")

    # Transforming above endpoints
    def transformPoints(p):
        return cv2.perspectiveTransform(np.array([[p]]), M, WARPED_SIZE).squeeze()
    
    bot_left = transformPoints(bot_left)
    top_left = transformPoints(top_left)
    bot_right = transformPoints(bot_right)
    top_right = transformPoints(top_right)
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
    vec = traj_bot - A_parallel_pt
    vec_mag = np.linalg.norm(vec)
    cv2.putText(img, str(vec_mag), (0, 200), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0, 0, 0), 2, cv2.LINE_AA) 

    return img

if __name__ == "__main__":
    clip  = VideoFileClip("../agri_videos/0123.mp4")
    mod_clip = clip.fl_image(main)

    mod_clip.write_videofile("../agri_videos/output_video.mp4", audio=False)
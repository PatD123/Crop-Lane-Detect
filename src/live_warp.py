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

ORIGINAL_SIZE = 1280, 720
WARPED_SIZE = 500, 600

def getROI():
    roi_points = np.array([[200, ORIGINAL_SIZE[1] - 55],
                       [ORIGINAL_SIZE[0] - 200, ORIGINAL_SIZE[1] - 55],
                       [ORIGINAL_SIZE[0] // 2, ORIGINAL_SIZE[1] - 295]])
    roi = np.zeros((720, 1280), np.uint8) # uint8 good for 0-255 so good for small numbers like colors
    cv2.fillPoly(roi, [roi_points], 1)
    return roi

def getLines(img):
    roi = getROI()

    # Employing Gaussian Blur
    img = cv2.GaussianBlur(img,(3,3),2)
    # Might need to skip horizontal lines when doing HoughLine

    # Canny + Hough Lines
    img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    _h_channel = img_HLS[:, :, 0]
    _l_channel  = img_HLS[:, :, 1]
    _s_channel = img_HLS[:, :, 2]

    #ret, p = cv2.threshold(_l_channel,140, 255,cv2.THRESH_BINARY)
    #ret, q = cv2.threshold(_l_channel,160,255,cv2.THRESH_BINARY)
    #_l_channel = cv2.bitwise_xor(p, q)

    _l_channel = cv2.equalizeHist(_l_channel)

    low_thresh = 100
    high_thresh = 200
    # Better to do Canny on lightness channel
    edges = cv2.Canny(_l_channel, low_thresh, high_thresh)
    new_img = cv2.bitwise_and(edges, edges, mask=roi)
    lines = cv2.HoughLinesP(new_img, 2, np.pi/180, 30, None, 180, 120)
    return lines

def main(img):
    lines = getLines(img)

    if lines is None:
        return img

    left_av = []
    right_av = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Average out the lines
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            if abs(slope) > 5:
                pass
            elif slope > 0:
                right_av.append([slope, intercept])
                #cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness = 2)
            else:
                left_av.append([slope, intercept])
                #cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness = 2)
    
    # Hard-coded VP
    vp = [664.16125, 419.31366]
    top = vp[1] + 65
    bot = ORIGINAL_SIZE[1] - 55

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

    y1 = bot
    y2 = top
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
    src_pts = np.float32([[ 486.2556, 488.81726], 
                          [ 788.2556, 488.81726], 
                          [1196.1389, 665.     ],
                          [  78.37237, 665.     ]])
    dst_pts = np.float32([[0, 0], [WARPED_SIZE[0], 0],
                       [WARPED_SIZE[0], WARPED_SIZE[1]],
                       [0, WARPED_SIZE[1]]])
    # Draw Trapezoid
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
    vec = traj_bot - A_parallel_pt
    vec_mag = np.linalg.norm(vec)
    cv2.putText(img, str(vec_mag), (0, 200), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0, 255, 0), 2, cv2.LINE_AA) 

    return img

if __name__ == "__main__":
    clip  = VideoFileClip("urmom.mp4")
    mod_clip = clip.fl_image(main)

    mod_clip.write_videofile("output_video.mp4", audio=False)
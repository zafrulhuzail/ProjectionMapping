import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle
import sys

#------------------------------------------------------------------------------
# Settings -> please configure
#------------------------------------------------------------------------------
bool_fullscreen =True
height = 720
width = 1280

aruco_size = 100
border_bits = 1
white_border_px = 5   # <<< NEW: white border thickness

#------------------------------------------------------------------------------
# Generating the aruco-board
#------------------------------------------------------------------------------
print("Press Q to quit")

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)

# generate four aruco grid elements for the four corners
aruco_corner_marker_list = []

for aruco_id in range(4):
    # Generate marker
    aruco_mrk = aruco.generateImageMarker(
        aruco_dict,
        aruco_id,
        aruco_size,
        borderBits=border_bits
    )
    aruco_mrk = cv2.cvtColor(aruco_mrk, cv2.COLOR_GRAY2RGB)

    # --------------------------------------------------
    # ADD WHITE BORDER AROUND MARKER
    # --------------------------------------------------
    aruco_mrk = cv2.copyMakeBorder(
        aruco_mrk,
        white_border_px,
        white_border_px,
        white_border_px,
        white_border_px,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)   # white border
    )

    aruco_corner_marker_list.append(aruco_mrk)

#------------------------------------------------------------------------------
# Setup output image to fullscreen
#------------------------------------------------------------------------------
cv2.namedWindow('Beamer_Window', cv2.WINDOW_NORMAL)
if bool_fullscreen==True:
    cv2.setWindowProperty(    'Beamer_Window',    cv2.WND_PROP_FULLSCREEN,    cv2.WINDOW_FULLSCREEN)

#------------------------------------------------------------------------------
# Generate the actual aruco-image for the beamer
#------------------------------------------------------------------------------
BeamerImage = np.zeros((height, width, 3), np.uint8)
BeamerImage[:] = (0, 0, 0)  # black background

h, w, _ = aruco_corner_marker_list[0].shape

# Top-left
BeamerImage[0:h, 0:w] = aruco_corner_marker_list[0]

# Top-right
BeamerImage[0:h, width - w:width] = aruco_corner_marker_list[1]

# Bottom-left
BeamerImage[height - h:height, 0:w] = aruco_corner_marker_list[2]

# Bottom-right
BeamerImage[height - h:height, width - w:width] = aruco_corner_marker_list[3]

#------------------------------------------------------------------------------
# Show & save
#------------------------------------------------------------------------------
cv2.imshow('Beamer_Window', BeamerImage)
cv2.imwrite("BeamerImage.png", BeamerImage)

cv2.waitKey()
cv2.destroyAllWindows()

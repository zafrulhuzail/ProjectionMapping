import numpy as np
import cv2
import cv2.aruco as aruco

# Select type of aruco marker (size)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
size=150
borderBits=1
white_border_px = 10   # <<< NEW: white border thickness

# Create an image from the marker
# second param is ID number
# last param is total image size
for number in range(4,15+1, 1): #(start, stop,step)
    aruco_mrk = aruco.generateImageMarker(aruco_dict, number, size, borderBits=borderBits)
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
    cv2.imwrite(f"test_marker{number}.jpg", aruco_mrk)

    # # Display the image to us
    cv2.imshow('frame', aruco_mrk)
    cv2.waitKey(500)


# Exit on any key
cv2.destroyAllWindows()

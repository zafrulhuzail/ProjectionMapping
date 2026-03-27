# System information:
# - Linux Mint 18.1 Cinnamon 64-bit
# - Python 2.7 with OpenCV 3.2.0

import numpy
import cv2
from cv2 import aruco
import pickle
import glob


#Setting up the detection
# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 8
CHARUCOBOARD_COLCOUNT = 6

# NEW way to get dictionary
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

# NEW CharucoBoard constructor
CHARUCO_BOARD = aruco.CharucoBoard(
    (CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
    squareLength=0.042,
    markerLength=0.021,
    dictionary=ARUCO_DICT
)


# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = [] # Corners discovered in all images processed
ids_all = [] # Aruco ids corresponding to corners discovered
image_size = None # Determined at runtime


# This requires a set of images or a video taken with the camera you want to calibrate
# I'm using a set of images taken with the camera with the naming convention:
# 'camera-pic-of-charucoboard-<NUMBER>.jpg'
# All images used should be the same size, which if taken with the same camera shouldn't be a problem
images = glob.glob('./pictures/calibration_images/*.jpg')


#------------------------------------------------------------------------------
# Load all images
#------------------------------------------------------------------------------
# Loop through images glob'ed
for iname in images:
    img = cv2.imread(iname)  # Open the image
    #cv2.imshow('input image', img) #display images before calibration
    #cv2.waitKey()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Convert image to grayscale

    #--- start tracking -> find aruco markers in the loade images
    corners, ids, _ = aruco.detectMarkers( image=gray,  dictionary=ARUCO_DICT)

    #-- check results
    # skip image if no markers are found
    if ids is None or len(ids) == 0:
        print("No ArUco markers detected in:", iname)
        continue

    #Elsewise, outline the found markers in our query image
    img = aruco.drawDetectedMarkers(
            image=img,
            corners=corners)

    # Get charuco corners and ids from detected aruco markers
    response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD)

    # validate output -> first check for interpolation
    if response is None or response <= 0:
        print("ChArUco interpolation failed for:", iname)
        continue

    # lastly check if corners and IDs are properly found
    if charuco_corners is None or charuco_ids is None:
        continue

    # if all of the above is true, check if each of the image has enough corner points (20)
    if response > 20:
        # Add these corners and ids to our calibration arrays
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)

        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)

        # If our image size is unknown, set it now
        if not image_size:
            image_size = gray.shape[::-1]

        # Reproportion the image, maxing width or height at 1000
        proportion = max(img.shape) / 1000.0
        img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))

        # Pause to display each image, waiting for key press
        cv2.imshow('Charuco board', img)
        if cv2.waitKey(0) == ord('q'):
            break

    else:
        print("Not able to detect a charuco board in image: {}".format(iname))

# Destroy any open CV windows
cv2.destroyAllWindows()


#------------------------------------------------------------------------------
# do check on the tracked results
#------------------------------------------------------------------------------
# Take the images that actually have marker in them

# Make sure at least one image was found
if len(images) < 1:
    # Calibration failed because there were no images, warn the user
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()

# Make sure we were able to calibrate on at least one charucoboard by checking
# if we ever determined the image size
if not image_size:
    # Calibration failed because we didn't see any charucoboards of the PatternSize used
    print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit for failure
    exit()


#------------------------------------------------------------------------------
# Finally get the calinration
#------------------------------------------------------------------------------
# Do points based camera calibration
calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)

# Print results of the calibtraiont, namely matrix and distortion coefficient
print(f"cameraMatrix:\n{cameraMatrix}\n")
print(f"distCoeffs:\n{distCoeffs}\n")

#--- Save values
f = open('./calibration/ProCamCalibration.pckl', 'wb')
pickle.dump((calibration, cameraMatrix, distCoeffs, rvecs, tvecs), f) # pickle was used -> soon to be replaced
f.close()

# Print to console our success
print('Calibration successful. Calibration file used: {}'.format('./calibration/calibration_files/ProCamCalibration.pckl'))

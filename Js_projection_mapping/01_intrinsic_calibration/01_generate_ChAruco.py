import cv2
import cv2.aruco as aruco

# Create ChArUco board, which is a set of Aruco markers in a chessboard setting
# meant for calibration
# the following call gets a ChArUco board of tiles 6 wide X 8 tall

# ChAruco board variables
CHARUCOBOARD_COLCOUNT = 6
CHARUCOBOARD_ROWCOUNT = 8

white_border_px = 10

# NEW way to get dictionary
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

# NEW CharucoBoard constructor
gridboard = aruco.CharucoBoard(
    (CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
    squareLength=0.04,
    markerLength=0.02,
    dictionary=ARUCO_DICT
)

# Create an image from the gridboard
img = gridboard.generateImage(outSize=(988, 1400))
# --------------------------------------------------
# ADD WHITE BORDER AROUND MARKER
img = cv2.copyMakeBorder(
    img,
    white_border_px,
    white_border_px,
    white_border_px,
    white_border_px,
    cv2.BORDER_CONSTANT,
    value=(255, 255, 255)   # white border
)

cv2.imwrite("./pictures/test_charuco.jpg", img)

# Display the image to us
cv2.imshow('Gridboard', img)
# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()

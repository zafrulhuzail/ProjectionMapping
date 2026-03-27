import cv2
import numpy as np

REFERENCE_IMAGE = "reference.jpg"   # put this in the same folder as tuto1.py
GOOD_MATCH_DISTANCE = 50            # lower = stricter
MATCH_THRESHOLD = 25                # increase if false positives happen

ref = cv2.imread(REFERENCE_IMAGE, cv2.IMREAD_GRAYSCALE)

if ref is None:
    print(f"Could not load reference image: {REFERENCE_IMAGE}")
    exit()

# created ORB objects and tells it to keep about 1000 keypoints/features
orb = cv2.ORB_create(nfeatures=1000)
# Detect: kp_ref = list of important points 
# Compute:des_ref = the numeric fingerprints for those points
kp_ref, des_ref = orb.detectAndCompute(ref, None)

if des_ref is None or len(kp_ref) < 10:
    print("Not enough features found in the reference image.")
    print("Use a clearer reference image with more visible details/text.")
    exit()

# compare each descriptor from the reference image, find the closest matches
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# =========================
# OPEN CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

print("Press Q to quit.")
print("Hold the card inside the green box.")

while True:
    #ret = success flag, either '0' or '1'
    #frame = the actual image coming from camera (NumPy array presenting the pixels)
    # heigt x width x color channels (480, 640, 3)
    # Keep grabbing images from the webcam one after another, if the webcam fails to
    # provide an image, print an error and stop the program loop
    ret, frame = cap.read()         
    if not ret:                     # ret == True'1', frame was read successfully
        print("Could not read frame from camera.")
        break










reference_path = "reference.jpg"
test_path = "test.jpg"

ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
print("OpenCV version:", cv2.__version__)

if ref is None:
    print("Could not load reference image")
    sys.exit()

if test is None:
    print("Could not load test image")
    sys.exit()

diff = cv2.absdiff(ref, test)

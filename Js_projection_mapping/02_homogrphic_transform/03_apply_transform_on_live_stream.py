import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle


def warp_image(image, H, output_width, output_height):
    warped = cv2.warpPerspective(
        image,
        H,
        (output_width, output_height)
    )
    return warped




# -------------------------------------------------
# Open video stream (0 = default camera)
# -------------------------------------------------
#output_height=  720
#output_width = 1280


cap = cv2.VideoCapture(1) # je nach Rechner: 0/1/2 durchprobieren (z.B. 0=NVIDIA Broadcast)

relative_cam_calibration_path = '../01_intrinsic_calibration/calibration/ProCamCalibration.pckl'
bool_load_cam_calib= True

relative_homographic_tranform_path= '../02_homogrphic_transform/homographic_tranform.pckl'


#------------------------------------------------------------------------------
# Loding camera calibration
#------------------------------------------------------------------------------
if bool_load_cam_calib== True:
    # Check for camera calibration data
    if not os.path.exists(relative_cam_calibration_path):
        print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
        exit()
    else:
        with open(relative_cam_calibration_path, 'rb') as f:
            pickle_cam_calib = pickle.load(f)
        calibration, cameraMatrix, distCoeffs, rvecs, tvecs = pickle_cam_calib

        # sanity check calibration content
        if distCoeffs is None or calibration is None or cameraMatrix is None :
            print("Could not interpret loaded calibration. Recalibrate camera intrinsics")
            exit()



#------------------------------------------------------------------------------
# Loading dewarp
#------------------------------------------------------------------------------
if bool_load_cam_calib== True:
    # Check for camera calibration data
    if not os.path.exists(relative_homographic_tranform_path):
        print("You need to calculate a camera transform. See '02'-directory for details.")
        exit()
    else:
        with open(relative_homographic_tranform_path, 'rb') as f:
            pickle_homgraphic_transform = pickle.load(f)
        H, output_width, output_height = pickle_homgraphic_transform

        # sanity check calibration content
        if H is None or output_width is None or output_height is None :
            print("Could not interpret homographic_tranform. Rerun your transform")
            exit()


# -------------------------------------------------
# ArUco setup (must match generation dictionary)
# -------------------------------------------------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)

parameters = aruco.DetectorParameters()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

print("Press ESC to exit")



# -------------------------------------------------
# Video loop
# -------------------------------------------------
if not cap.isOpened():
    raise IOError("Cannot open video stream")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # apply camera calibration (undistort)
    if bool_load_cam_calib:
        frame = cv2.undistort(frame, cameraMatrix, distCoeffs)

    # warp transform
    dewarped_img=warp_image(frame, H, output_width, output_height)

    cv2.imshow("Warped / Rectified View", dewarped_img)




    # ---------------------------------------------
    # Detect markers
    # ---------------------------------------------
    gray = cv2.cvtColor(dewarped_img, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )
    # do stuff with this information



    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
cap.release()
cv2.destroyAllWindows()

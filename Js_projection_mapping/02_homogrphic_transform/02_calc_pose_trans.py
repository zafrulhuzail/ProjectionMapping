import cv2
import cv2.aruco as aruco
import numpy as np
import pickle
import os




def compute_homography_from_aruco(
    ids,
    corners,
    output_width,
    output_height
):
    """
    ids: numpy array of shape (N,)
    corners: list of detected marker corners
    output_width, output_height: target image size
    """

    # Build lookup: id -> marker center
    marker_centers = {}

    for marker_id, corner in zip(ids, corners):
        center = np.mean(corner[0], axis=0)
        marker_centers[int(marker_id)] = center

    # Ensure all required markers exist
    required_ids = [0, 1, 2, 3]
    if not all(mid in marker_centers for mid in required_ids):
        raise ValueError("Not all required ArUco markers (0,1,2,3) detected")

    # Source points (camera image)
    src_pts = np.array([
        marker_centers[0],  # top-left
        marker_centers[1],  # top-right
        marker_centers[2],  # bottom-left
        marker_centers[3],  # bottom-right
    ], dtype=np.float32)

    # Destination points (perfect rectangle)
    dst_pts = np.array([
        [0, 0],
        [output_width - 1, 0],
        [0, output_height - 1],
        [output_width - 1, output_height - 1],
    ], dtype=np.float32)

    # Compute homography
    H, status = cv2.findHomography(src_pts, dst_pts)

    return H


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
output_height=  720
output_width = 1280


cap = cv2.VideoCapture(0) # check if we want a different marker

relative_cam_calibration_path = '../01_intrinsic_calibration/calibration/ProCamCalibration.pckl'
bool_load_cam_calib= True


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

    #load and apply calibration


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------
    # Detect markers
    # ---------------------------------------------
    corners, ids, rejected = aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    # ---------------------------------------------
    # Sort detections by marker ID
    # ---------------------------------------------
    if ids is not None:
        ids = ids.flatten()

        sorted_data = sorted(zip(ids, corners), key=lambda x: x[0])
        sorted_ids, sorted_corners = zip(*sorted_data)

        sorted_ids = np.array(sorted_ids)
        sorted_corners = list(sorted_corners)

        # Draw markers
        aruco.drawDetectedMarkers(frame, sorted_corners, sorted_ids)

        # Draw centers and labels
        for marker_id, corner in zip(sorted_ids, sorted_corners):
            center = np.mean(corner[0], axis=0).astype(int)
            cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ID {marker_id}",
                tuple(center + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )
        # ---------------------------------------------
        # Calc transfrom
        # ---------------------------------------------
        #if cv2.waitKey(1) & 0xFF == ord('h'):
        if len(ids)==4:
            print ("calc homography")

            H= compute_homography_from_aruco   (ids, corners, output_width, output_height)

            #[optional] save it using pickel:
            f = open('homographic_tranform.pckl', 'wb')
            pickle.dump((H, output_width, output_height), f) # pickle was used -> soon to be replaced
            f.close()

            #apply transform
            warped_img=warp_image(frame, H, output_width, output_height)


            cv2.imshow("Warped / Rectified View", warped_img)


        # Optional: access markers explicitly by ID
        marker_by_id = {mid: crn for mid, crn in zip(sorted_ids, sorted_corners)}
        if all(k in marker_by_id for k in [0, 1, 2, 3]):
            # Example: compute something when all 4 are visible
            pass

    # ---------------------------------------------
    # Show frame
    # ---------------------------------------------
    cv2.imshow("ArUco Video Detection (Sorted by ID)", frame)


    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
cap.release()
cv2.destroyAllWindows()

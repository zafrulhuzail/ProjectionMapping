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
cap = cv2.VideoCapture(1) # je nach Rechner: 0/1/2 durchprobieren (z.B. 0=NVIDIA Broadcast)

# camera calibration
relative_cam_calibration_path = '../01_intrinsic_calibration/calibration/ProCamCalibration.pckl'
bool_load_cam_calib= False  # True = mit Kamera-Kalibrierung (Phase 1), False = ueberspringen (fuer Top-Down-Setup ok)

# Dewarping
relative_homographic_tranform_path= '../02_homogrphic_transform/homographic_tranform.pckl'
bool_load_transform = True

# aruco tracking
overlay_img_path= "overlay.png"
bool_draw_detection_test= True

bool_fullscreen= False
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
if bool_load_transform== True:
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

#------------------------------------------------------------------------------
# Setup output image to fullscreen
#------------------------------------------------------------------------------
cv2.namedWindow('Beamer_Window', cv2.WINDOW_NORMAL)
if bool_fullscreen==True:
    cv2.setWindowProperty( 'Beamer_Window',    cv2.WND_PROP_FULLSCREEN,    cv2.WINDOW_FULLSCREEN)



#------------------------------------------------------------------------------
# Loading a transparent image for overlay
#------------------------------------------------------------------------------
def overlay_transparent(background, overlay, x, y):
    """
    background: BGR image
    overlay: BGRA image (with alpha channel)
    x, y: CENTER position of overlay
    """

    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]

    # Convert center position to top-left
    x0 = int(x - ov_w / 2)
    y0 = int(y - ov_h / 2)

    # Compute overlay region bounds
    x1 = max(x0, 0)
    y1 = max(y0, 0)
    x2 = min(x0 + ov_w, bg_w)
    y2 = min(y0 + ov_h, bg_h)

    # Check if overlay is completely outside
    if x1 >= x2 or y1 >= y2:
        return background

    # Corresponding overlay region
    ov_x1 = x1 - x0
    ov_y1 = y1 - y0
    ov_x2 = ov_x1 + (x2 - x1)
    ov_y2 = ov_y1 + (y2 - y1)

    # Extract regions
    roi = background[y1:y2, x1:x2]
    overlay_crop = overlay[ov_y1:ov_y2, ov_x1:ov_x2]

    overlay_rgb = overlay_crop[:, :, :3]
    overlay_alpha = overlay_crop[:, :, 3:] / 255.0

    # Alpha blending
    roi[:] = (1.0 - overlay_alpha) * roi + overlay_alpha * overlay_rgb

    return background




# Load transparent image (must be PNG with alpha)
overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
if overlay_img is None or overlay_img.shape[2] != 4:
    raise IOError("Overlay image must be RGBA (PNG with alpha)")



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

    if bool_load_cam_calib == True:
        #apply camera calibration
        frame = cv2.undistort(
            frame,
            cameraMatrix,
            distCoeffs
        )

    # warp transform
    if bool_load_transform== True:
        dewarped_img=warp_image(frame, H, output_width, output_height)
    else:
        dewarped_img= frame

    #cv2.imshow("Warped / Rectified View", dewarped_img)

    #create a result image based on the the dewardped image
    BeamerImage = np.zeros(dewarped_img.shape,  np.uint8)

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
    if ids is not None:
        ids = ids.flatten()

        sorted_data = sorted(zip(ids, corners), key=lambda x: x[0])
        sorted_ids, sorted_corners = zip(*sorted_data)

        sorted_ids = np.array(sorted_ids)
        sorted_corners = list(sorted_corners)

        # Calc centers
        for marker_id, corner in zip(sorted_ids, sorted_corners):
            center = np.mean(corner[0], axis=0).astype(int)
            if marker_id==4:
                cv2.circle(dewarped_img, tuple(center), 15, (0, 0, 255), -1)


            #sample code to overlay an image with alpha blending
            if marker_id==5:
                cv2.circle(dewarped_img, tuple(center), 5, (255, 255, 255), -1)

                # Optional: resize overlay
                overlay_img = cv2.resize(overlay_img, (300, 300))
                 # Put overlay at position (x, y)
                dewarped_img = overlay_transparent(dewarped_img, overlay_img, x= tuple(center)[0], y= tuple(center)[1])
            else:
                #draw detection beamer
                cv2.circle(BeamerImage, tuple(center), 30, (0, 255, 255), -1)
                cv2.circle(dewarped_img, tuple(center), 30, (0, 255, 255), -1) # draw on live stream


                #draw detection beamer
                if bool_draw_detection_test== True:
                    cv2.putText(
                        BeamerImage,
                        f"ID {marker_id}",
                        tuple(center + np.array([10, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA
                    )
                #darw on live strream
                if bool_draw_detection_test== True:
                    cv2.putText(
                        dewarped_img,
                        f"ID {marker_id}",
                        tuple(center + np.array([10, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA
                    )

            # Draw markers
            aruco.drawDetectedMarkers(BeamerImage, sorted_corners, sorted_ids)
            aruco.drawDetectedMarkers(dewarped_img, sorted_corners, sorted_ids)


    cv2.imshow("Beamer_Window", BeamerImage)
    cv2.imshow("output", dewarped_img)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
cap.release()
cv2.destroyAllWindows()

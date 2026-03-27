from pathlib import Path
import os
import pickle

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------
# Settings
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs" / "detect" / "train5" / "weights" / "best.pt"
HOMOGRAPHY_PATH = ROOT / "Js_projection_mapping" / "02_homogrphic_transform" / "homographic_tranform.pckl"
CALIBRATION_PATH = ROOT / "Js_projection_mapping" / "01_intrinsic_calibration" / "calibration" / "ProCamCalibration.pckl"

CAMERA_INDEX = 1  # change to 0/1/2 if needed
CONFIDENCE = 0.25
USE_CAMERA_CALIBRATION = False
USE_HOMOGRAPHY = True
FULLSCREEN_PROJECTOR = False


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def warp_image(image, H, output_width, output_height):
    return cv2.warpPerspective(image, H, (output_width, output_height))


def create_fire_overlay(width, height):
    width = max(8, int(width))
    height = max(8, int(height))
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    # base flame body
    cv2.ellipse(
        overlay,
        (width // 2, int(height * 0.66)),
        (max(6, width // 3), max(8, int(height * 0.28))),
        0,
        0,
        360,
        (0, 90, 255, 180),
        -1,
    )
    cv2.ellipse(
        overlay,
        (width // 2, int(height * 0.55)),
        (max(5, width // 4), max(7, int(height * 0.22))),
        0,
        0,
        360,
        (0, 165, 255, 210),
        -1,
    )
    cv2.ellipse(
        overlay,
        (width // 2, int(height * 0.45)),
        (max(4, width // 6), max(5, int(height * 0.14))),
        0,
        0,
        360,
        (210, 240, 255, 235),
        -1,
    )

    # tip of the flame
    pts = np.array(
        [
            [width // 2, max(0, int(height * 0.04))],
            [int(width * 0.25), int(height * 0.48)],
            [int(width * 0.75), int(height * 0.48)],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(overlay, pts, (40, 200, 255, 170))

    return overlay


def alpha_blend_rgba(background_bgr, overlay_rgba, x, y):
    bg_h, bg_w = background_bgr.shape[:2]
    ov_h, ov_w = overlay_rgba.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(bg_w, x + ov_w)
    y1 = min(bg_h, y + ov_h)

    if x0 >= x1 or y0 >= y1:
        return background_bgr

    ov_x0 = x0 - x
    ov_y0 = y0 - y
    ov_x1 = ov_x0 + (x1 - x0)
    ov_y1 = ov_y0 + (y1 - y0)

    roi = background_bgr[y0:y1, x0:x1]
    overlay_crop = overlay_rgba[ov_y0:ov_y1, ov_x0:ov_x1]

    rgb = overlay_crop[:, :, :3].astype(np.float32)
    alpha = (overlay_crop[:, :, 3:4].astype(np.float32) / 255.0)
    roi[:] = ((1.0 - alpha) * roi.astype(np.float32) + alpha * rgb).astype(np.uint8)
    return background_bgr


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    print(f"[info] loading YOLO model: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    camera_matrix = dist_coeffs = None
    if USE_CAMERA_CALIBRATION:
        print(f"[info] loading camera calibration: {CALIBRATION_PATH}")
        if not CALIBRATION_PATH.exists():
            raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_PATH}")
        with open(CALIBRATION_PATH, "rb") as f:
            calibration, camera_matrix, dist_coeffs, rvecs, tvecs = pickle.load(f)

    H = output_width = output_height = None
    if USE_HOMOGRAPHY:
        print(f"[info] loading homography: {HOMOGRAPHY_PATH}")
        if not HOMOGRAPHY_PATH.exists():
            raise FileNotFoundError(f"Homography file not found: {HOMOGRAPHY_PATH}")
        with open(HOMOGRAPHY_PATH, "rb") as f:
            H, output_width, output_height = pickle.load(f)
        print(f"[info] homography output size: {output_width}x{output_height}")

    print(f"[info] opening camera index {CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")

    cv2.namedWindow("Beamer_Window", cv2.WINDOW_NORMAL)
    cv2.namedWindow("debug_output", cv2.WINDOW_NORMAL)
    if FULLSCREEN_PROJECTOR:
        cv2.setWindowProperty("Beamer_Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_counter = 0
    print("[info] started. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        frame_counter += 1
        if not ret or frame is None:
            print("[error] failed to read frame")
            break

        if USE_CAMERA_CALIBRATION and camera_matrix is not None and dist_coeffs is not None:
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        if USE_HOMOGRAPHY and H is not None:
            dewarped_img = warp_image(frame, H, output_width, output_height)
        else:
            dewarped_img = frame

        results = model.predict(dewarped_img, conf=CONFIDENCE, verbose=False)
        result = results[0]
        annotated = result.plot()
        beamer_image = np.zeros(dewarped_img.shape, np.uint8)

        num_boxes = 0 if result.boxes is None else len(result.boxes)
        if frame_counter == 1 or frame_counter % 60 == 0:
            print(f"[debug] frame={frame_counter} detections={num_boxes}")

        if result.boxes is not None:
            names = result.names
            for idx, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]

                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                overlay = create_fire_overlay(box_w, box_h)
                alpha_blend_rgba(beamer_image, overlay, x1, y1)

                cv2.rectangle(beamer_image, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(
                    beamer_image,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            beamer_image,
            f"Detections: {num_boxes}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Beamer_Window", beamer_image)
        cv2.imshow("debug_output", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

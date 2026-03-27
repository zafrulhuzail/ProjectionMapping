from pathlib import Path
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

CAMERA_INDEX = 0  # change to 0/1/2 if needed
CONFIDENCE = 0.025
USE_CAMERA_CALIBRATION = False
USE_HOMOGRAPHY = True
FULLSCREEN_PROJECTOR = False
LABEL_BORDER_COLOR = (0, 220, 255)
LABEL_FILL_COLOR = (0, 70, 110)
LABEL_TEXT_COLOR = (255, 255, 255)
MIN_BOX_SIZE_FOR_LABEL = 36


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def warp_image(image, H, output_width, output_height):
    return cv2.warpPerspective(image, H, (output_width, output_height))


def fit_text_scale(text, max_width, max_height, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for scale in [1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        if text_w <= max_width and (text_h + baseline) <= max_height:
            return scale, text_w, text_h, baseline
    (text_w, text_h), baseline = cv2.getTextSize(text, font, 0.35, max(1, thickness - 1))
    return 0.35, text_w, text_h, baseline


def draw_label_above_box(image, x1, y1, x2, y2, text):
    img_h, img_w = image.shape[:2]
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < MIN_BOX_SIZE_FOR_LABEL or box_h < MIN_BOX_SIZE_FOR_LABEL:
        return image

    thickness = 2 if box_w >= 140 else 1
    max_text_w = max(30, int(box_w * 1.1))
    max_text_h = max(18, int(box_h * 0.35))
    scale, text_w, text_h, baseline = fit_text_scale(text, max_text_w, max_text_h, thickness)

    label_w = text_w + 18
    label_h = text_h + baseline + 16
    gap = 10

    label_x1 = x1 + max(0, (box_w - label_w) // 2)
    label_x1 = max(0, min(label_x1, img_w - label_w))

    # preferred position: above the box
    label_y1 = y1 - label_h - gap
    if label_y1 < 0:
        # fallback: just below the top edge inside image, still near the box
        label_y1 = min(img_h - label_h, y2 + gap)
    if label_y1 < 0 or label_y1 + label_h > img_h:
        return image

    label_x2 = label_x1 + label_w
    label_y2 = label_y1 + label_h

    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), LABEL_FILL_COLOR, -1)
    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), LABEL_BORDER_COLOR, 2)

    # connector line from label to box
    line_start = (label_x1 + label_w // 2, label_y2)
    line_end = (x1 + box_w // 2, y1)
    cv2.line(image, line_start, line_end, LABEL_BORDER_COLOR, 2)

    text_x = label_x1 + max(8, (label_w - text_w) // 2)
    text_y = label_y1 + max(text_h + 6, (label_h + text_h) // 2 - baseline)
    text_y = min(text_y, label_y2 - baseline - 4)

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        LABEL_TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    return image


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
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]

                x1 = max(0, min(x1, beamer_image.shape[1] - 1))
                y1 = max(0, min(y1, beamer_image.shape[0] - 1))
                x2 = max(0, min(x2, beamer_image.shape[1] - 1))
                y2 = max(0, min(y2, beamer_image.shape[0] - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(beamer_image, (x1, y1), (x2, y2), LABEL_BORDER_COLOR, 2)
                draw_label_above_box(beamer_image, x1, y1, x2, y2, label)

                cv2.putText(
                    annotated,
                    f"projected: {label} {conf:.2f}",
                    (x1, max(20, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
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

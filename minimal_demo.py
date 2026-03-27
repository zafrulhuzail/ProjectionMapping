"""
Minimal test: webcam window + second window for the projector (HDMI display).

Run:
  python -m venv .venv
  .venv\\Scripts\\activate
  pip install -r requirements.txt
  python minimal_demo.py

If the Beamer window lands on the wrong screen, set the horizontal offset of your
second monitor (typical: width of primary display, e.g. 1920):
  python minimal_demo.py --beamer-x 1920   # typical: primary width if Beamer is on the right

Keys:
  f = toggle fullscreen on Beamer window
  q / ESC = quit
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal camera + beamer test window.")
    p.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default 0).",
    )
    p.add_argument(
        "--beamer-x",
        type=int,
        default=0,
        help="Move Beamer window to this screen X offset (e.g. 1920 if Beamer is right of 1920px-wide laptop).",
    )
    p.add_argument(
        "--beamer-w",
        type=int,
        default=1280,
        help="Beamer window width before fullscreen.",
    )
    p.add_argument(
        "--beamer-h",
        type=int,
        default=720,
        help="Beamer window height before fullscreen.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}. Try --camera 1")
        return

    win_cam = "Camera (previews)"
    win_beamer = "Beamer / HDMI (drag to projector or press f)"
    cv2.namedWindow(win_cam, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_beamer, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_beamer, args.beamer_w, args.beamer_h)
    cv2.moveWindow(win_beamer, args.beamer_x, 0)

    fullscreen = False
    print("Keys: f = fullscreen Beamer window | q / ESC = quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame grab failed; exiting.")
            break

        # --- PC preview (native camera size, capped for small laptops) ---
        preview = frame
        h, w = preview.shape[:2]
        max_w = 960
        if w > max_w:
            scale = max_w / float(w)
            preview = cv2.resize(preview, (int(w * scale), int(h * scale)))
        cv2.imshow(win_cam, preview)

        # --- Beamer canvas: black + embedded live view + colored frame (feedback test) ---
        canvas = np.zeros((args.beamer_h, args.beamer_w, 3), dtype=np.uint8)
        small = cv2.resize(frame, (640, 360))
        y0 = (args.beamer_h - 360) // 2
        x0 = (args.beamer_w - 640) // 2
        canvas[y0 : y0 + 360, x0 : x0 + 640] = small
        cv2.rectangle(canvas, (0, 0), (args.beamer_w - 1, args.beamer_h - 1), (0, 255, 0), 8)
        cv2.putText(
            canvas,
            "BEAMER OK - green border",
            (24, args.beamer_h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win_beamer, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("f"):
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win_beamer, cv2.WND_PROP_FULLSCREEN, prop)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

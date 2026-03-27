import numpy as np
import cv2
import os
import pickle
from ultralytics import YOLO


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
CAMERA_INDEX = 1                # externe USB-Kamera (0=NVIDIA Broadcast, 1/2 probieren)
BRACKET_LEN  = 20               # Laenge der Eck-Brackets in Pixel
BRACKET_T    = 3                # Strichstaerke
bool_fullscreen = False

relative_homographic_tranform_path = '../02_homogrphic_transform/homographic_tranform.pckl'

# Platzhalter-Modell (vortrainiert, 80 Klassen)
# --> spaeter tauschen gegen: YOLO("mein_modell.pt")
model = YOLO("yolov8n.pt")

# Farbe pro Klassen-ID (zyklisch)
COLORS = [
    (0, 255, 128),
    (0, 200, 255),
    (255, 80,  80),
    (255, 200,  0),
    (180,  0, 255),
]


# -------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------
def warp_image(image, H, output_width, output_height):
    return cv2.warpPerspective(image, H, (output_width, output_height))


def draw_brackets(img, x1, y1, x2, y2, color, length=BRACKET_LEN, thickness=BRACKET_T):
    # oben-links
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # oben-rechts
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # unten-links
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # unten-rechts
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


# -------------------------------------------------
# Homographie laden
# -------------------------------------------------
if not os.path.exists(relative_homographic_tranform_path):
    print("Homographie fehlt. Erst 02_calc_pose_trans.py ausfuehren.")
    exit()

with open(relative_homographic_tranform_path, 'rb') as f:
    H, output_width, output_height = pickle.load(f)

# -------------------------------------------------
# Kamera + Fenster
# -------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise IOError(f"Kamera {CAMERA_INDEX} nicht verfuegbar")

cv2.namedWindow('Beamer_Window', cv2.WINDOW_NORMAL)
if bool_fullscreen:
    cv2.setWindowProperty('Beamer_Window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Press ESC to exit")

# -------------------------------------------------
# Loop
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    dewarped_img = warp_image(frame, H, output_width, output_height)
    BeamerImage  = np.zeros(dewarped_img.shape, np.uint8)

    # YOLO detection
    results = model(dewarped_img, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        color  = COLORS[cls_id % len(COLORS)]

        draw_brackets(BeamerImage, x1, y1, x2, y2, color)

    cv2.imshow("Beamer_Window", BeamerImage)
    cv2.imshow("output", dewarped_img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

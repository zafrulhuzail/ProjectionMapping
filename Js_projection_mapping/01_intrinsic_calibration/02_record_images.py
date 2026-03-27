import cv2
import os
from datetime import datetime


print("Pleas get your ChAruco board ready -> either print it on you beamer/display or print it out")

# ====== CONFIG ======
save_dir = "./pictures/calibration_images"   # change this if you want another folder
camera_index = 0              # 0 is default webcam
# ====================

# Create folder if it doesn't exist
os.makedirs(save_dir, exist_ok=True)
print(save_dir)
# Open webcam
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

print("📸 Press SPACE to save image")
print("❌ Press 'q' to quit")

img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE key → save image
    if key == 32:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"img_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)

        cv2.imwrite(filepath, frame)
        img_count += 1
        print(f"✅ Saved: {filepath}")

    # 'q' key → quit
    elif key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

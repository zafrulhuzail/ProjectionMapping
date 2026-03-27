import cv2

# Change this if your webcam is not camera 0
CAMERA_INDEX = 0

# Common webcam resolutions to test (width, height)
RESOLUTIONS = [
    (320, 240),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 800),
    (1366, 768),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("Testing supported resolutions:\n")

supported = []

for width, height in RESOLUTIONS:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if actual_width == width and actual_height == height:
        supported.append((width, height))
        print(f"✅ Supported: {width} x {height}")
        ret, frame = cap.read()
        cv2.imshow(f"supported_res {width} x {height} ", frame)
    else:
        print(f"❌ Not supported: {width} x {height} (got {actual_width} x {actual_height})")


cv2.waitKey(8000) # pause for 8 seconds
cap.release()

print("\nSummary of supported resolutions:")
for w, h in supported:
    print(f"  - {w} x {h}")

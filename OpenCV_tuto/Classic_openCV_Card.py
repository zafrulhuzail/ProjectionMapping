import cv2
import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================
REFERENCE_FOLDER = "references"   # folder containing one clean image per card
CAMERA_INDEX = 0                  # usually 0 for default webcam

CARD_WIDTH = 300                  # warped card width
CARD_HEIGHT = 420                 # warped card height

MIN_CARD_AREA = 5000              # ignore tiny contours (tiny edges, noise, background objects)
MAX_CARDS_TO_PROCESS = 10         # limit to 10 cards for speed
MIN_MATCHES = 20                  # below this, label becomes "Unknown"
SHOW_DEBUG_WINDOWS = True         # show edges and warped card windows

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def order_points(pts):
    """
    Take 4 corner points and return them in this order:
    top-left, top-right, bottom-right, bottom-left
    when OpenCV detects a rectangle, the 4 points can come in any random order
    """
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)         # (x + y)top left has the smallest sum, bottom left has the largest sum
    diff = np.diff(pts, axis=1) # (y - x) top right has the smallest difference, bottom - left has the largest difference 

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def warp_card(image, corners, width=CARD_WIDTH, height=CARD_HEIGHT):
    """
    Perspective-warp the detected card so it becomes a flat, front-facing card.
    Webcam card image may be rotated, tilted, trapezoid
    this function mathematically transforms it into a clean rectangle
    """
    rect = order_points(corners)        # make sure corners are in the right order

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)          # Computes the transformation matrix
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def preprocess_for_features(image_bgr):
    """
    Convert image to grayscale and lightly normalize it.
    Prepares image before feature extraction
    Color is not often necessary for ORB
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)            # ORB might detect unstable features caused by
    return gray                                         # sensor noise, compression artifacts, very tiny texture variation
                                                        # Too much blur would remove useful details

def load_reference_cards(folder, orb):
    """
    Load all reference images from a folder.
    For each reference card:
    - resize to a fixed size
    - convert to grayscale
    - compute ORB keypoints/descriptors

    Returns a list of dictionaries.
    """
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(f"Reference folder not found: {folder}")

    references = []

    for path in folder_path.iterdir():
        if path.suffix.lower() not in supported_ext:
            continue

        img = cv2.imread(str(path))
        if img is None:
            print(f"Warning: could not load {path}")
            continue

        img = cv2.resize(img, (CARD_WIDTH, CARD_HEIGHT))        # resizes ref image as the warped webcan cards
        gray = preprocess_for_features(img)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        label = path.stem.replace("_", " ").replace("-", " ")       # gets filename and turns it into a readable label

        references.append({
            "label": label,
            "image": img,
            "gray": gray,
            "kp": keypoints,
            "desc": descriptors
        })

        print(f"Loaded reference: {label} | keypoints: {0 if keypoints is None else len(keypoints)}")

    if not references:
        raise ValueError("No valid reference images found in the references folder.")

    return references


def find_card_contours(frame):
    """
    Detect possible card contours in the current frame.

    Strategy:
    - grayscale
    - blur
    - edge detect
    - close gaps
    - find external contours
    - keep large 4-corner contours
    """
    # Step 1: Greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # Step 2: Gaussian blur     
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #Step3 3: Canny edge detection, finds places where brightness changes sharply
    edges = cv2.Canny(blur, 60, 150)

    #Step 4: morphology
    kernel = np.ones((3, 3), np.uint8)
    #Step4.1: makes white edge pixels thicker
    edges = cv2.dilate(edges, kernel, iterations=1)
    # Step 4.2: Closing: dilation followed by erosion
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 5: Find contours
    # contour = a boundary line around a white region in the binary edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_contours = []
    # Step 6: Search by area(larger contours first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CARD_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)   

        # simplifies the contour to a polygon
        # approx, 4 corners, shape is a convex = likely a card
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = approx.reshape(4, 2)
            card_contours.append(corners)

        if len(card_contours) >= MAX_CARDS_TO_PROCESS:
            break

    return card_contours, edges


def match_card(warped_card, references, orb, matcher):
    """
    Compare the warped camera card to every reference card using ORB descriptors.

    Returns:
    - best label
    - best score (number of good matches)
    """
    gray = preprocess_for_features(warped_card)
    kp_card, desc_card = orb.detectAndCompute(gray, None)

    if desc_card is None or kp_card is None or len(kp_card) < 8:
        return "Unknown", 0

    best_label = "Unknown"
    best_score = 0

    #Step 3: compare against all references, for each ref, it gets the saved descriptors
    for ref in references:
        desc_ref = ref["desc"]
        if desc_ref is None:
            continue

        # KNN matching + Lowe ratio test
        # for each descriptor in the webcam card, find the 2 nearest descriptors in the reference image
        raw_matches = matcher.knnMatch(desc_card, desc_ref, k=2)

        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        #Step 6: score = the number of accepted good matches
        score = len(good_matches)

        if score > best_score:
            best_score = score          #highest score found so far
            best_label = ref["label"]   # best label found so far

    #Step 8: Threshold
    if best_score < MIN_MATCHES:
        return "Unknown", best_score

    return best_label, best_score


def draw_label(frame, corners, label, score):
    """
    Draw polygon around the card and show its label.
    """
    corners_int = corners.astype(int)
    cv2.polylines(frame, [corners_int], True, (0, 255, 0), 3)

    ordered = order_points(corners)
    top_left = tuple(ordered[0].astype(int))

    text = f"{label} ({score})"
    text_pos = (top_left[0], max(30, top_left[1] - 10))

    cv2.putText(frame, text, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


# ============================================================
# MAIN
# ============================================================

def main():
    # ORB detector
    orb = cv2.ORB_create(
        nfeatures=1500,
        scaleFactor=1.2,
        nlevels=8
    )

    # Matcher for ORB descriptors
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Load references
    references = load_reference_cards(REFERENCE_FOLDER, orb)
    print(f"\nLoaded {len(references)} reference cards.\n")

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Optional webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        display = frame.copy()      #make a copy for drawing

        # Find possible cards in the frame
        card_contours, edges = find_card_contours(frame)

        first_warped = None

        for corners in card_contours:
            warped = warp_card(frame, corners, CARD_WIDTH, CARD_HEIGHT)

            if first_warped is None:
                first_warped = warped.copy()

            label, score = match_card(warped, references, orb, matcher)
            draw_label(display, corners, label, score)

        # Status text
        cv2.putText(display, f"Detected cards: {len(card_contours)}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Turbine Card Detector", display)

        if SHOW_DEBUG_WINDOWS:
            cv2.imshow("Edges Debug", edges)
            if first_warped is not None:
                cv2.imshow("Warped Card Debug", first_warped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
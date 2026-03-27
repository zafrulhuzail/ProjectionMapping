import cv2
import numpy as np

# ----------------------------
# CHANGE THESE FILE NAMES
# ----------------------------
REFERENCE_PATH = "fanrotor_ori.jpeg"            # card you trust
TEST_PATH = "fanrotor_env_false.jpeg"            # camera image / new card

# ----------------------------
# HELPERS
# ----------------------------

# function that takes in path input and returns the image
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

# 
def align_to_reference(ref_img, test_img, max_features=4000, keep_percent=0.25):
    """
    Align test_img to ref_img using ORB keypoints + homography.
    Returns warped test image with same size as ref_img.
    """
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)

    if des1 is None or des2 is None:
        raise RuntimeError("Could not find enough ORB features.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < 10:
        raise RuntimeError("Not enough matches to align images.")

    matches = sorted(matches, key=lambda x: x.distance)

    keep = max(10, int(len(matches) * keep_percent))
    matches = matches[:keep]

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_test = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_test, pts_ref, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography could not be computed.")

    aligned = cv2.warpPerspective(test_img, H, (ref_img.shape[1], ref_img.shape[0]))
    return aligned

def crop_phrase_region(img):
    """
    Crop the bottom phrase region containing:
    'suck - squeeze - bang - blow'
    Coordinates are relative, so they work across sizes.
    Tune these if needed.
    """
    h, w = img.shape[:2]

    # bottom strip where the underlined phrase is
    y1 = int(h * 0.86)
    y2 = int(h * 0.98)
    x1 = int(w * 0.08)
    x2 = int(w * 0.94)

    return img[y1:y2, x1:x2]

def detect_underlined_word(phrase_roi, debug_name="debug"):
    """
    Detect which of the 4 words is underlined.
    Strategy:
    - grayscale
    - threshold to binary inverse
    - keep mostly horizontal strokes with morphology
    - count underline pixels in 4 fixed x-zones
    """
    gray = cv2.cvtColor(phrase_roi, cv2.COLOR_BGR2GRAY)

    # Threshold: black text/underline becomes white
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Keep long-ish horizontal structures (underline strokes)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    underline_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # Slight close to reconnect broken underline segments
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    underline_mask = cv2.morphologyEx(underline_mask, cv2.MORPH_CLOSE, close_kernel)

    h, w = underline_mask.shape

    # Word zones across the phrase ROI
    # You may tune these boundaries once you test on real images
    zones = {
        "suck":    (0.00, 0.20),
        "squeeze": (0.20, 0.53),
        "bang":    (0.53, 0.77),
        "blow":    (0.77, 1.00),
    }

    scores = {}
    vis = cv2.cvtColor(underline_mask, cv2.COLOR_GRAY2BGR)

    for word, (xa, xb) in zones.items():
        x1 = int(w * xa)
        x2 = int(w * xb)

        zone = underline_mask[:, x1:x2]
        score = cv2.countNonZero(zone)
        scores[word] = score

        cv2.rectangle(vis, (x1, 0), (x2, h - 1), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{word}:{score}",
            (x1 + 5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    detected_word = max(scores, key=scores.get)

    return detected_word, scores, binary, underline_mask, vis

def compare_picture_region(ref_img, aligned_test_img):
    """
    Optional: compare the main picture region and mark changed areas.
    Good for debugging whether something visual changed above too.
    """
    h, w = ref_img.shape[:2]

    # Main picture area, not text
    y1 = int(h * 0.18)
    y2 = int(h * 0.70)
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)

    ref_roi = ref_img[y1:y2, x1:x2]
    test_roi = aligned_test_img[y1:y2, x1:x2]

    gray1 = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(test_roi, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    vis = ref_roi.copy()
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area > 300:
            x, y, ww, hh = cv2.boundingRect(c)
            cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 0, 255), 2)

    return diff, thresh, vis

# ----------------------------
# MAIN
# ----------------------------
ref_img = load_image(REFERENCE_PATH)
test_img = load_image(TEST_PATH)

aligned_test = align_to_reference(ref_img, test_img)

# Detect underlined word in both images
ref_phrase = crop_phrase_region(ref_img)
test_phrase = crop_phrase_region(aligned_test)

ref_word, ref_scores, ref_binary, ref_underlines, ref_vis = detect_underlined_word(ref_phrase, "ref")
test_word, test_scores, test_binary, test_underlines, test_vis = detect_underlined_word(test_phrase, "test")

print("Reference underlined word:", ref_word)
print("Reference scores:", ref_scores)
print("Test underlined word:", test_word)
print("Test scores:", test_scores)

if ref_word == test_word:
    print("Result: same underlined word.")
else:
    print(f"Result: DIFFERENT underlined word -> reference={ref_word}, test={test_word}")

# Optional: compare the main picture region too
pic_diff, pic_thresh, pic_vis = compare_picture_region(ref_img, aligned_test)

# ----------------------------
# SHOW RESULTS
# ----------------------------
cv2.imshow("Reference", ref_img)
cv2.imshow("Aligned Test", aligned_test)

cv2.imshow("Reference Phrase ROI", ref_phrase)
cv2.imshow("Test Phrase ROI", test_phrase)

cv2.imshow("Reference Binary", ref_binary)
cv2.imshow("Test Binary", test_binary)

cv2.imshow("Reference Underline Mask", ref_underlines)
cv2.imshow("Test Underline Mask", test_underlines)

cv2.imshow("Reference Zones", ref_vis)
cv2.imshow("Test Zones", test_vis)

cv2.imshow("Picture Diff", pic_diff)
cv2.imshow("Picture Diff Threshold", pic_thresh)
cv2.imshow("Picture Diff Boxes", pic_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()
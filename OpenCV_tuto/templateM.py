import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original and template images
original_img = cv2.imread('fanrotor_env_false.jpeg')
template_img = cv2.imread('fanrotor_ori.jpeg')

# Check template size relative to the original image
original_h, original_w = original_img.shape[:2]
template_h, template_w = template_img.shape[:2]

# Scaling algorithm to find the best match
scales = np.arange(0.5, 0.01, -0.01) 
best_val = -1
best_loc = None
best_match = None
best_scale = 1.0
result = None
scale_values = []  
correlation_values = []

for scale in scales:
    resized_template = cv2.resize(template_img, (int(template_w * scale), int(template_h * scale)))

    # Check if the template is larger than the original image
    if resized_template.shape[0] > original_h or resized_template.shape[1] > original_w:
        continue

    # Apply normalized correlation coefficient matching
    result = cv2.matchTemplate(original_img, resized_template, cv2.TM_CCOEFF_NORMED)
    
    # Find the maximum correlation value and its location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # If a new max_val is found, update the best match
    if max_val > best_val:
        best_val = max_val
        best_loc = max_loc
        best_match = resized_template
        best_scale = scale

    # Save the scale and correlation value
    scale_values.append(scale)
    correlation_values.append(max_val)

# Check best value for threshold
threshold = 0.87
if best_val >= threshold:
    top_left = best_loc
    h, w = best_match.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(original_img, top_left, bottom_right, (0, 255, 0), 2)
else:
    result = np.zeros_like(original_img)

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title("Matched Image")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB))
plt.title("Template Image")

plt.subplot(2, 2, 3)
plt.imshow(result, cmap='plasma')
plt.title("Match Confidence Map")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.plot(scale_values, correlation_values)
plt.title("Correlation Values for Different Scales")
plt.xlabel("Scale")
plt.ylabel("Correlation Value")
plt.grid(True)
plt.tight_layout(pad=0)
plt.show()
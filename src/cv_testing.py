import cv2
import numpy as np

# Load your image
image_path = 'img/house.png'  # Replace with your image file
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

def show(title, image):
    cv2.imshow(title, image)
    # Wait for a key press, return if ESC is pressed

    # Loop until ESC key or window is closed
    while True:
        key = cv2.waitKey(1) & 0xFF
        # ESC key to exit
        if key == 27:
            break
        # Detect if window is closed (optional but robust)
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


# 1. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show('Grayscale', gray)

# Inverted Canny Edges
edges = cv2.Canny(gray, 100, 200)
# inverted_edges = cv2.bitwise_not(edges)
show('Inverted Canny Edges', edges)


# 4. Binary Thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
inverted_edges = cv2.bitwise_not(thresh)
show('Threshold', inverted_edges)

cv2.imwrite('global_map.png', inverted_edges)


# # 5. Dilation
# kernel = np.ones((5, 5), np.uint8)
# dilated = cv2.dilate(thresh, kernel, iterations=1)
# show('Dilation', dilated)

# # 6. Erosion
# eroded = cv2.erode(dilated, kernel, iterations=1)
# show('Erosion', eroded)

# # 7. Rotation
# (h, w) = img.shape[:2]
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, 45, 1.0)
# rotated = cv2.warpAffine(img, M, (w, h))
# show('Rotation (45 deg)', rotated)

# # 8. Scaling
# scaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
# show('Scaling (50%)', scaled)

# # 9. Affine Transformation
# pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
# M_affine = cv2.getAffineTransform(pts1, pts2)
# affine = cv2.warpAffine(img, M_affine, (w, h))
# show('Affine Transform', affine)

# # 10. Perspective Transformation
# pts1 = np.float32([[321, 113], [671, 114], [321,609], [626, 609]])
# pts2 = np.float32([[347,146], [646,144], [336,575], [608,575]])
# M_persp = cv2.getPerspectiveTransform(pts1, pts2)
# perspective = cv2.warpPerspective(img, M_persp, (w, h))
# show('Perspective Transform', perspective)
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

def preprocess_variants(img_gray):
    results = {}

    # Original
    results['Original'] = img_gray

    # Gaussian Blur
    gauss = cv2.GaussianBlur(img_gray, (5, 5), 1)
    results['Gaussian Blur'] = gauss

    # Bilateral Filter
    bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
    results['Bilateral Filter'] = bilateral

    # Adaptive Threshold on Bilateral
    adapt_thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 5)
    results['Adaptive Threshold'] = adapt_thresh

    # Morphological Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_open = cv2.morphologyEx(adapt_thresh, cv2.MORPH_OPEN, kernel)
    results['Morph Open'] = morph_open

    # Skeletonization
    binary = (morph_open > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    results['Skeletonized'] = skeleton

    # Canny Edge Detection
    edges = cv2.Canny(img_gray, 50, 150)
    results['Canny Edges'] = edges

    return results

def show_results(images_dict):
    n = len(images_dict)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    for i, (title, img) in enumerate(images_dict.items()):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    path = 'src/img/global_map.png'  # Change to your image path
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Could not load image: {path}")
        return

    results = preprocess_variants(img_gray)
    show_results(results)

if __name__ == '__main__':
    main()

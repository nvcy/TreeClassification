# src/features/lbp_extractor.py

import os
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from PIL import Image

# Directories
MULTIVIEW_DIR = "outputs/multi_view/test"
FEATURES_DIR = "outputs/features/test"
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

def extract_lbp(image):
    """
    Extract LBP histogram from a grayscale image
    """
    gray = rgb2gray(np.array(image))
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, METHOD)
    # Compute histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_features_from_views(view_dir):
    """
    Extract LBP features for all multi-view images
    """
    features = []
    labels = []
    for class_name in os.listdir(view_dir):
        class_path = os.path.join(view_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for f in os.listdir(class_path):
            if not f.endswith(".png"):
                continue
            img_path = os.path.join(class_path, f)
            image = Image.open(img_path)
            lbp_hist = extract_lbp(image)
            features.append(lbp_hist)
            labels.append(class_name)
    features = np.array(features)
    labels = np.array(labels)
    return features, labels

def main():
    features, labels = extract_features_from_views(MULTIVIEW_DIR)
    print("Extracted LBP features shape:", features.shape)
    # Save features
    os.makedirs(FEATURES_DIR, exist_ok=True)
    np.save(os.path.join(FEATURES_DIR, "features.npy"), features)
    np.save(os.path.join(FEATURES_DIR, "labels.npy"), labels)
    print("Features and labels saved!")

if __name__ == "__main__":
    main()

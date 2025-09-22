import os
import numpy as np
import open3d as o3d
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"
FEATURES_TRAIN_DIR = BASE_DIR / "outputs" / "features3d" / "train"
FEATURES_TEST_DIR = BASE_DIR / "outputs" / "features3d" / "test"

# FPFH parameters
VOXEL_SIZE = 0.05
RADIUS_NORMAL = VOXEL_SIZE * 2
RADIUS_FEATURE = VOXEL_SIZE * 5


def load_point_cloud(file_path: Path):
    """
    Load point cloud from txt, xyz, or pts file
    """
    ext = file_path.suffix.lower()
    if ext in [".txt", ".xyz", ".pts"]:
        pts = np.loadtxt(file_path)
        if pts.shape[1] >= 3:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
            return pcd
        else:
            raise ValueError(f"File {file_path} does not contain 3D coordinates")
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def preprocess_point_cloud(pcd, voxel_size=VOXEL_SIZE):
    """
    Downsample and estimate normals
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS_NORMAL, max_nn=30)
    )
    return pcd_down


def compute_fpfh(pcd_down):
    """
    Compute FPFH descriptor
    """
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS_FEATURE, max_nn=100)
    )
    return np.array(fpfh.data).T  # (num_points, 33)


def extract_features_from_dir(root_dir: Path):
    """
    Extract FPFH features from all point clouds inside a dataset folder
    """
    features, labels = [], []
    for class_name in os.listdir(root_dir):
        class_path = root_dir / class_name
        if not class_path.is_dir():
            continue
        for f in os.listdir(class_path):
            file_path = class_path / f
            try:
                pcd = load_point_cloud(file_path)
                pcd_down = preprocess_point_cloud(pcd)
                fpfh = compute_fpfh(pcd_down)
                # Global feature vector by averaging over points
                feature_vec = np.mean(fpfh, axis=0)
                features.append(feature_vec)
                labels.append(class_name)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
    return np.array(features), np.array(labels)


def main():
    # --- Train set ---
    print("Extracting FPFH features for training set...")
    features_train, labels_train = extract_features_from_dir(TRAIN_DIR)
    FEATURES_TRAIN_DIR.mkdir(parents=True, exist_ok=True)  # create folder
    np.save(FEATURES_TRAIN_DIR / "features.npy", features_train)
    np.save(FEATURES_TRAIN_DIR / "labels.npy", labels_train)
    print("Train features shape:", features_train.shape)

    # --- Test set ---
    print("Extracting FPFH features for test set...")
    features_test, labels_test = extract_features_from_dir(TEST_DIR)
    FEATURES_TEST_DIR.mkdir(parents=True, exist_ok=True)  # create folder
    np.save(FEATURES_TEST_DIR / "features.npy", features_test)
    np.save(FEATURES_TEST_DIR / "labels.npy", labels_test)
    print("Test features shape:", features_test.shape)


if __name__ == "__main__":
    main()

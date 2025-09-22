# src/data_preprocessing/visualization.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define dataset paths
BASE_DIR = Path(__file__).parent.parent.parent  # adjust number of parent depending on your folder structure
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"

def load_point_cloud(file_path):
    """
    Load point cloud from .pts, .xyz, or .txt file.
    Returns a Nx3 or Nx6 numpy array (x, y, z, [r, g, b]).
    """
    try:
        data = np.loadtxt(file_path)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def visualize_2d(points, title="2D Projection", save_path=None):
    """
    Plot 2D projections of point cloud (XY, XZ, YZ)
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].scatter(points[:, 0], points[:, 1], s=1)
    axs[0].set_title("XY Projection")
    axs[1].scatter(points[:, 0], points[:, 2], s=1)
    axs[1].set_title("XZ Projection")
    axs[2].scatter(points[:, 1], points[:, 2], s=1)
    axs[2].set_title("YZ Projection")
    
    for ax in axs:
        ax.set_xlabel("X/Y/Z")
        ax.set_ylabel("Y/Z")
    
    fig.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_3d(points, title="3D Point Cloud", save_path=None):
    """
    3D scatter plot of point cloud
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def explore_dataset(data_dir):
    """
    Explore dataset: number of samples per class, average points per sample
    """
    stats = {}
    for class_path in data_dir.iterdir():
        if not class_path.is_dir():
            continue
        files = [f for f in class_path.iterdir() if f.suffix in (".pts", ".xyz", ".txt")]
        num_files = len(files)
        total_points = 0
        for f in files:
            data = load_point_cloud(f)
            if data is not None:
                total_points += data.shape[0]
        avg_points = total_points / num_files if num_files > 0 else 0
        stats[class_path.name] = {"num_samples": num_files, "avg_points": avg_points}
    
    print("Dataset Statistics:")
    for cls, info in stats.items():
        print(f"{cls}: {info['num_samples']} samples, avg {info['avg_points']:.1f} points per sample")
    return stats

def main():
    # Explore training dataset
    stats = explore_dataset(TRAIN_DIR)

    # Visualize some samples
    for class_path in TRAIN_DIR.iterdir():
        if not class_path.is_dir():
            continue
        files = [f for f in class_path.iterdir() if f.suffix in (".pts", ".xyz", ".txt")]
        if len(files) == 0:
            continue
        sample_file = files[0]
        points = load_point_cloud(sample_file)
        if points is not None:
            visualize_2d(points, title=f"{class_path.name} - 2D Projection")
            visualize_3d(points, title=f"{class_path.name} - 3D View")
        break  # visualize only first class for testing

if __name__ == "__main__":
    main()

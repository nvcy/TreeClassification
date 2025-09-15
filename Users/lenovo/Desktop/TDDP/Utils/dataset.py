import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=1024):
        self.root_dir = root_dir
        self.num_points = num_points
        self.classes = sorted(os.listdir(root_dir))
        self.files = []
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            for ext in ["*.pts", "*.txt", "*.xyz"]:
                for f in glob.glob(os.path.join(cls_dir, ext)):
                    self.files.append((f, label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        points = np.loadtxt(file_path, dtype=np.float32)

        if points.shape[1] > 3:
            points = points[:, :3]

        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[choice, :]

        points = points - np.mean(points, axis=0)
        dist = np.max(np.linalg.norm(points, axis=1))
        points = points / dist

        return torch.from_numpy(points).float(), label

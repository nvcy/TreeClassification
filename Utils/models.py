
"""
Custom DGCNN & PointNet implementations (robust for variable input sizes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utility functions
# =========================
def knn(x, k):
    """
    x: (B, C, N) tensor
    returns: indices of k-nearest neighbors (B, N, k)
    """
    num_points = x.size(2)
    k = min(k, num_points)  # safety clamp

    inner = -2 * torch.matmul(x.transpose(2, 1), x)   # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)       # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]      # (B, N, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Constructs edge features for each point and its neighbors.
    x: (B, C, N)
    returns: (B, 2C, N, k)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()  # (B, C, N)
    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # edge feature = neighbor - center, concat center
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


# =========================
# PointNet
# =========================
class PointNet(nn.Module):
    def __init__(self, emb_dims=1024, num_classes=7, dropout=0.5):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.linear1 = nn.Linear(emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # input: (B, N, 3) → (B, 3, N)
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, emb_dims)
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


# =========================
# DGCNN
# =========================
class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024, num_classes=7, dropout=0.5):
        super(DGCNN, self).__init__()
        self.k = k

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, 1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, 1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, 1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, 1, bias=False),
                                   nn.BatchNorm1d(emb_dims),
                                   nn.LeakyReLU(0.2))

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(dropout)

        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # input: (B, N, 3) → (B, 3, N)
        x = x.permute(0, 2, 1)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)

        x = self.conv5(x)  # (B, emb_dims, N)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

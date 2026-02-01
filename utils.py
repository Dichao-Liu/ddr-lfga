import os
import logging
import random
import torch
import torch.optim as optim
import torchvision
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from kmeans_pytorch import kmeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import shutil



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)

 
def tensor_to_image(t):  # (C,H,W) ➜ (H,W,C) for plt
    return min_max_norm(t).cpu().numpy().transpose(1, 2, 0)


def save_combined_images(originals, highs, lows, g_lows, composites,
                         save_dir, epoch_idx):
    """
    originals/highs/lows: (B,3,H,W)
    g_lows/composites: list of 3 tensors each (B,3,H,W)
    Save one big image: row→sample，col→variant
    """
    os.makedirs(save_dir, exist_ok=True)
    B = originals.size(0)
    variants = [originals, highs, lows] + g_lows + composites
    rows = []
    for i in range(B):
        row_imgs = [tensor_to_image(v[i]) for v in variants]
        rows.append(np.concatenate(row_imgs, axis=1))
    grid = np.concatenate(rows, axis=0)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, B * 2))
    plt.imshow(grid)
    plt.axis('off')
    fname = os.path.join(save_dir, f"epoch_{epoch_idx:03d}_vis.png")
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"[Visualization] saved → {fname}")
    
    
    
    
def compute_high_freq(x):
    """
    Sobel magnitude → min‑max → map to [-1,1]; no grad required.
    x: (B,3,H,W) already normalized to [-1,1]
    """
    B, C, H, W = x.shape
    HPF_PAD      = 1       # Sobel padding
    device, dtype = x.device, x.dtype
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=dtype,
                           device=device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=dtype,
                           device=device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    with torch.no_grad():                # no gradient for HPF
        gx = F.conv2d(x, sobel_x, padding=HPF_PAD, groups=C)
        gy = F.conv2d(x, sobel_y, padding=HPF_PAD, groups=C)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)     # magnitude
        # per‑image min‑max
        grad_flat = grad.view(B, -1)
        gmin = grad_flat.min(dim=1)[0].view(B, 1, 1, 1)
        gmax = grad_flat.max(dim=1)[0].view(B, 1, 1, 1)
        grad_norm = (grad - gmin) / (gmax - gmin + 1e-6)
        high = grad_norm * 2 - 1                       # to [-1,1]
    return high.detach()                               # ensure no grad




class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_class, feature_dim=2048):
        super().__init__()
        self.Features = nn.Sequential(*net_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls = nn.Sequential(nn.Linear(feature_dim, num_class))
        self.type_cls = nn.Sequential(nn.Linear(feature_dim, 1))


    def forward(self, x):
        feature = self.pool(self.Features(x)).squeeze()
        return self.cls(feature), feature, self.type_cls(feature)

        




class _netG(nn.Module):
    def __init__(self, input_dim=2048, fc_dim=768):
        super(_netG, self).__init__()
        self.fc_dim = fc_dim
        

        # first linear layer
        self.fc1 = nn.Linear(input_dim, self.fc_dim)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.fc_dim, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        

        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 48, 6, 1, 0, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        
        self.ps7 = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(int(48/2**2)),
            nn.ReLU(True),
        )

        self.ps8 = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.squeeze()
        x = self.fc1(x).view(-1, self.fc_dim, 1, 1)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        x = self.conv6(x)
        x = self.ps7(x)
        x = self.ps8(x)
        
        return x



def mk_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class PCA(object):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def mixup_data(x, y, alpha=1.0):

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix(data, target, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    # Compute output
    target_a = target
    target_b = shuffled_target
    return data, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


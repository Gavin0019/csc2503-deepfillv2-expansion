import torch
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

def gaussian_kernel_1d(kernel_size, sigma, device, dtype):
    # e.g. kernel_size=21, sigma=7
    x = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d

def binary_to_soft_alpha_gaussian(mask, kernel_size=21, sigma=7.0, gamma=1.0):
    """
    mask: [B, 1, H, W] in {0,1}, 1 = hole.
    Returns alpha: [B, 1, H, W] in [0,1].
    """
    B, C, H, W = mask.shape
    device, dtype = mask.device, mask.dtype

    k1d = gaussian_kernel_1d(kernel_size, sigma, device, dtype)
    k2d = torch.outer(k1d, k1d)
    k2d = k2d.view(1, 1, kernel_size, kernel_size)

    # keep same shape by padding
    pad = kernel_size // 2
    alpha = F.conv2d(mask, k2d, padding=pad)

    # normalize to [0,1]
    alpha = alpha / (alpha.max(dim=2, keepdim=True)[0]
                        .max(dim=3, keepdim=True)[0]
                        .clamp(min=1e-6))

    # optional sharpening/softening
    if gamma != 1.0:
        alpha = alpha.pow(gamma)

    return alpha.clamp(0.0, 1.0)

def binary_to_soft_alpha_distance(mask, max_dist=20.0):
    """
    mask: [B, 1, H, W] tensor in {0,1}, 1 = hole.
    max_dist: distance (in pixels) over which we fade from 1 to 0 near boundary.
    """
    mask_np = mask.detach().cpu().numpy()
    B, _, H, W = mask_np.shape
    alphas = []

    for b in range(B):
        h = mask_np[b, 0]  # [H,W]
        # distance inside the hole to nearest *boundary*
        dist_inside = distance_transform_edt(h == 1)
        # normalize: center of big holes ~1, boundary ~0
        alpha = np.clip(dist_inside / max_dist, 0.0, 1.0)
        alphas.append(alpha[None, ...])  # add channel dim

    alpha_np = np.stack(alphas, axis=0)  # [B,1,H,W]
    alpha = torch.from_numpy(alpha_np).to(mask.device, dtype=mask.dtype)
    return alpha

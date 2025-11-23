import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTENSIONS)


class ImageDataset(Dataset):
    def __init__(self, folder_path, 
                       img_shape, # [W, H, C]
                       random_crop=False, 
                       scan_subdirs=False, 
                       transforms=None
                       ):
        super().__init__()
        self.img_shape = img_shape
        self.random_crop = random_crop

        self.mode = 'RGB'
        if img_shape[2] == 1:
            self.mode = 'L' # convert to greyscale

        if scan_subdirs:
            self.data = self.make_dataset_from_subdirs(folder_path)
        else:
            self.data = [entry.path for entry in os.scandir(folder_path) 
                                              if is_image_file(entry.name)]

        self.transforms = T.ToTensor()
        if transforms != None:
            self.transforms = T.Compose(transforms + [self.transforms])

    def make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    samples.append(os.path.join(root, fname))

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = pil_loader(self.data[index], self.mode)

        if self.random_crop:
            w, h = img.size
            if w < self.img_shape[0] or h < self.img_shape[1]:
                img = T.Resize(max(self.img_shape[:2]))(img)
            img = T.RandomCrop(self.img_shape[:2])(img)
        else:
            img = T.Resize(self.img_shape[:2])(img)

        img = self.transforms(img)
        img.mul_(2).sub_(1) # [0, 1] -> [-1, 1]

        return img


class OutpaintingDataset(Dataset):
    """Dataset for outpainting training.

    For outpainting, we need to:
    1. Load a full image (this becomes the ground truth)
    2. Crop a smaller center region (this simulates the "original" image)
    3. Create a mask indicating the cropped-out borders

    The model learns to reconstruct the full image from the center crop.
    """

    def __init__(self, folder_path,
                 img_shape,  # [H, W, C] - target output shape
                 min_crop_ratio=0.5,  # minimum ratio of center crop
                 max_crop_ratio=0.8,  # maximum ratio of center crop
                 random_crop=True,
                 scan_subdirs=False,
                 transforms=None):
        super().__init__()
        self.img_shape = img_shape
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio
        self.random_crop = random_crop

        self.mode = 'RGB'
        if img_shape[2] == 1:
            self.mode = 'L'

        if scan_subdirs:
            self.data = self._make_dataset_from_subdirs(folder_path)
        else:
            self.data = [entry.path for entry in os.scandir(folder_path)
                         if is_image_file(entry.name)]

        self.transforms = T.ToTensor()
        if transforms is not None:
            self.transforms = T.Compose(transforms + [self.transforms])

    def _make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    samples.append(os.path.join(root, fname))
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Returns:
            img_full: Full image (ground truth) [-1, 1], shape [C, H, W]
            img_masked: Image with borders zeroed out [-1, 1], shape [C, H, W]
            mask: Binary mask where 1=outpaint region, shape [1, H, W]
            padding: (pad_top, pad_bottom, pad_left, pad_right)
        """
        img = pil_loader(self.data[index], self.mode)
        H, W, C = self.img_shape

        # Resize/crop to target shape
        if self.random_crop:
            w, h = img.size
            if w < W or h < H:
                img = T.Resize(max(H, W))(img)
            img = T.RandomCrop((H, W))(img)
        else:
            img = T.Resize((H, W))(img)

        # Apply transforms and normalize
        img_full = self.transforms(img)
        img_full.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]

        # Random crop ratio for this sample
        crop_ratio_h = random.uniform(self.min_crop_ratio, self.max_crop_ratio)
        crop_ratio_w = random.uniform(self.min_crop_ratio, self.max_crop_ratio)

        # Calculate padding (how much to outpaint on each side)
        inner_h = int(H * crop_ratio_h)
        inner_w = int(W * crop_ratio_w)

        # Random position for asymmetric padding
        total_pad_h = H - inner_h
        total_pad_w = W - inner_w
        pad_top = random.randint(0, total_pad_h)
        pad_bottom = total_pad_h - pad_top
        pad_left = random.randint(0, total_pad_w)
        pad_right = total_pad_w - pad_left

        # Create mask (1 = outpaint region, 0 = known region)
        mask = torch.ones((1, H, W), dtype=torch.float32)
        mask[:, pad_top:H-pad_bottom, pad_left:W-pad_right] = 0.

        # Create masked image (zero out the borders we want to outpaint)
        img_masked = img_full.clone()
        img_masked = img_masked * (1. - mask)

        return img_full, img_masked, mask, (pad_top, pad_bottom, pad_left, pad_right)


class MixedInpaintOutpaintDataset(Dataset):
    """Dataset that provides both inpainting and outpainting samples.

    Combines regular images with mask generation for mixed training.
    """

    def __init__(self, folder_path,
                 img_shape,  # [H, W, C]
                 outpaint_ratio=0.5,  # probability of outpainting vs inpainting
                 min_crop_ratio=0.5,
                 max_crop_ratio=0.8,
                 random_crop=True,
                 scan_subdirs=False,
                 transforms=None):
        super().__init__()
        self.img_shape = img_shape
        self.outpaint_ratio = outpaint_ratio
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio
        self.random_crop = random_crop

        self.mode = 'RGB'
        if img_shape[2] == 1:
            self.mode = 'L'

        if scan_subdirs:
            self.data = self._make_dataset_from_subdirs(folder_path)
        else:
            self.data = [entry.path for entry in os.scandir(folder_path)
                         if is_image_file(entry.name)]

        self.transforms = T.ToTensor()
        if transforms is not None:
            self.transforms = T.Compose(transforms + [self.transforms])

    def _make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    samples.append(os.path.join(root, fname))
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Returns:
            img_full: Full image (ground truth) [-1, 1]
            mask: Binary mask where 1=region to fill
            is_outpaint: Boolean indicating if this is an outpainting sample
        """
        img = pil_loader(self.data[index], self.mode)
        H, W, C = self.img_shape

        # Resize/crop to target shape
        if self.random_crop:
            w, h = img.size
            if w < W or h < H:
                img = T.Resize(max(H, W))(img)
            img = T.RandomCrop((H, W))(img)
        else:
            img = T.Resize((H, W))(img)

        img_full = self.transforms(img)
        img_full.mul_(2).sub_(1)

        # Decide: outpainting or inpainting
        is_outpaint = random.random() < self.outpaint_ratio

        if is_outpaint:
            # Outpainting mask
            mask = self._create_outpaint_mask(H, W)
        else:
            # Return None for mask - train.py will generate inpainting mask
            mask = None

        return img_full, mask, is_outpaint

    def _create_outpaint_mask(self, H, W):
        """Create outpainting mask for borders."""
        crop_ratio_h = random.uniform(self.min_crop_ratio, self.max_crop_ratio)
        crop_ratio_w = random.uniform(self.min_crop_ratio, self.max_crop_ratio)

        inner_h = int(H * crop_ratio_h)
        inner_w = int(W * crop_ratio_w)

        total_pad_h = H - inner_h
        total_pad_w = W - inner_w
        pad_top = random.randint(0, total_pad_h)
        pad_bottom = total_pad_h - pad_top
        pad_left = random.randint(0, total_pad_w)
        pad_right = total_pad_w - pad_left

        mask = torch.ones((1, H, W), dtype=torch.float32)
        mask[:, pad_top:H-pad_bottom, pad_left:W-pad_right] = 0.

        return mask

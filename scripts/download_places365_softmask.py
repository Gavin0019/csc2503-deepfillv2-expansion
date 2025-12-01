#!/usr/bin/env python
"""
Download Places365 small validation split (val_256) and create a flat subset.

This script will:
  1. Download Places365 val_256 using torchvision.datasets.Places365(small=True).
  2. Locate the val_256 folder under data/places365/places365/val_256.
  3. Copy up to N images into data/places365/val_10k_flat as a flat folder.

Usage (from project root):
    conda activate deepfillv2
    python scripts/download_places365_subset.py \
        --subset-size 10000 \
        --seed 42

You can omit the flags to use the defaults.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

from torchvision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Places365 val_256 and create val_10k_flat subset."
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10000,
        help="Number of images to copy into the flat subset (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling subset (default: 42).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Project root = two levels up from this file: project_root/scripts/this_file.py
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    data_root = project_root / "data" / "places365"
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Data root:    {data_root}")

    # -------------------------------------------------------------------------
    # 1. Download Places365 val_256 (small=256x256)
    # -------------------------------------------------------------------------
    print("[INFO] Downloading Places365 val_256 (small) via torchvision...")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])

    dataset = datasets.Places365(
        root=str(data_root),
        split="val",
        small=True,
        download=True,
        transform=transform,
    )

    print(f"[INFO] Download complete. Total val images (dataset): {len(dataset)}")

    # Torchvision will create something like:
    #   data/places365/places365/val_256/...
    val_dir = data_root / "val_256"
    if not val_dir.is_dir():
        raise RuntimeError(
            f"Could not find val_256 directory at: {val_dir}\n"
            "Inspect data/places365 to see where torchvision placed the images."
        )

    print(f"[INFO] Using val_256 directory: {val_dir}")

    # -------------------------------------------------------------------------
    # 2. Create flat subset val_10k_flat/
    # -------------------------------------------------------------------------
    out_dir = data_root / "val_10k_flat"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output subset directory: {out_dir}")

    # Collect all image paths under val_256
    exts = (".jpg", ".jpeg", ".png")
    all_imgs = []
    for root, _, files in os.walk(val_dir):
        for fname in files:
            if fname.lower().endswith(exts):
                all_imgs.append(Path(root) / fname)

    total_imgs = len(all_imgs)
    if total_imgs == 0:
        raise RuntimeError(f"No images found under {val_dir}")

    print(f"[INFO] Found {total_imgs} images in val_256.")

    subset_size = min(args.subset_size, total_imgs)
    random.seed(args.seed)
    subset = random.sample(all_imgs, subset_size)

    print(f"[INFO] Copying {subset_size} images to {out_dir}...")

    for i, src in enumerate(subset):
        # Keep original extension, but give a simple indexed filename
        dst = out_dir / f"{i:06d}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)

    print("[INFO] Done.")
    print(f"[INFO] Subset created at: {out_dir}")
    print(f"[INFO] Example file count: {sum(1 for _ in out_dir.iterdir())}")


if __name__ == "__main__":
    main()

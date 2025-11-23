#!/usr/bin/env python3
"""
Download Places365 dataset and optionally create a subset for training.

Places365 Standard (small images, 256x256):
- Train: ~1.8M images, ~24GB
- Validation: 36,500 images

Usage:
    # Download full training set
    python scripts/download_places365.py --output_dir data/places365

    # Download and create a subset (e.g., 50,000 images)
    python scripts/download_places365.py --output_dir data/places365 --subset_size 50000

    # Download only validation set (smaller, good for testing)
    python scripts/download_places365.py --output_dir data/places365 --val_only

    # Create subset from existing download
    python scripts/download_places365.py --output_dir data/places365 --subset_only --subset_size 50000
"""

import os
import sys
import argparse
import random
import shutil
import tarfile
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Please install required packages:")
    print("  pip install requests tqdm")
    sys.exit(1)


# Places365 URLs (256x256 standard images)
PLACES365_URLS = {
    'train': 'http://data.csail.mit.edu/places/places365/train_256_places365standard.tar',
    'val': 'http://data.csail.mit.edu/places/places365/val_256.tar',
    'categories': 'https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt',
}

# File sizes for progress indication
EXPECTED_SIZES = {
    'train': 24_000_000_000,  # ~24GB
    'val': 500_000_000,       # ~500MB
}


def download_file(url, output_path, desc=None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    desc = desc or output_path.name

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return output_path


def extract_tar(tar_path, output_dir):
    """Extract tar file with progress."""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, output_dir)
    print(f"Extracted to {output_dir}")


def create_subset(source_dir, output_dir, subset_size, seed=42):
    """Create a random subset of the dataset.

    Args:
        source_dir: Directory containing the full dataset (with category subdirs)
        output_dir: Directory to save the subset
        subset_size: Number of images to include
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Collect all image files
    print("Scanning for images...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    all_images = []

    for ext in image_extensions:
        all_images.extend(source_dir.rglob(f'*{ext}'))
        all_images.extend(source_dir.rglob(f'*{ext.upper()}'))

    all_images = list(set(all_images))  # Remove duplicates
    print(f"Found {len(all_images)} images")

    if len(all_images) < subset_size:
        print(f"Warning: Requested {subset_size} images but only {len(all_images)} available")
        subset_size = len(all_images)

    # Random sample
    print(f"Selecting {subset_size} random images...")
    selected_images = random.sample(all_images, subset_size)

    # Copy to output directory, preserving category structure
    print(f"Copying to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(selected_images, desc="Copying"):
        # Preserve relative path (category/image.jpg)
        rel_path = img_path.relative_to(source_dir)
        dest_path = output_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_path)

    print(f"Created subset with {subset_size} images at {output_dir}")
    return output_dir


def count_images(directory):
    """Count images in directory."""
    directory = Path(directory)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    count = 0
    for ext in image_extensions:
        count += len(list(directory.rglob(f'*{ext}')))
        count += len(list(directory.rglob(f'*{ext.upper()}')))
    return count


def main():
    parser = argparse.ArgumentParser(description='Download Places365 dataset')
    parser.add_argument('--output_dir', type=str, default='data/places365',
                        help='Output directory for the dataset')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Create a subset with this many images (default: use full dataset)')
    parser.add_argument('--val_only', action='store_true',
                        help='Download only validation set (smaller, ~36k images)')
    parser.add_argument('--subset_only', action='store_true',
                        help='Only create subset from existing download (skip download)')
    parser.add_argument('--keep_tar', action='store_true',
                        help='Keep the downloaded tar files after extraction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for subset selection')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.subset_only:
        # Download categories file
        categories_path = output_dir / 'categories_places365.txt'
        if not categories_path.exists():
            print("Downloading categories list...")
            download_file(PLACES365_URLS['categories'], categories_path)

        # Download validation set
        val_tar = output_dir / 'val_256.tar'
        val_dir = output_dir / 'val_256'

        if not val_dir.exists():
            if not val_tar.exists():
                print("\nDownloading validation set (~500MB)...")
                download_file(PLACES365_URLS['val'], val_tar, "val_256.tar")
            extract_tar(val_tar, output_dir)
            if not args.keep_tar:
                val_tar.unlink()
        else:
            print(f"Validation set already exists at {val_dir}")

        # Download training set (unless val_only)
        if not args.val_only:
            train_tar = output_dir / 'train_256_places365standard.tar'
            train_dir = output_dir / 'data_256'  # This is what the tar extracts to

            if not train_dir.exists():
                if not train_tar.exists():
                    print("\nDownloading training set (~24GB)...")
                    print("This may take a while depending on your connection speed.")
                    download_file(PLACES365_URLS['train'], train_tar, "train_256.tar")
                extract_tar(train_tar, output_dir)
                if not args.keep_tar:
                    train_tar.unlink()
            else:
                print(f"Training set already exists at {train_dir}")

    # Create subset if requested
    if args.subset_size:
        # Determine source directory
        if args.val_only:
            source_dir = output_dir / 'val_256'
        else:
            source_dir = output_dir / 'data_256'

        if not source_dir.exists():
            print(f"Error: Source directory {source_dir} does not exist")
            print("Please download the dataset first (remove --subset_only flag)")
            return

        subset_dir = output_dir / f'subset_{args.subset_size}'
        create_subset(source_dir, subset_dir, args.subset_size, args.seed)

        print(f"\n{'='*60}")
        print("To use this subset for training, update your config:")
        print(f"  dataset_path: '{subset_dir.absolute()}'")
        print(f"{'='*60}")

    # Print summary
    print("\n" + "="*60)
    print("Download Summary:")
    print("="*60)

    for subdir in ['val_256', 'data_256', f'subset_{args.subset_size}' if args.subset_size else None]:
        if subdir and (output_dir / subdir).exists():
            count = count_images(output_dir / subdir)
            print(f"  {subdir}: {count:,} images")

    print(f"\nDataset location: {output_dir.absolute()}")

    if not args.subset_size:
        print("\nTo create a subset later, run:")
        print(f"  python scripts/download_places365.py --output_dir {output_dir} --subset_only --subset_size 50000")


if __name__ == '__main__':
    main()

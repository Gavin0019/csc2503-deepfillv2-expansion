"""
Validation script for outpainting model.

Usage:
    python validate.py --checkpoint checkpoints/outpaint/exp0/states.pth --val_dir data/places365/val_256
"""

import argparse
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms as T
from PIL import Image
import glob
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Validate outpainting model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation images directory')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of validation samples to use')
    parser.add_argument('--crop_ratio', type=float, default=0.6,
                        help='Ratio of image to keep (center crop)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for validation')

    args = parser.parse_args()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize metrics (data_range=1.0 for normalized [0,1] images)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Determine model type from checkpoint
    checkpoint_data = torch.load(args.checkpoint, map_location=device)

    if 'stage1.conv1.conv.weight' in checkpoint_data['G'].keys():
        from model.networks import Generator
        print("Using networks.py (PyTorch-native)")
    else:
        from model.networks_tf import Generator
        print("Using networks_tf.py (TF-compatible)")

    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    generator.load_state_dict(checkpoint_data['G'], strict=True)
    generator.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Get validation images
    val_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        val_images.extend(glob.glob(f'{args.val_dir}/**/{ext}', recursive=True))

    if len(val_images) == 0:
        print(f"No images found in {args.val_dir}")
        return

    val_images = val_images[:args.num_samples]
    print(f"Validating on {len(val_images)} images")

    psnr_scores = []
    ssim_scores = []

    for img_path in tqdm(val_images, desc="Validating"):
        try:
            # Load and resize image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((args.img_size, args.img_size))

            # Calculate crop dimensions
            crop_size = int(args.img_size * args.crop_ratio)
            pad_total = args.img_size - crop_size
            pad_top = pad_total // 2
            pad_left = pad_total // 2

            # Ground truth
            ground_truth = T.ToTensor()(img).unsqueeze(0).to(device)

            # Create outpaint mask (border=1, center=0)
            mask = torch.ones(1, 1, args.img_size, args.img_size).to(device)
            mask[:, :, pad_top:pad_top+crop_size, pad_left:pad_left+crop_size] = 0

            # Prepare input
            img_tensor = T.ToTensor()(img).unsqueeze(0).to(device)
            img_masked = img_tensor * (1 - mask)
            img_masked = img_masked * 2 - 1  # Normalize to [-1, 1]

            ones_x = torch.ones_like(img_masked)[:, 0:1, :, :]
            x = torch.cat([img_masked, ones_x, ones_x * mask], dim=1)

            # Generate
            with torch.no_grad():
                _, output = generator(x, mask)

            # Complete image (keep center, use generated for borders)
            output_complete = img_tensor * (1 - mask) + output * mask

            # Compute metrics only on outpainted regions (masked areas)
            # Normalize back to [0, 1] for metrics
            output_norm = (output + 1) / 2
            gt_norm = ground_truth

            psnr_score = psnr(output_norm * mask, gt_norm * mask)
            ssim_score = ssim(output_norm * mask, gt_norm * mask)

            psnr_scores.append(psnr_score.item())
            ssim_scores.append(ssim_score.item())

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Print results
    print("\n" + "="*60)
    print("Validation Results:")
    print("="*60)
    print(f"Number of images: {len(psnr_scores)}")
    print(f"Average PSNR: {sum(psnr_scores)/len(psnr_scores):.2f} dB")
    print(f"Average SSIM: {sum(ssim_scores)/len(ssim_scores):.4f}")
    print(f"PSNR std: {torch.tensor(psnr_scores).std().item():.2f}")
    print(f"SSIM std: {torch.tensor(ssim_scores).std().item():.4f}")
    print("="*60)


if __name__ == '__main__':
    main()

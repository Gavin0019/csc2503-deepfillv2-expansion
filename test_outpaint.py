"""
Test script for outpainting with DeepFillv2.

Expands an image in all directions (or specified directions).

Usage:
    # Expand all sides by 25%
    python test_outpaint.py --image input.png --expand_ratio 0.25

    # Expand with custom padding (in pixels)
    python test_outpaint.py --image input.png --pad_top 64 --pad_bottom 64 --pad_left 64 --pad_right 64

    # Expand to specific output size
    python test_outpaint.py --image input.png --output_size 512 512
"""

import argparse
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Test outpainting')
parser.add_argument("--image", type=str, required=True,
                    help="Path to the input image file")
parser.add_argument("--out", type=str, default=None,
                    help="Path for the output file (default: input_outpainted.png)")
parser.add_argument("--checkpoint", type=str,
                    default="pretrained/states_pt_places2.pth",
                    help="Path to the checkpoint file")

# Expansion options (choose one method)
parser.add_argument("--expand_ratio", type=float, default=None,
                    help="Expand all sides by this ratio (e.g., 0.25 = 25% on each side)")
parser.add_argument("--pad_top", type=int, default=0,
                    help="Pixels to expand on top")
parser.add_argument("--pad_bottom", type=int, default=0,
                    help="Pixels to expand on bottom")
parser.add_argument("--pad_left", type=int, default=0,
                    help="Pixels to expand on left")
parser.add_argument("--pad_right", type=int, default=0,
                    help="Pixels to expand on right")
parser.add_argument("--output_size", type=int, nargs=2, default=None,
                    help="Target output size (H W), image will be centered")

# Other options
parser.add_argument("--save_stages", action="store_true",
                    help="Also save stage1 and stage2 outputs")
parser.add_argument("--save_masked", action="store_true",
                    help="Also save the masked input")


def main():
    args = parser.parse_args()

    # Set output path
    if args.out is None:
        base_name = args.image.rsplit('.', 1)[0]
        args.out = f"{base_name}_outpainted.png"

    # Load checkpoint and determine model type
    generator_state_dict = torch.load(args.checkpoint, map_location='cpu')['G']

    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from model.networks import Generator
    else:
        from model.networks_tf import Generator

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_cuda_if_available else 'cpu')

    # Set up network
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    generator.load_state_dict(generator_state_dict, strict=True)
    generator.eval()

    # Load image
    image = Image.open(args.image).convert('RGB')
    orig_w, orig_h = image.size
    print(f"Original image size: {orig_w} x {orig_h}")

    # Calculate padding
    if args.output_size is not None:
        # Expand to specific output size, centering the original
        out_h, out_w = args.output_size
        pad_top = (out_h - orig_h) // 2
        pad_bottom = out_h - orig_h - pad_top
        pad_left = (out_w - orig_w) // 2
        pad_right = out_w - orig_w - pad_left
    elif args.expand_ratio is not None:
        # Expand by ratio
        pad_top = int(orig_h * args.expand_ratio)
        pad_bottom = int(orig_h * args.expand_ratio)
        pad_left = int(orig_w * args.expand_ratio)
        pad_right = int(orig_w * args.expand_ratio)
    else:
        # Use explicit padding values
        pad_top = args.pad_top
        pad_bottom = args.pad_bottom
        pad_left = args.pad_left
        pad_right = args.pad_right

    # Ensure we have some expansion
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        print("Warning: No expansion specified. Using default 25% expansion on all sides.")
        pad_top = int(orig_h * 0.25)
        pad_bottom = int(orig_h * 0.25)
        pad_left = int(orig_w * 0.25)
        pad_right = int(orig_w * 0.25)

    # Calculate output size
    out_h = orig_h + pad_top + pad_bottom
    out_w = orig_w + pad_left + pad_right

    # Ensure dimensions are divisible by 8 (required by network)
    grid = 8
    out_h = (out_h // grid) * grid
    out_w = (out_w // grid) * grid

    # Recalculate padding to match adjusted output size
    total_pad_h = out_h - orig_h
    total_pad_w = out_w - orig_w
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    print(f"Output size: {out_w} x {out_h}")
    print(f"Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")

    # Convert image to tensor
    image_tensor = T.ToTensor()(image)  # [C, H, W], range [0, 1]

    # Create padded image (pad with zeros = black)
    # F.pad order: (left, right, top, bottom)
    image_padded = F.pad(image_tensor, (pad_left, pad_right, pad_top, pad_bottom),
                         mode='constant', value=0)

    # Create mask (1 = outpaint region, 0 = known region)
    mask = torch.ones((1, out_h, out_w), dtype=torch.float32)
    mask[:, pad_top:pad_top+orig_h, pad_left:pad_left+orig_w] = 0.

    # Prepare for network
    image_padded = image_padded.unsqueeze(0).to(device)  # [1, C, H, W]
    mask = mask.unsqueeze(0).to(device)  # [1, 1, H, W]

    # Normalize to [-1, 1]
    image_padded = image_padded * 2 - 1

    # Prepare input
    image_masked = image_padded * (1. - mask)
    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)

    print("Running outpainting...")
    with torch.inference_mode():
        x_stage1, x_stage2 = generator(x, mask)

    # Complete image: keep original where mask=0, use generated where mask=1
    image_outpainted = image_padded * (1. - mask) + x_stage2 * mask

    # Save output
    def save_tensor_as_image(tensor, path):
        """Convert tensor [-1, 1] to image and save."""
        img = ((tensor[0].permute(1, 2, 0) + 1) * 127.5)
        img = img.clamp(0, 255).to(device='cpu', dtype=torch.uint8)
        Image.fromarray(img.numpy()).save(path)

    save_tensor_as_image(image_outpainted, args.out)
    print(f"Saved outpainted image: {args.out}")

    if args.save_stages:
        base = args.out.rsplit('.', 1)[0]
        save_tensor_as_image(x_stage1, f"{base}_stage1.png")
        save_tensor_as_image(x_stage2, f"{base}_stage2.png")
        print(f"Saved stage outputs: {base}_stage1.png, {base}_stage2.png")

    if args.save_masked:
        base = args.out.rsplit('.', 1)[0]
        # For masked input, show the padded image before generation
        masked_viz = image_padded * (1. - mask)
        save_tensor_as_image(masked_viz, f"{base}_masked.png")
        print(f"Saved masked input: {base}_masked.png")


if __name__ == '__main__':
    main()

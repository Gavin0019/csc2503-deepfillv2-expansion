"""
Automatic inpainting with Canny edge detection.

Automatically detects edges and generates masks for inpainting.

Usage:
    # Basic Canny edge detection
    python test_inpaint_auto.py --image input.jpg --checkpoint pretrained/states_pt_places2.pth

    # With custom parameters
    python test_inpaint_auto.py --image input.jpg --checkpoint states.pth \
        --canny_low 50 --canny_high 150 --dilate_iter 3

    # Canny with flood fill (creates connected regions)
    python test_inpaint_auto.py --image input.jpg --checkpoint states.pth \
        --flood_fill --seed_x 256 --seed_y 256
"""

import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T


def generate_auto_mask(image, **kwargs):
    """
    Generate inpainting mask automatically using Canny edge detection.

    Args:
        image: PIL Image
        **kwargs: Canny-specific parameters

    Returns:
        mask: torch.Tensor (1, 1, H, W), 1=inpaint, 0=keep
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    # Edge detection - finds edges in the image
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,
                     kwargs.get('threshold1', 100),
                     kwargs.get('threshold2', 200))

    # Dilate edges to create fillable regions
    kernel_size = kwargs.get('kernel_size', 5)
    iterations = kwargs.get('iterations', 2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=iterations)

    # Apply flood fill if requested
    if kwargs.get('flood_fill', False):
        # Start flood fill from center point
        seed_point = kwargs.get('seed_point', (w // 2, h // 2))
        mask_copy = mask.copy()

        # Invert mask for flood fill (fill non-edge regions)
        mask_inv = cv2.bitwise_not(mask_copy)

        # Create a slightly larger image for flood fill
        h_ff, w_ff = mask_inv.shape
        mask_floodfill = np.zeros((h_ff + 2, w_ff + 2), np.uint8)

        # Flood fill from seed point
        cv2.floodFill(mask_inv, mask_floodfill, seed_point, 255)

        # Invert to get the filled region as the mask
        mask = cv2.bitwise_not(mask_inv)

    # Convert to torch tensor (1, 1, H, W)
    mask_tensor = torch.from_numpy(mask).float() / 255.0
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

    return mask_tensor


def main():
    parser = argparse.ArgumentParser(description='Automatic inpainting with Canny edge detection')
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--out", type=str, default=None,
                        help="Path for output file (default: input_inpainted_auto.png)")
    parser.add_argument("--checkpoint", type=str,
                        default="pretrained/states_pt_places2.pth",
                        help="Path to checkpoint file")

    # Canny edge detection parameters
    parser.add_argument("--canny_low", type=int, default=100,
                        help="Canny low threshold (default: 100)")
    parser.add_argument("--canny_high", type=int, default=200,
                        help="Canny high threshold (default: 200)")
    parser.add_argument("--dilate_kernel", type=int, default=5,
                        help="Dilation kernel size (default: 5)")
    parser.add_argument("--dilate_iter", type=int, default=2,
                        help="Dilation iterations (default: 2)")
    parser.add_argument("--flood_fill", action="store_true",
                        help="Apply flood fill to create connected regions")
    parser.add_argument("--seed_x", type=int, default=None,
                        help="Flood fill seed point x-coordinate (default: center)")
    parser.add_argument("--seed_y", type=int, default=None,
                        help="Flood fill seed point y-coordinate (default: center)")

    # Visualization options
    parser.add_argument("--save_mask", action="store_true",
                        help="Save generated mask")
    parser.add_argument("--save_stages", action="store_true",
                        help="Save stage1 and stage2 outputs")

    args = parser.parse_args()

    # Set output path
    if args.out is None:
        base_name = args.image.rsplit('.', 1)[0]
        args.out = f"{base_name}_inpainted_auto.png"

    # Load checkpoint and determine model type
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu')

    if 'stage1.conv1.conv.weight' in checkpoint_data['G'].keys():
        from model.networks import Generator
        print("Using networks.py (PyTorch-native)")
    else:
        from model.networks_tf import Generator
        print("Using networks_tf.py (TF-compatible)")

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available()
                         and use_cuda_if_available else 'cpu')
    print(f"Using device: {device}")

    # Set up network
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    generator.load_state_dict(checkpoint_data['G'], strict=True)
    generator.eval()

    # Load image
    image = Image.open(args.image).convert('RGB')
    orig_w, orig_h = image.size
    print(f"Original image size: {orig_w} x {orig_h}")

    # Resize to multiple of 8 (required by network)
    grid = 8
    new_h = (orig_h // grid) * grid
    new_w = (orig_w // grid) * grid
    if new_h != orig_h or new_w != orig_w:
        image = image.resize((new_w, new_h))
        print(f"Resized to: {new_w} x {new_h}")

    # Generate mask automatically using Canny edge detection
    print("Generating mask using Canny edge detection...")

    # Set seed point for flood fill if provided
    seed_point = None
    if args.seed_x is not None and args.seed_y is not None:
        seed_point = (args.seed_x, args.seed_y)
    elif args.seed_x is not None or args.seed_y is not None:
        print("Warning: Both seed_x and seed_y must be provided. Using center point.")

    mask = generate_auto_mask(
        image,
        threshold1=args.canny_low,
        threshold2=args.canny_high,
        kernel_size=args.dilate_kernel,
        iterations=args.dilate_iter,
        flood_fill=args.flood_fill,
        seed_point=seed_point
    ).to(device)

    # Count masked pixels
    mask_ratio = mask.sum() / mask.numel()
    print(f"Mask coverage: {mask_ratio * 100:.2f}% of image")

    # Save mask if requested
    if args.save_mask:
        mask_img = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        mask_path = f"{args.out.rsplit('.', 1)[0]}_mask.png"
        Image.fromarray(mask_img).save(mask_path)
        print(f"Saved mask: {mask_path}")

    # Prepare image tensor
    image_tensor = T.ToTensor()(image).unsqueeze(0).to(device)
    image_tensor = image_tensor * 2 - 1  # Normalize to [-1, 1]

    # Prepare input (masked image + mask channels)
    image_masked = image_tensor * (1. - mask)
    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)

    print("Running inpainting...")
    with torch.inference_mode():
        x_stage1, x_stage2 = generator(x, mask)

    # Complete image: keep original where mask=0, use generated where mask=1
    image_inpainted = image_tensor * (1. - mask) + x_stage2 * mask

    # Save output
    def save_tensor_as_image(tensor, path):
        """Convert tensor [-1, 1] to image and save."""
        img = ((tensor[0].permute(1, 2, 0) + 1) * 127.5)
        img = img.clamp(0, 255).to(device='cpu', dtype=torch.uint8)
        Image.fromarray(img.numpy()).save(path)

    save_tensor_as_image(image_inpainted, args.out)
    print(f"Saved inpainted image: {args.out}")

    if args.save_stages:
        base = args.out.rsplit('.', 1)[0]
        save_tensor_as_image(x_stage1, f"{base}_stage1.png")
        save_tensor_as_image(x_stage2, f"{base}_stage2.png")
        print(f"Saved stage outputs: {base}_stage1.png, {base}_stage2.png")


if __name__ == '__main__':
    main()

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class DictConfig(object):
    """Creates a Config object from a dict 
       such that object attributes correspond to dict keys.    
    """

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __str__(self):
        return '\n'.join(f"{key}: {val}" for key, val in self.__dict__.items())

    def __repr__(self):
        return self.__str__()


def get_config(fname):
    with open(fname, 'r') as stream:
        config_dict = yaml.load(stream, Loader)
    config = DictConfig(config_dict)
    return config


def pt_to_image(img):
    return img.detach_().cpu().mul_(0.5).add_(0.5)


def save_states(fname, gen, dis, g_optimizer, d_optimizer, n_iter, config):
    state_dicts = {'G': gen.state_dict(),
                   'D': dis.state_dict(),
                   'G_optim': g_optimizer.state_dict(),
                   'D_optim': d_optimizer.state_dict(),
                   'n_iter': n_iter}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{fname}")
    print("Saved state dicts!")


def output_to_img(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out


@torch.inference_mode()
def infer_deepfill(generator,
                   image,
                   mask,
                   return_vals=['inpainted', 'stage1']):

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    image = (image*2 - 1.)  # map image values to [-1, 1] range
    # 1.: masked 0.: unmasked
    mask = (mask > 0.).to(dtype=torch.float32)

    image_masked = image * (1.-mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]  # sketch channel
    x = torch.cat([image_masked, ones_x, ones_x*mask],
                  dim=1)  # concatenate channels

    x_stage1, x_stage2 = generator(x, mask)

    image_compl = image * (1.-mask) + x_stage2 * mask

    output = []
    for return_val in return_vals:
        if return_val.lower() == 'stage1':
            output.append(output_to_img(x_stage1))
        elif return_val.lower() == 'stage2':
            output.append(output_to_img(x_stage2))
        elif return_val.lower() == 'inpainted':
            output.append(output_to_img(image_compl))
        else:
            print(f'Invalid return value: {return_val}')

    return output


def random_bbox(config):
    """Generate a random tlhw.

    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width, _ = config.img_shapes
    maxt = img_height - config.vertical_margin - config.height
    maxl = img_width - config.horizontal_margin - config.width
    t = np.random.randint(config.vertical_margin, maxt)
    l = np.random.randint(config.horizontal_margin, maxl)

    return (t, l, config.height, config.width)


def bbox2mask(config, bbox):
    """Generate mask tensor from bbox.

    Args:
        bbox: tuple, (top, left, height, width)

    Returns:
        torch.Tensor: output with shape [1, 1, H, W]

    """
    img_height, img_width, _ = config.img_shapes
    mask = torch.zeros((1, 1, img_height, img_width),
                       dtype=torch.float32)
    h = np.random.randint(config.max_delta_height // 2 + 1)
    w = np.random.randint(config.max_delta_width // 2 + 1)
    mask[:, :, bbox[0]+h: bbox[0]+bbox[2]-h,
         bbox[1]+w: bbox[1]+bbox[3]-w] = 1.
    return mask


def brush_stroke_mask(config):
    """Generate brush stroke mask \\
    (Algorithm 1) from `Generative Image Inpainting with Contextual Attention`(Yu et al., 2019) \\
    Returns:
        torch.Tensor: output with shape [1, 1, H, W]

    """
    min_num_vertex = 4
    max_num_vertex = 12
    min_width = 12
    max_width = 40

    mean_angle = 2*np.pi / 5
    angle_range = 2*np.pi / 15

    H, W, _ = config.img_shapes

    average_radius = np.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(
                    2*np.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)),
                      int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * np.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * np.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (1, 1, H, W))
    return torch.Tensor(mask)


##############################################################################
# Outpainting Mask Generation
##############################################################################

def outpaint_mask_all_sides(config, expand_ratio=None, min_ratio=0.1, max_ratio=0.3):
    """Generate mask for outpainting in all directions.

    The mask indicates where the model should generate new content (mask=1).
    The center region contains the original image (mask=0).

    Args:
        config: config object with img_shapes [H, W, C]
        expand_ratio: fixed expansion ratio, or None for random
        min_ratio: minimum expansion ratio (used if expand_ratio is None)
        max_ratio: maximum expansion ratio (used if expand_ratio is None)

    Returns:
        torch.Tensor: mask [1, 1, H, W] where 1=outpaint region, 0=known region
        tuple: (pad_top, pad_bottom, pad_left, pad_right) padding amounts in pixels
    """
    H, W, _ = config.img_shapes
    mask = torch.ones((1, 1, H, W), dtype=torch.float32)

    if expand_ratio is None:
        # Random expansion ratio for each side
        ratio_top = np.random.uniform(min_ratio, max_ratio)
        ratio_bottom = np.random.uniform(min_ratio, max_ratio)
        ratio_left = np.random.uniform(min_ratio, max_ratio)
        ratio_right = np.random.uniform(min_ratio, max_ratio)
    else:
        ratio_top = ratio_bottom = ratio_left = ratio_right = expand_ratio

    pad_top = int(H * ratio_top)
    pad_bottom = int(H * ratio_bottom)
    pad_left = int(W * ratio_left)
    pad_right = int(W * ratio_right)

    # Ensure we have at least some center region
    inner_h = H - pad_top - pad_bottom
    inner_w = W - pad_left - pad_right

    if inner_h < 32 or inner_w < 32:
        # Fallback to symmetric 25% padding
        pad_top = pad_bottom = H // 4
        pad_left = pad_right = W // 4

    # Center region is known (mask=0)
    mask[:, :, pad_top:H-pad_bottom, pad_left:W-pad_right] = 0.

    return mask, (pad_top, pad_bottom, pad_left, pad_right)


def outpaint_mask_random_sides(config, min_sides=1, max_sides=4,
                                min_ratio=0.15, max_ratio=0.35):
    """Generate mask for outpainting on random subset of sides.

    Useful for training the model to handle various outpainting scenarios.

    Args:
        config: config object with img_shapes [H, W, C]
        min_sides: minimum number of sides to expand
        max_sides: maximum number of sides to expand
        min_ratio: minimum expansion ratio per side
        max_ratio: maximum expansion ratio per side

    Returns:
        torch.Tensor: mask [1, 1, H, W] where 1=outpaint region
        dict: info about which sides were expanded
    """
    H, W, _ = config.img_shapes
    mask = torch.zeros((1, 1, H, W), dtype=torch.float32)

    # Randomly select which sides to expand
    sides = ['top', 'bottom', 'left', 'right']
    num_sides = np.random.randint(min_sides, max_sides + 1)
    selected_sides = np.random.choice(sides, size=num_sides, replace=False)

    expansion_info = {side: 0 for side in sides}

    for side in selected_sides:
        ratio = np.random.uniform(min_ratio, max_ratio)

        if side == 'top':
            pad = int(H * ratio)
            mask[:, :, :pad, :] = 1.
            expansion_info['top'] = pad
        elif side == 'bottom':
            pad = int(H * ratio)
            mask[:, :, H-pad:, :] = 1.
            expansion_info['bottom'] = pad
        elif side == 'left':
            pad = int(W * ratio)
            mask[:, :, :, :pad] = 1.
            expansion_info['left'] = pad
        elif side == 'right':
            pad = int(W * ratio)
            mask[:, :, :, W-pad:] = 1.
            expansion_info['right'] = pad

    return mask, expansion_info


def outpaint_mask_fixed(img_shapes, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0):
    """Generate outpainting mask with fixed padding amounts.

    Useful for inference when you know exactly how much to expand.

    Args:
        img_shapes: [H, W, C] target image shape
        pad_top, pad_bottom, pad_left, pad_right: padding in pixels

    Returns:
        torch.Tensor: mask [1, 1, H, W]
    """
    H, W = img_shapes[0], img_shapes[1]
    mask = torch.ones((1, 1, H, W), dtype=torch.float32)

    # Center region is known (mask=0)
    if H - pad_top - pad_bottom > 0 and W - pad_left - pad_right > 0:
        mask[:, :, pad_top:H-pad_bottom, pad_left:W-pad_right] = 0.

    return mask


def test_contextual_attention(imageA, imageB, contextual_attention):
    """Test contextual attention layer with 3-channel image input
    (instead of n-channel feature).

    """
    rate = 2
    stride = 1
    grid = rate*stride

    b = Image.open(imageA)
    b = b.resize((b.width//2, b.height//2), resample=Image.BICUBIC)
    b = T.ToTensor()(b)

    _, h, w = b.shape
    b = b[:, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print('Size of imageA: {}'.format(b.shape))

    f = T.ToTensor()(Image.open(imageB))
    _, h, w = f.shape
    f = f[:, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print('Size of imageB: {}'.format(f.shape))

    yt, flow = contextual_attention(f*255., b*255.)

    return yt, flow

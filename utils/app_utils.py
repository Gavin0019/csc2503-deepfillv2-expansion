import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from utils.misc import infer_deepfill
from model import load_model

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def _load_models(config_path, device='cuda'):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader)

    config = {name:cfg for name, cfg in config.items() 
                       if os.path.exists(cfg['path'])}
    print(f"Found {len(config)} models in config.")

    loaded_models = {}

    for name, cfg in config.items():
        print(name)
        is_loaded = False
        if cfg['load_at_startup']:
            model = load_model(cfg['path'], device)
            loaded_models[name] = model
            is_loaded = True
        config[name]['is_loaded'] = is_loaded

    return config, loaded_models


class Inpainter:
    def __init__(self, device=None):
        self.available_models = None
        self.loaded_models = None
        self.host = ""
        self.device = torch.device('cuda'
                      if torch.cuda.is_available() else 'cpu') \
                      if device is None else device

    def load_models(self, config_path):
        available_models, loaded_models = _load_models(config_path, self.device)
        print(f"Available models: {list(available_models.keys())}")
        print(f"Loaded models: {list(loaded_models.keys())}")
        self.available_models = available_models
        self.loaded_models = loaded_models

    def get_model_info(self):
        model_data = []
        for name, cfg in self.available_models.items():
            model_dict = cfg.copy()
            model_dict['name'] = name
            model_dict['type'] = 'df'
            model_data.append(model_dict)

        return model_data

    def check_requested_models(self, models):
        for name in models:
            if name not in self.loaded_models:
                path = self.available_models[name]['path']
                model = load_model(path, self.device)
                if model is None:
                    print(f"model @ {path} not found!")
                    continue
                self.loaded_models[name] = model
                # mark only this model as loaded
                self.available_models[name]['is_loaded'] = True
                print(f'Loaded model: {name}')

    def inpaint(self, image, mask, models, max_size=512):

        req_models = models.split(',')
        self.check_requested_models(req_models)

        image_pil = Image.open(image.file).convert('RGB')
        mask_pil = Image.open(mask.file)

        mw, mh = mask_pil.size
        scale = max_size / max(mw, mh)

        mask_pil = mask_pil.resize(
            (max_size, int(scale*mh)) if mw > mh else (int(scale*mw), max_size))
        image_pil = image_pil.resize(mask_pil.size)

        image, mask = ToTensor()(image_pil), ToTensor()(mask_pil)

        response_data = []
        for idx_model, model_name in enumerate(req_models):
            return_vals = self.available_models[model_name]['return_vals']
            model_output_list = []
            outputs = infer_deepfill(
                self.loaded_models[model_name],
                image.to(self.device), 
                mask.to(self.device),
                return_vals=return_vals
            )
            for idx_out, output in enumerate(outputs):
                Image.fromarray(output) \
                    .save(f'app/files/out_{idx_model}_{idx_out}.png')

                model_output_list.append({
                    'name': return_vals[idx_out],
                    'file': f'{self.host}/files/out_{idx_model}_{idx_out}.png'
                })

            model_output_dict = {
                'name': model_name,
                'output': model_output_list
            }
            response_data.append(model_output_dict)

        return response_data
    
    def outpaint(self,
                 image,
                 model_name: str,
                 expand_ratio: float = 0.25,
                 pad_top: int = 0,
                 pad_bottom: int = 0,
                 pad_left: int = 0,
                 pad_right: int = 0):
        """
        Outpaint an image by expanding the canvas and filling the new regions.

        Arguments
        ---------
        image:      FastAPI UploadFile (image file)
        model_name: key from app/models.yaml (e.g. 'pt_places2')
        expand_ratio: if >0, expand each side by this fraction of original size
        pad_*:      explicit padding in pixels (used if expand_ratio == 0)

        Returns
        -------
        response_data: list with one entry, same structure as /api/inpaint:
            [
              {
                'name': model_name,
                'output': [
                  {
                    'name': 'outpainted',
                    'file': '<url-to-image>'
                  }
                ]
              }
            ]
        """

        # Make sure the model is loaded
        self.check_requested_models([model_name])
        generator = self.loaded_models[model_name]

        # Load image
        image_pil = Image.open(image.file).convert('RGB')
        orig_w, orig_h = image_pil.size

        # --- compute padding (port of test_outpaint.py logic) ---
        if expand_ratio is not None and expand_ratio > 0:
            pad_top = int(orig_h * expand_ratio)
            pad_bottom = int(orig_h * expand_ratio)
            pad_left = int(orig_w * expand_ratio)
            pad_right = int(orig_w * expand_ratio)
        # else: use explicit pad_* values as given

        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            # fallback default (25% each side)
            pad_top = int(orig_h * 0.25)
            pad_bottom = int(orig_h * 0.25)
            pad_left = int(orig_w * 0.25)
            pad_right = int(orig_w * 0.25)

        # target size before enforcing /8
        out_h = orig_h + pad_top + pad_bottom
        out_w = orig_w + pad_left + pad_right

        grid = 8
        out_h = (out_h // grid) * grid
        out_w = (out_w // grid) * grid

        # Recalculate padding to match adjusted output size (center original)
        total_pad_h = out_h - orig_h
        total_pad_w = out_w - orig_w
        pad_top = total_pad_h // 2
        pad_bottom = total_pad_h - pad_top
        pad_left = total_pad_w // 2
        pad_right = total_pad_w - pad_left

        # --- build padded image & mask (1 = outpaint region) ---
        image_tensor = ToTensor()(image_pil)  # [C, H, W], in [0,1]

        # F.pad order: (left, right, top, bottom)
        image_padded = F.pad(
            image_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0.0,
        )  # [C, out_h, out_w]

        mask = torch.ones((1, out_h, out_w), dtype=torch.float32)
        mask[:, pad_top:pad_top + orig_h, pad_left:pad_left + orig_w] = 0.0

        # Prepare for network
        image_padded = image_padded.unsqueeze(0).to(self.device)   # [1,3,H,W]
        mask = mask.unsqueeze(0).to(self.device)                   # [1,1,H,W]

        # Normalize to [-1, 1]
        image_padded = image_padded * 2.0 - 1.0

        # Network input
        image_masked = image_padded * (1.0 - mask)
        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)

        # Forward pass
        with torch.inference_mode():
            x_stage1, x_stage2 = generator(x, mask)

        # Composite: keep original where mask=0, use generated where mask=1
        image_outpainted = image_padded * (1.0 - mask) + x_stage2 * mask

        # Convert [-1,1] tensor -> uint8 HWC image
        img = ((image_outpainted[0].permute(1, 2, 0) + 1.0) * 127.5)
        img = img.clamp(0, 255).to(device='cpu', dtype=torch.uint8).numpy()
        out_img = Image.fromarray(img)

        # Save into app/files
        out_fname = f'outpaint_{model_name}.png'
        out_path = os.path.join('app', 'files', out_fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_img.save(out_path)

        # Build response in same shape as /api/inpaint
        model_output_list = [{
            'name': 'outpainted',
            'file': f'{self.host}/files/{out_fname}'
        }]

        model_output_dict = {
            'name': model_name,
            'output': model_output_list,
        }

        return [model_output_dict]

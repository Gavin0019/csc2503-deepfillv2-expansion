# DeepFillv2 PyTorch - Inpainting & Outpainting

A PyTorch implementation of **Free-Form Image Inpainting with Gated Convolution** (DeepFillv2) based on the [original paper](https://arxiv.org/abs/1806.03589). This repository extends the original implementation with outpainting capabilities and automatic mask generation.

## Features

- **Image Inpainting**: Fill in missing or masked regions of images
- **Image Outpainting**: Extend images beyond their original boundaries
- **Automatic Mask Generation**: Canny edge detection with flood fill for automatic region detection
- **Interactive Web Interface**: Draw masks and run inpainting/outpainting in real-time
- **Multiple Pretrained Models**: Places2 and CelebA-HQ weights available
- **Validation Metrics**: PSNR, SSIM, and LPIPS evaluation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Gavin0019/csc2503-deepfillv2-expansion.git
cd deepfillv2-pytorch

# Install dependencies
pip install torch torchvision numpy pillow pyyaml tensorboard
pip install opencv-python  # For automatic mask generation

# Optional: For validation metrics
pip install lpips torchmetrics

# Optional: For web interface
pip install fastapi python-multipart "uvicorn[standard]"
```

### Download Pretrained Models

Download the pretrained weights and place them in the `pretrained/` directory:

- **Places2** (TF-compatible): [Download](https://drive.google.com/u/0/uc?id=1tvdQRmkphJK7FYveNAKSMWC6K09hJoyt&export=download)
- **CelebA-HQ** (TF-compatible): [Download](https://drive.google.com/u/0/uc?id=1fTQVSKWwWcKYnmeemxKWImhVtFQpESmm&export=download)
- **Places2** (PyTorch-native): [Download](https://drive.google.com/u/0/uc?id=1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa&export=download)
- **CelebA-HQ** (PyTorch-native): [Download](https://drive.google.com/u/0/uc?id=17oJ1dJ9O3hkl2pnl8l2PtNVf2WhSDtB7&export=download)

Save as:
- `pretrained/states_tf_places2.pth` (TF-compatible)
- `pretrained/states_pt_places2.pth` (PyTorch-native)

## Usage

### Basic Inpainting

```bash
# Standard inpainting with a mask
python test.py --image input.jpg --mask mask.png --out output.png \
    --checkpoint pretrained/states_pt_places2.pth
```

### Automatic Inpainting

Automatically detect edges and generate masks using Canny edge detection:

```bash
# Basic automatic mask generation
python test_inpaint_auto.py --image input.jpg \
    --checkpoint pretrained/states_pt_places2.pth

# Custom Canny parameters
python test_inpaint_auto.py --image input.jpg \
    --checkpoint pretrained/states_pt_places2.pth \
    --canny_low 50 --canny_high 150 --dilate_iter 3

# With flood fill for connected regions
python test_inpaint_auto.py --image input.jpg \
    --checkpoint pretrained/states_pt_places2.pth \
    --flood_fill --seed_x 256 --seed_y 256

# Save intermediate outputs
python test_inpaint_auto.py --image input.jpg \
    --checkpoint pretrained/states_pt_places2.pth \
    --save_mask --save_stages
```

**Automatic Inpainting Parameters:**
- `--canny_low`: Canny low threshold (default: 100)
- `--canny_high`: Canny high threshold (default: 200)
- `--dilate_kernel`: Dilation kernel size (default: 5)
- `--dilate_iter`: Dilation iterations (default: 2)
- `--flood_fill`: Enable flood fill to create connected regions
- `--seed_x`, `--seed_y`: Seed point for flood fill (default: center)

### Outpainting

Extend images beyond their original boundaries:

```bash
# Expand by 25% on all sides (default)
python test_outpaint.py --image input.jpg \
    --checkpoint pretrained/states_pt_places2.pth \
    --out output.png

# Custom expansion ratio
python test_outpaint.py --image input.jpg \
    --checkpoint pretrained/states_pt_places2.pth \
    --expand_ratio 0.5

# Explicit padding (pixels)
python test_outpaint.py --image input.jpg \
    --checkpoint pretrained/states_pt_places2.pth \
    --pad_top 100 --pad_bottom 100 --pad_left 150 --pad_right 150

# Save intermediate stages
python test_outpaint.py --image input.jpg \
    --checkpoint pretrained/states_pt_places2.pth \
    --save_stages
```

### Validation

Evaluate model performance on a validation dataset:

```bash
pip install lpips torchmetrics

python validate.py \
    --checkpoint pretrained/states_pt_places2.pth \
    --val_dir data/places365/val_256 \
    --num_samples 100 \
    --crop_ratio 0.6
```

Metrics computed:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **LPIPS** (Learned Perceptual Image Patch Similarity)

## Training

### Train Inpainting Model

```bash
# Train with config file
python train.py --config configs/train.yaml

# Monitor training with TensorBoard
tensorboard --logdir tb_logs
```

### Train Outpainting Model

```bash
# Train outpainting model
python train_outpaint.py --config configs/train_outpaint.yaml

# Custom settings
python train_outpaint.py --config configs/train_outpaint.yaml \
    --crop_ratio 0.6 --batch_size 8
```

## Web Interface

Launch the interactive web application with inpainting and outpainting capabilities:

```bash
# Start the server
python app.py

# Open browser to http://127.0.0.1:8000
```

**Features:**
- Draw custom masks for inpainting
- Adjust outpainting parameters (expand ratio, padding)
- Real-time visualization
- Multiple model support

Configure models in `app/models.yaml`:

```yaml
pt_places2:
  path: pretrained/states_pt_places2.pth
  load_at_startup: True
  return_vals: [inpainted, stage1]
```

## Project Structure

```
deepfillv2-pytorch/
├── model/
│   ├── networks.py          # PyTorch-native networks
│   ├── networks_tf.py       # TensorFlow-compatible networks
│   └── losses.py            # Loss functions
├── utils/
│   ├── data.py              # Dataset loaders
│   ├── misc.py              # Utility functions
│   └── app_utils.py         # Web app utilities
├── configs/
│   ├── train.yaml           # Inpainting training config
│   └── train_outpaint.yaml  # Outpainting training config
├── app/
│   └── frontend/            # React web interface
├── test.py                  # Basic inpainting test
├── test_inpaint_auto.py     # Automatic mask generation
├── test_outpaint.py         # Outpainting test
├── validate.py              # Validation metrics
├── train.py                 # Train inpainting
├── train_outpaint.py        # Train outpainting
└── app.py                   # Web application
```

## Model Architectures

This repository provides two network implementations:

1. **`networks_tf.py`**: TensorFlow-compatible implementation
   - Use with converted official weights
   - Maintains compatibility with original TF implementation

2. **`networks.py`**: PyTorch-native implementation
   - Optimized for PyTorch
   - Fine-tuned weights available

Both use the same gated convolution architecture from the DeepFillv2 paper.

## Examples

### Inpainting Results

Input → Masked → Inpainted:

![Inpainting Example](examples/inpaint/case1.png) ![Masked](examples/inpaint/case1_masked.png) ![Result](examples/inpaint/case1_out.png)

### Outpainting Results

Original → Outpainted (25% expansion):

*Add your outpainting examples here*

## Advanced Usage

### Jupyter Notebook

Interactive examples in `test.ipynb`:
- Load and test models
- Visualize results
- Experiment with parameters

### Custom Datasets

For training on custom datasets:

```python
from utils.data import InpaintDataset

dataset = InpaintDataset(
    image_dir='path/to/images',
    mask_dir='path/to/masks',
    img_size=256
)
```

### API Usage

```python
from model import load_model
from PIL import Image
import torch

# Load model
generator = load_model('pretrained/states_pt_places2.pth')

# Prepare inputs
image = Image.open('input.jpg')
mask = Image.open('mask.png')

# Run inference
with torch.inference_mode():
    output = generator(image, mask)
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- numpy
- Pillow
- opencv-python (for automatic mask generation)
- pyyaml
- tensorboard (for training)
- lpips, torchmetrics (for validation)
- fastapi, uvicorn (for web interface)

## Citation

If you use this code, please cite the original DeepFillv2 paper:

```bibtex
@article{yu2018free,
  title={Free-form image inpainting with gated convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}
```

## Acknowledgments

- Original TensorFlow implementation: [JiahuiYu/generative_inpainting](https://github.com/JiahuiYu/generative_inpainting)
- Base PyTorch port: [nipponjo/deepfillv2-pytorch](https://github.com/nipponjo/deepfillv2-pytorch)

## License

This project is for research and educational purposes.

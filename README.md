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

# (Recommended) Create and activate a virtual environment / conda env
# python -m venv .venv
# source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

```

### Download Pretrained Models

Download the pretrained weights and place them in the `pretrained/` directory:

- **Places2** (TF-compatible): [Download](...)
- **CelebA-HQ** (TF-compatible): [Download](...)
- **Places2** (PyTorch-native): [Download](...)
- **CelebA-HQ** (PyTorch-native): [Download](...)

Save as:
- `pretrained/states_tf_places2.pth` (TF-compatible)
- `pretrained/states_pt_places2.pth` (PyTorch-native)

Optionally, you can also use our fine-tuned outpainting checkpoint:

We provide our fine-tuned outpainting checkpoint (trained for 20–50k iterations on a 50k-image Places365 subset):

#### 1. Outpainting Model (ours)
Fine-tuned for 20–50k steps on a 50k-image Places365 subset using border expansion masks.

- **Download (Google Drive):** [Outpainting Checkpoint](https://drive.google.com/file/d/1sk4nfm2WC0JUzJA4kMk1q-a6RiMclQDn/view?usp=sharing)
Save as: pretrained/states_outpaint_places365_50k.pth

#### 2. Soft-mask Inpainting Model (ours)
Trained using blurred α-masks and soft compositing to produce smoother interior inpainting boundaries.

- **Download (Google Drive):** [Soft-mask Inpainting Checkpoint](https://drive.google.com/file/d/1iB0oZvlGhRIDv-_RGlWt0idABAHw0fQB/view?usp=sharing)

Save as: pretrained/states_softmask_inpaint.pth


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

## Dataset Setup

This project uses subsets of the **Places365-Standard** dataset for both outpainting and soft-mask inpainting.  
To simplify setup, we provide automated download scripts that download the required images and arrange them into the correct directory structure.

All dataset preparation scripts are located in the `scripts/` directory.

---

### Downloading Data for Outpainting

Use the following script to download and prepare the Places365 subset for **outpainting**:

```bash
python scripts/download_places365_outpaint.py
```

### Downloading Data for Soft-mask Inpainting

For the soft-mask inpainting experiments, run:

```bash
python scripts/download_places365_softmask.py
```
This script sets up the data in a separate folder with the structure expected by the soft-mask training code.

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

### Training the Soft-mask Inpainting Model

To train the soft-mask model, simply run:
```bash
python train_softmask.py --config configs/train_soft_alpha.yaml
```

## Web Interface

We provide an interactive web interface for manual inpainting and outpainting, built with **FastAPI** (backend) and **React** (frontend). The interface allows you to upload an image, draw masks, adjust outpainting parameters, and run inference using any of the pretrained models.

### Screenshot

<p align="center">
  <img src="images/ui.png" width="700"/>
</p>

---
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

---

### User Guide

#### 1. Load Image
- Click **Choose File** to upload an image.

#### 2. Draw Mask (Inpainting)
- Use the brush tools on the right.
- Adjust brush size using the circle icons.
- Draw over regions you want the model to fill in.

#### 3. Outpainting Controls
- **Expand ratio** controls how much to extend the borders.
- **Pad top/bottom/left/right** specify exact pixel padding for asymmetric expansion.

#### 4. Run Model
- **Inpaint** → fills interior masked regions  
- **Outpaint** → expands the image canvas outward using the selected model (make sure to have outpainting weights downloaded)

Results appear below the canvas.

#### 5. Editing Tools
- **Undo / Redo** — revert mask strokes  
- **Clear** — remove all mask strokes  
- **Save mask / Save image** — export mask or result image  

---

## Examples

### Inpainting Results

Input → Masked → Inpainted:

| Input | Masked | Inpainted |
|-------|--------|-----------|
| <img src="examples/inpaint/case1.png" width="250"/> | <img src="examples/inpaint/case1_masked.png" width="250"/> | <img src="examples/inpaint/case1_out.png" width="250"/> |


### Outpainting Results

Input → Masked → Outpainted:

| Input | Masked | Outpainted |
|-------|--------|-----------|
| <img src="images/input.jpg" width="250"/> | <img src="images/masked.jpg" width="250"/> | <img src="images/outpainted.png" width="250"/> |

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

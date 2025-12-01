import os
from pathlib import Path

import torchvision
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
data_root = ROOT / "data" / "places365"

data_root.mkdir(parents=True, exist_ok=True)

# Simple transform that just returns the raw 256x256 image for now
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # just in case
])

print("Downloading Places365 val split (small=256x256) to:", data_root)
dataset = torchvision.datasets.Places365(
    root=str(data_root),
    split="val",
    small=True,
    download=True,
    transform=transform,
)

print("Total val images:", len(dataset))
print("Done.")

<div align="center">

# 🔍 Image Upscaling with Real-ESRGAN

### AI-Powered 4× Image Enhancement using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Real-ESRGAN](https://img.shields.io/badge/Model-Real--ESRGAN%20x4plus-blueviolet?style=for-the-badge)](https://github.com/xinntao/Real-ESRGAN)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

> Upscale any blurry or low-resolution image to **4× its original size** using the state-of-the-art **Real-ESRGAN** super-resolution model — GPU-accelerated for blazing-fast inference (~0.68s per image).

</div>

---

## ⚡ What It Does

**Real-ESRGAN** (Real Enhanced Super-Resolution Generative Adversarial Network) is a state-of-the-art AI model that restores and upscales real-world degraded images. This project wraps it into a clean Jupyter notebook pipeline that:

| Step | Action |
|------|--------|
| 📂 **Input** | Reads images from the `inputs/` folder |
| 🚀 **Enhance** | Runs Real-ESRGAN x4plus on each image at 4× scale |
| ⚡ **GPU** | Uses CUDA (FP16) for fast inference |
| 💾 **Output** | Saves upscaled images to `outputs/` folder |
| ⏱️ **Speed** | ~0.68 seconds per image on RTX 4050 |

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| **Model** | Real-ESRGAN x4plus (`RealESRGAN_x4plus.pth`) |
| **Architecture** | RRDBNet (Residual-in-Residual Dense Block) |
| **Scale Factor** | 4× upscaling |
| **Precision** | FP16 (half precision) for GPU speed |
| **Input Channels** | 3 (RGB) |
| **Num Blocks** | 23 |
| **Model Size** | ~64 MB |
| **Device** | CUDA (GPU) / CPU fallback |

---

## 📁 Repository Structure

```
📦 Image-Upscaling-py/
├── 📓 unblur.ipynb                  # Main upscaling pipeline notebook
├── 🤖 RealESRGAN_x4plus.pth         # Pre-trained Real-ESRGAN model weights
├── 📂 inputs/                       # Place your input images here
├── 📂 outputs/                      # Upscaled results saved here
├── 📋 requirment.txt                # Python dependencies
└── 📄 README.md                     # Project documentation
```

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/chyk2468/Image-Upscaling-py.git
cd Image-Upscaling-py
```

### 2️⃣ Install Dependencies

```bash
pip install torch torchvision basicsr realesrgan pillow numpy
```

> 💡 For GPU support, install the CUDA-compatible version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

### 3️⃣ Add Your Images

Drop any `.png`, `.jpg`, `.jpeg`, or `.bmp` images into the `inputs/` folder.

### 4️⃣ Run the Notebook

```bash
jupyter notebook unblur.ipynb
```

Run all cells — your upscaled images will appear in `outputs/`.

---

## ⚙️ Pipeline Walkthrough

### Step 1 — Imports & Setup
```python
import torch
from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os, time
```

### Step 2 — GPU Detection
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available())
# Output: CUDA available: True
# Device: NVIDIA GeForce RTX 4050 Laptop GPU
```

### Step 3 — Load Model Architecture
```python
model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64,
    num_block=23, num_grow_ch=32, scale=4
)
```

### Step 4 — Initialize Upsampler
```python
upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',
    model=model,
    tile=0,        # set >0 to handle large images with limited VRAM
    tile_pad=10,
    pre_pad=0,
    half=True,     # FP16 for faster GPU inference
    device=device
)
```

### Step 5 — Batch Process All Images
```python
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img = Image.open(os.path.join(input_folder, filename)).convert('RGB')
        img_np = np.array(img)

        output, _ = upsampler.enhance(img_np, outscale=4)

        Image.fromarray(output).save(os.path.join(output_folder, filename))
```

**Example Output:**
```
Processing: photo.png
✅ Saved: outputs\photo.png
⏱ Time taken: 0.68 seconds

🎉 All images processed successfully!
```

---

## 🔄 Upscaling Pipeline

```
  Input Images (inputs/)
        │
        ▼
  ┌──────────────────────┐
  │   Read image with    │
  │   PIL → NumPy array  │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │   Real-ESRGAN x4+    │
  │   RRDBNet (23 blocks)│
  │   FP16 · CUDA GPU    │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  4× Resolution       │
  │  Enhanced Output     │
  │  Restored Textures   │
  └──────────┬───────────┘
             │
             ▼
  Output Images (outputs/)
```

---

## 💡 Tips & Configuration

| Parameter | Default | When to Change |
|-----------|---------|----------------|
| `tile=0` | No tiling | Set `tile=512` if you run out of VRAM on large images |
| `half=True` | FP16 | Set `False` if you get artifacts on older GPUs |
| `outscale=4` | 4× | Change to `2` or `8` for different scale factors |
| Input formats | PNG/JPG/JPEG/BMP | Add more extensions as needed |

---

## 🛠️ Technologies Used

<div align="center">

| Category | Tool |
|----------|------|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch (CUDA) |
| **Model** | Real-ESRGAN (basicsr / realesrgan) |
| **Image Processing** | Pillow, NumPy |
| **Notebook** | Jupyter |
| **Hardware** | NVIDIA GPU (RTX 4050 tested) |

</div>

---

## 📚 References

- **Real-ESRGAN** — [github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **BasicSR** — [github.com/XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR)
- **Wang et al. (2021)** — *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*

---

## 👤 Author

**Yashwant Kumar Chitchula**  
B.Tech CSE (AI & ML) — VIT Chennai

[![GitHub](https://img.shields.io/badge/GitHub-chyk2468-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/chyk2468)

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

</div>

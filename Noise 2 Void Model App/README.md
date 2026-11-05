# RamanN2V

A general-purpose PyQt5 GUI application for **Noise2Void (N2V) denoising**
of hyperspectral imaging data (Raman, fluorescence, reflectance, etc.).

Supports both **band-wise 2D N2V** and **3D (spatial+spectral) N2V** modes.

---

## Features

### 1. Data Tab
- Select a folder of `.mat` files
- Auto-detect spatial/spectral shape
- Auto-pad small images (e.g. `100×100`) to multiple of patch size
- Generate overlapping patches (`.npy`) with configurable patch size and stride

> **Preprocessing is left to the user.** Apply your own AsLS, SNV, baseline
> correction, normalization, etc. before loading `.mat` files.
> This keeps the tool generic across different hyperspectral domains.

### 2. Train Tab
- Choose **2D (band-wise)** or **3D (spatial+spectral)** N2V mode
- Configurable epochs, batch size, learning rate, mask ratio
- Spectral chunk size for 3D mode
- Real-time loss plot
- Auto-save best model + `_config.json`

### 3. Predict Tab
- Load trained `.pth` model (mode auto-detected from checkpoint)
- Run inference on a folder of `.mat` files
- Sliding-window patch processing with Gaussian blending
- Side-by-side preview (Original vs Denoised)
- Per-pixel spectrum comparison plot
- Output saved as `_denoised.mat`

---

## Installation

```bash
pip install -r requirements.txt
```

GPU recommended for training; CPU works for inference but slower.

---

## Run

```bash
python main.py
```

---

## File Structure

```
raman_n2v/
├── main.py              # Entry point + global stylesheet
├── model.py             # UNet2D / UNet3D
├── n2v_core.py          # Blind-spot mask + N2V loss
├── dataset.py           # PyTorch Dataset (2D/3D)
├── utils.py             # mat I/O, padding, patch extract/reconstruct
├── tabs/
│   ├── data_tab.py
│   ├── train_tab.py
│   └── predict_tab.py
├── requirements.txt
└── README.md
```

---

## Pipeline

```
.mat input (e.g. 200×200×1340, preprocessed by user)
       ↓
Pad to multiple of patch size + sliding-window patches
       ↓
N2V training (self-supervised, no GT needed)
       ↓
Predict on new .mat → denoised .mat output
```

---

## Notes

- **2D mode**: each spectral band processed independently. Fast, low memory,
  ignores spectral correlation.
- **3D mode**: spatial+spectral chunks (default 64 bands). Captures spectral
  correlation, more memory-intensive.
- **Mask ratio**: fraction of pixels masked per training step (default 0.5%).
- Patch size must be a multiple of 8 (UNet has 3 downsampling levels = 2³).

---

## License

MIT

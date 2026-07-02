# Spectral Vision Toolkit

A collection of self-contained GUI tools covering a full **spectroscopic imaging research pipeline** —
from hyperspectral / multispectral / Raman data viewing and denoising, through deep-learning
segmentation, detection, and pixel-wise quantification, to photon-transport simulation.

Each tool is an independent PyQt5 application with its own README, requirements, and entry point
(`python main.py`). Built to systematize an end-to-end research workflow into reusable tools.

> **Data policy:** This repository contains **source code only**. No experimental, clinical, or
> unpublished research data is included. Where needed, small synthetic/placeholder samples are used.
> Expected input formats are documented per tool (see each tool's README).

---

## Pipeline overview

```
   Acquire / View        Preprocess  Analyze             Simulate / Reference
 ┌────────────────┐    ┌───────────────────────────┐   ┌──────────────────────┐
 │ Hyperspectral  │ →  │ SegToolkit  (segmentation)│   │ SORS Monte Carlo     │
 │ Viewer         │    │ YOLO Toolkit (detection)  │   │ (photon transport)   │
 │                │    │ QuantToolkit (regression) │   │ Spectrum Library     │
 └────────────────┘    └───────────────────────────┘   └──────────────────────┘
```
## Hyperspectral Viewer  
<img width="997" height="602" alt="Image" src="https://github.com/user-attachments/assets/c49933ba-91a1-476f-a725-0516dfda5f33" />


## SegToolkit 
<img width="1224" height="712" alt="Image" src="https://github.com/user-attachments/assets/4026042c-9d8a-4efb-abb7-2fcd188bb9b0" />

## YOLO Toolkit 
<img width="1224" height="712" alt="Image" src="https://github.com/user-attachments/assets/2cb2e0dd-71b6-464a-a67e-ed98d2b6e17a" />

## QuantToolkit
<img width="1224" height="712" alt="Image" src="https://github.com/user-attachments/assets/19f9e9ee-91c2-4ba6-9963-53d38f97a396" />

## SORS Monte Carlo  
<img width="1152" height="777" alt="Image" src="https://github.com/user-attachments/assets/dcb9add9-d650-41ab-9271-671d5483be05" />

## Spectrum Library   
<img width="805" height="536" alt="Image" src="https://github.com/user-attachments/assets/4085e5e1-3024-4198-beca-836d4cfb11a9" />

---

## Tools

| Tool | Purpose | Key methods |
|------|---------|-------------|
| **[Hyperspectral Viewer](./Hyperspectral Viewer)** | Explore and inspect hyperspectral / multispectral cubes (band navigation, per-pixel spectra) | Multi-format cube I/O |
| **[SegToolkit](./SegToolkit)** | Semantic segmentation training / eval / inference for multi-channel, grayscale, RGB images | UNet, UNet++, SegFormer, DeepLabV3+, Mask2Former |
| **[YOLO Toolkit](./yolo-toolkit)** | Object detection training + label editor | Ultralytics YOLO |
| **[QuantToolkit](./QuantToolkit)** | Pixel-wise quantitative regression (e.g. concentration) from multispectral images | Patch-CNN & dense U-Net regression, K-Fold CV |
| **[Spectrum Library Maker](./spectrum-library-maker)** | Build and manage reference spectral libraries | Spectrum I/O, labeling |
| **[SORS Monte Carlo Simulator](./sors-monte-carlo)** | Predict photon transport / optical behavior for SORS in layered turbid media | GPU Monte Carlo (PMCX/MCX), NumPy fallback |

> Adjust the list above to match the tools actually present. Each row links to that tool's folder.

---

## Input data formats

No data files are shipped. Tools expect the following formats (documented in detail per tool):

| Type | Formats | Example shape |
|------|---------|---------------|
| Hyperspectral / multispectral cube | `.mat`, `.npy`, `.hdr`, `.tif` | `H × W × Bands` (e.g. `200 × 200 × 1340`) |
| RGB / grayscale image | `.png`, `.jpg`, `.tif` | `H × W [× 3]` |
| Segmentation label | `.png` (single-channel mask) | `H × W`, `0 = background` |
| Regression label | `.png` / `.npy` (per-pixel value) | `H × W` |
| Spectrum | `.csv` | `wavelength, intensity` columns |

---

## Tech stack

- **Language:** Python
- **GUI:** PyQt5
- **DL:** PyTorch, segmentation-models-pytorch, HuggingFace, Ultralytics YOLO
- **Scientific:** NumPy, SciPy, scikit-learn, PMCX/MCX (Monte Carlo)

---

## About

These tools were developed to support spectroscopic imaging research spanning hyperspectral /
SWIR imaging, Raman spectroscopy, and materials/tissue analysis. The emphasis is on covering the
**whole pipeline — acquisition, preprocessing, analysis, and physical simulation** — with minimal,
direct implementations rather than heavy frameworks.

*Comments and variable names in some tools are in Korean (original research code).*

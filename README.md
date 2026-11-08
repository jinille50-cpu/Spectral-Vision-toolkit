# Spectral Vision Toolkit

GUI tools for a **vision inspection and spectral imaging research workflow** —
from labeling and model training to defect detection, plus multi-band image analysis and
physical simulation.

The core is **Lumen Vision**, a single desktop app that connects the whole vision workflow:
**label → train → evaluate → inspect**. Everything runs locally on one PC, so image data
never leaves the lab.

> **Data policy:** This repository contains **source code only**. No experimental, clinical, or
> unpublished research data is included. Where needed, small synthetic/placeholder samples are used.
> Expected input formats are documented per tool.

---

## Lumen Vision

<img width="997" alt="Lumen Vision home" src="Lumen%20Vision/docs/home.png" />

One PySide6 app with five tools, opened from a single home screen.

| # | Tool | What it does | Status |
|---|------|--------------|--------|
| 01 | **Labeler** | Box and segmentation labeling, export to YOLO / mask format | In progress |
| 02 | **Segmentation** | Train, evaluate, and run semantic segmentation (RGB, grayscale, multi-band) | Done |
| 03 | **YOLO** | Object detection training and inference | In progress |
| 04 | **PatchCore** | Anomaly detection from normal samples only, with heatmaps | In progress |
| 05 | **HSI Viewer** | Multi-band cube viewer with SAM 3 segmentation and per-object spectra | Done |

### Workflow

```
  Label                Train / Evaluate             Inspect
┌──────────────┐    ┌───────────────────────┐    ┌──────────────────────┐
│ Labeler      │ →  │ Segmentation          │ →  │ Inference            │
│ (box / mask) │    │ YOLO                  │    │ PatchCore (anomaly)  │
└──────────────┘    └───────────────────────┘    └──────────────────────┘
        ↑                                                  │
        └──────── model-assisted pre-labeling ─────────────┘
```

### Highlights

- **One workflow, one app** — labels, checkpoints, and results flow between tools without manual path setup
  (e.g. finished training opens the evaluation tab with the new checkpoint already loaded).
- **Multi-band ready** — segmentation models and the viewer accept more than 3 channels;
  color augmentations are switched off automatically for non-RGB inputs.
- **Model choice** — UNet / UNet++, DeepLabV3+, SegFormer (B0–B5), Mask2Former, DINOv2/v3 backbones,
  YOLOv8–YOLO26, PatchCore. Models available in the code appear on the home screen automatically.
- **Training options** — folder split, ratio split, or K-fold (OOF) validation; Albumentations presets;
  automatic class detection with suggested class weights.
- **Researcher conveniences** — recent folders/files and last-used settings are remembered across sessions.
- **Offline by design** — SAM 3, YOLO, and segmentation models all run on the local GPU.

### Roadmap

- Labeler: batch pre-labeling with SAM 3 text prompts, review screen (accept / fix / delete, low-confidence first)
- Labeler: pre-labeling with your own trained YOLO / segmentation model
- Labeler: multi-band cube labeling with band-selectable false color
- ONNX export for deployment

---

## Other tools

Standalone apps, each with its own README and entry point.

| Tool | Purpose | Key methods |
|------|---------|-------------|
| **[QuantToolkit](./QuantToolkit)** | Pixel-wise quantitative regression (e.g. concentration) from multi-band images | Patch-CNN & dense U-Net regression, K-fold CV |
| **[Spectrum Library Maker](./spectrum-library-maker)** | Build and manage reference spectral libraries | Spectrum I/O, labeling |
| **[SORS Monte Carlo Simulator](./sors-monte-carlo)** | Photon transport in layered turbid media for spatially offset Raman | GPU Monte Carlo (PMCX/MCX), NumPy fallback |
| **[Raman Data Processing](./raman-data-processing)** | Batch baseline correction, smoothing, and x-axis calibration | AsLS, Savitzky–Golay |
| **[Raman Denoising](./raman-denoising)** | Self-supervised denoising of spectral images | Noise2Void, Noise2Fast, Neighbor2Neighbor |

> Adjust the list and links to match the folders actually in the repository.

---

## Input data formats

No data files are shipped. Tools expect the following formats:

| Type | Formats | Example shape |
|------|---------|---------------|
| Multi-band cube | `.mat`, `.npy`, `.hdr`, `.tif` | `H × W × Bands` |
| RGB / grayscale image | `.png`, `.jpg`, `.tif` | `H × W [× 3]` |
| Segmentation label | `.png` (single-channel mask) | `H × W`, `0 = background` |
| Detection label | `.txt` (YOLO format) | one box per line |
| Regression label | `.png` / `.npy` (per-pixel value) | `H × W` |
| Spectrum | `.csv` | `wavelength, intensity` columns |

---

## Tech stack

- **Language:** Python 3.10
- **GUI:** PySide6 (Lumen Vision), PyQt5 (some standalone tools)
- **Deep learning:** PyTorch, segmentation-models-pytorch, HuggingFace, Ultralytics YOLO, SAM 3, Anomalib
- **Scientific:** NumPy, SciPy, scikit-learn, Albumentations, PMCX/MCX

---

## About

Developed to support imaging research across vision inspection, multi-band (SWIR) imaging,
Raman spectroscopy, and materials/tissue analysis. The goal is to cover the **whole pipeline —
labeling, training, inspection, and physical simulation** — with direct, readable implementations
rather than heavy frameworks.

*Comments and some UI text are in Korean (original research code).*

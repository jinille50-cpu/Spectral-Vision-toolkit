# Hyperspectral Viewer

An interactive viewer for hyperspectral / multispectral image cubes. Explore false-color
composites, scroll through individual bands, and extract per-pixel spectra by clicking points
directly on the image.

Built with **matplotlib widgets + tkinter** (single-file, no heavy GUI framework).

![Viewer](docs/screenshot.png)

---

## Features

- **Cube formats**: `.mat` (incl. MATLAB v7.3 via h5py), `.tif` / `.tiff`, `.npy` — 2D or 3D (`H × W × Bands`)
- **False-color composite**: assign any band to R / G / B channels
- **Contrast control**: low / high percentile sliders + optional color inversion (`1 - x`)
- **Band browser**: slider to view any single band (grayscale)
- **Multi-point spectrum picker**: click to add points, drag to move, right-click to remove
- **Spatial binning**: N×N averaging around each point for smoother spectra
- **SNV normalization**: toggle standard normal variate on extracted spectra
- **Zoom & pan**: scroll to zoom on the image
- **Export**: spectra to `.txt` / `.csv` (per-point columns) · false-color image to `.png` / `.jpg` / `.tif`

---

## Installation

```bash
pip install numpy scipy matplotlib h5py pillow tifffile
```

(`tkinter` ships with standard Python.)

---

## Run

```bash
python Hyperspectral_Viewer.py
```

---

## Controls

| Action | Effect |
|--------|--------|
| **Left click** (image) | Add a spectrum point |
| **Drag** point | Move point (spectrum updates live) |
| **Right click** | Remove nearest point |
| **Scroll** | Zoom in / out |
| **Band slider** | View a single band in grayscale |
| **False Color R,G,B** | Set bands for the composite |
| **Contrast Low / High** | Adjust percentile stretch |
| **Bin Size** | N×N spatial averaging per point |
| **SNV / Autoscale / Invert** | Toggle spectrum normalization, Y-autoscale, color inversion |
| **Export Spectra** | Save picked spectra (`.txt` / `.csv`) |
| **Save False Color** | Save current composite image |

---

## Input format

No data is included in this repository. The viewer expects a hyperspectral cube:

| Format | Notes |
|--------|-------|
| `.mat` | first array found; MATLAB v7.3 supported via h5py |
| `.tif` / `.tiff` | multi-page = bands |
| `.npy` | `H × W × Bands` (2D arrays are treated as single-band) |

Preprocessing (baseline correction, normalization, etc.) is left to the user.

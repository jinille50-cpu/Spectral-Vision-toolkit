"""Hyperspectral cube loading (MAT v6/v7/v7.3, HDF5, TIFF, NPY, ENVI) -> (H, W, B) float32."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

SUPPORTED_EXTS = (".mat", ".h5", ".hdf5", ".tif", ".tiff", ".npy", ".hdr", ".raw", ".img", ".dat")
FILE_FILTER = (
    "Hyperspectral files (*.mat *.h5 *.hdf5 *.tif *.tiff *.npy *.hdr);;"
    "MAT files (*.mat);;HDF5 files (*.h5 *.hdf5);;TIFF files (*.tif *.tiff);;"
    "NumPy files (*.npy);;ENVI header (*.hdr);;All files (*.*)"
)


@dataclass
class Cube:
    data: np.ndarray                      # (H, W, B) float32, C-contiguous
    path: str = ""
    wavelengths: np.ndarray | None = None  # (B,) in nm if known
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_bands(self) -> int:
        return self.data.shape[2]

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def stem(self) -> str:
        return os.path.splitext(self.name)[0]

    def band_axis(self) -> np.ndarray:
        """X axis for spectra: wavelengths if available else band indices."""
        if self.wavelengths is not None and len(self.wavelengths) == self.n_bands:
            return self.wavelengths
        return np.arange(self.n_bands, dtype=np.float32)

    @property
    def has_wavelengths(self) -> bool:
        return self.wavelengths is not None and len(self.wavelengths) == self.n_bands


# --------------------------------------------------------------------------- helpers
def _to_hwb(arr: np.ndarray) -> np.ndarray:
    """Heuristically reorder a 2D/3D array to (H, W, B)."""
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported array shape {arr.shape}")
    s = arr.shape
    # Smallest axis is usually bands; tifffile returns (B, H, W) for multi-page
    if s[0] <= s[2] and s[0] < s[1]:
        return np.moveaxis(arr, 0, -1)
    return arr


def _find_hdf5_dataset(f):
    import h5py

    if "image" in f and isinstance(f["image"], h5py.Dataset):
        return "image"
    keys = [k for k in f.keys() if not k.startswith("#")]
    best, best_size = None, -1
    for k in keys:
        item = f[k]
        if isinstance(item, h5py.Dataset) and item.ndim == 3:
            size = int(np.prod(item.shape))
            if size > best_size:
                best, best_size = k, size
    if best is not None:
        return best
    for k in keys:
        item = f[k]
        if isinstance(item, h5py.Dataset) and item.ndim == 2:
            return k
    return None


def _wavelengths_from_dict(d: dict, n_bands: int) -> np.ndarray | None:
    for key in ("wavelength", "wavelengths", "wl", "lambda", "bands"):
        for k in d.keys():
            if k.lower() == key:
                try:
                    w = np.asarray(d[k], dtype=np.float32).ravel()
                    if len(w) == n_bands:
                        return w
                except Exception:
                    pass
    return None


# --------------------------------------------------------------------------- loaders
def _load_mat(path: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
    import scipy.io as sio

    try:
        md = sio.loadmat(path)
        valid = {k: v for k, v in md.items() if not k.startswith("__")}
        data = None
        if "image" in valid and isinstance(valid["image"], np.ndarray):
            data = valid["image"]
        else:
            for v in valid.values():
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    data = v
                    break
            if data is None:
                for v in valid.values():
                    if isinstance(v, np.ndarray) and v.ndim == 2 and v.size > 64:
                        data = v
                        break
        if data is None:
            raise ValueError("No 2D/3D array found in MAT file")
        data = _to_hwb(data)
        return data, _wavelengths_from_dict(valid, data.shape[2]), {"format": "mat-v7"}
    except (NotImplementedError, ValueError, OSError) as e:
        if "No 2D/3D" in str(e):
            raise
    # v7.3 (HDF5)
    import h5py

    with h5py.File(path, "r") as f:
        key = _find_hdf5_dataset(f)
        if key is None:
            raise ValueError("No valid 2D/3D dataset found in v7.3 MAT file")
        data = f[key][:]
        # MATLAB stores column-major -> reverse axes
        data = np.transpose(data, (2, 1, 0)) if data.ndim == 3 else data.T
        wl = None
        for k in f.keys():
            if k.lower() in ("wavelength", "wavelengths", "wl"):
                try:
                    w = np.asarray(f[k][:], dtype=np.float32).ravel()
                    if len(w) == data.shape[-1]:
                        wl = w
                except Exception:
                    pass
        return _to_hwb(data), wl, {"format": "mat-v7.3", "dataset": key}


def _load_h5(path: str):
    import h5py

    with h5py.File(path, "r") as f:
        key = _find_hdf5_dataset(f)
        if key is None:
            raise ValueError("No valid 2D/3D dataset found")
        data = f[key][:]
        attrs = dict(f[key].attrs)
        wl = _wavelengths_from_dict(attrs, data.shape[-1] if data.ndim == 3 else 1)
        if data.ndim == 3:
            # If band axis is already last leave it; MATLAB-written files are reversed
            if data.shape[0] > data.shape[2]:
                pass
            else:
                data = np.transpose(data, (2, 1, 0))
        return _to_hwb(data), wl, {"format": "hdf5", "dataset": key}


def _load_tiff(path: str):
    import tifffile

    data = tifffile.imread(path)
    return _to_hwb(np.asarray(data)), None, {"format": "tiff"}


def _load_npy(path: str):
    data = np.load(path, mmap_mode="r")
    data = np.asarray(data)
    return _to_hwb(data), None, {"format": "npy"}


def _load_envi(path: str):
    import spectral.io.envi as envi

    hdr = path if path.lower().endswith(".hdr") else os.path.splitext(path)[0] + ".hdr"
    img = envi.open(hdr)
    data = np.asarray(img.load())
    wl = None
    try:
        bands = img.bands.centers
        if bands and len(bands) == data.shape[-1]:
            wl = np.asarray(bands, dtype=np.float32)
    except Exception:
        pass
    return _to_hwb(data), wl, {"format": "envi"}


def load_cube(path: str) -> Cube:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        data, wl, meta = _load_mat(path)
    elif ext in (".h5", ".hdf5"):
        data, wl, meta = _load_h5(path)
    elif ext in (".tif", ".tiff"):
        data, wl, meta = _load_tiff(path)
    elif ext == ".npy":
        data, wl, meta = _load_npy(path)
    elif ext in (".hdr", ".raw", ".img", ".dat"):
        data, wl, meta = _load_envi(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    data = np.ascontiguousarray(data, dtype=np.float32)
    meta["dtype_original"] = str(data.dtype)
    return Cube(data=data, path=path, wavelengths=wl, meta=meta)


def load_wavelength_file(path: str, n_bands: int) -> np.ndarray | None:
    """Load a plain text / csv list of wavelengths (one per band)."""
    try:
        w = np.loadtxt(path, delimiter=None, dtype=np.float32).ravel()
        if len(w) != n_bands:
            w = np.loadtxt(path, delimiter=",", dtype=np.float32).ravel()
        return w if len(w) == n_bands else None
    except Exception:
        return None

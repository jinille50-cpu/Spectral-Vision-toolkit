"""
Utility: .mat loading (v7 + v7.3), patch extraction, image reconstruction.
"""
import os
import numpy as np
import scipy.io as sio


def load_mat_image(path):
    """
    .mat 파일에서 가장 큰 3D array를 반환.
    MATLAB v7 (scipy) + v7.3 (h5py) 자동 감지.
    return: (H, W, C) np.ndarray
    """
    # 먼저 scipy로 시도 (v7 이하)
    scipy_failed = False
    try:
        data = sio.loadmat(path)
        candidates = [
            (k, v) for k, v in data.items()
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3
        ]
        if candidates:
            candidates.sort(key=lambda kv: kv[1].size, reverse=True)
            return candidates[0][1].astype(np.float32)
        # scipy로는 읽혔지만 3D 없음 → 그대로 에러
        raise RuntimeError(f"No 3D array found in {path}")
    except (NotImplementedError, ValueError) as e:
        # v7.3는 보통 NotImplementedError, 가끔 ValueError
        msg = str(e).lower()
        if any(k in msg for k in ["v7.3", "hdf", "mat-file", "unknown mat",
                                    "version 0"]):
            scipy_failed = True
        else:
            raise

    if not scipy_failed:
        raise RuntimeError(f"Unexpected: scipy did not flag v7.3 for {path}")

    # h5py로 시도
    try:
        import h5py
    except ImportError:
        raise RuntimeError(
            "MATLAB v7.3 file detected but h5py not installed. "
            "Run: pip install h5py"
        )

    with h5py.File(path, "r") as f:
        candidates = []

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                candidates.append((name, obj.shape, obj.size))

        f.visititems(_visit)
        if not candidates:
            raise RuntimeError(f"No 3D dataset found in {path}")

        candidates.sort(key=lambda x: x[2], reverse=True)
        name = candidates[0][0]

        # MATLAB이 v7.3로 저장 시 차원 순서가 reverse됨
        # MATLAB array (H, W, C) → HDF5 dataset shape (C, W, H)
        arr = f[name][()]
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 1, 0))
        return arr.astype(np.float32)


def find_mat_files(folder):
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(".mat")
    ])


def pad_to_size(image, target_size=128, mode="reflect"):
    H, W, C = image.shape
    pad_h = max(0, target_size - H)
    pad_w = max(0, target_size - W)
    if pad_h == 0 and pad_w == 0:
        return image
    return np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode=mode,
    )


def extract_patches(image, patch_size=128, stride=None):
    if stride is None:
        stride = patch_size // 2

    H, W, C = image.shape
    patches = []
    positions = []

    if H < patch_size or W < patch_size:
        return [image], [(0, 0)]

    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)

    for y in ys:
        for x in xs:
            patches.append(image[y:y + patch_size, x:x + patch_size, :])
            positions.append((y, x))
    return patches, positions


def reconstruct_from_patches(patches, positions, full_shape, patch_size=128):
    H, W, C = full_shape
    out = np.zeros(full_shape, dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    g = _gaussian_kernel(patch_size)

    for patch, (y, x) in zip(patches, positions):
        out[y:y + patch_size, x:x + patch_size, :] += patch * g[:, :, None]
        weight[y:y + patch_size, x:x + patch_size] += g

    weight = np.maximum(weight, 1e-8)
    out = out / weight[:, :, None]
    return out


def _gaussian_kernel(size):
    sigma = size / 4.0
    ax = np.arange(size) - (size - 1) / 2.0
    gauss = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return np.outer(gauss, gauss).astype(np.float32)

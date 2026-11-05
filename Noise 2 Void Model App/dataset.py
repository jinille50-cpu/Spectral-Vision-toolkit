"""
Dataset classes for self-supervised denoising methods.

Methods supported:
  - 2D N2V:  blind-spot mask in 2D (band별)
  - 3D N2V:  blind-spot mask in 3D (spatial+spectral chunk)
  - N2F:     Noise2Fast checkerboard split (band별)
  - N2N2:    Neighbor2Neighbor random subsample (band별)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from n2v_core import (
    generate_mask_2d, generate_mask_3d,
    replace_with_neighbor_2d, replace_with_neighbor_3d,
    checkerboard_split, neighbor_subsample_2d,
)


# ─────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────
def _safe_normalize(arr):
    """NaN/Inf 안전 + std=0 안전 normalize"""
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    m, s = arr.mean(), arr.std()
    if s < 1e-8 or not np.isfinite(s):
        s = 1.0
    if not np.isfinite(m):
        m = 0.0
    return (arr - m) / s


def _list_npy(patch_dir):
    files = sorted([
        os.path.join(patch_dir, f)
        for f in os.listdir(patch_dir) if f.endswith(".npy")
    ])
    if not files:
        raise RuntimeError(f"No .npy files in {patch_dir}")
    return files


# ─────────────────────────────────────────────────────────
# 2D N2V
# ─────────────────────────────────────────────────────────
class N2VDataset2D(Dataset):
    def __init__(self, patch_dir, mask_ratio=0.005, neighbor_radius=2,
                 bands_per_sample=8):
        self.files = _list_npy(patch_dir)
        self.mask_ratio = mask_ratio
        self.neighbor_radius = neighbor_radius
        self.bands_per_sample = max(1, bands_per_sample)
        sample = np.load(self.files[0])
        self.H, self.W, self.C = sample.shape

    def __len__(self):
        return len(self.files) * self.bands_per_sample

    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        patch = np.load(self.files[file_idx])
        band_idx = np.random.randint(0, self.C)
        img = patch[:, :, band_idx].astype(np.float32)
        img = _safe_normalize(img)

        mask = generate_mask_2d(img.shape, self.mask_ratio)
        masked = replace_with_neighbor_2d(img, mask, self.neighbor_radius)

        return (
            torch.from_numpy(masked).unsqueeze(0),
            torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )


# ─────────────────────────────────────────────────────────
# 3D N2V
# ─────────────────────────────────────────────────────────
class N2VDataset3D(Dataset):
    def __init__(self, patch_dir, spectral_chunk=64, mask_ratio=0.005, neighbor_radius=2):
        self.files = _list_npy(patch_dir)
        self.spectral_chunk = spectral_chunk
        self.mask_ratio = mask_ratio
        self.neighbor_radius = neighbor_radius
        sample = np.load(self.files[0])
        self.H, self.W, self.C = sample.shape
        if self.C < spectral_chunk:
            raise RuntimeError(
                f"Patch spectral dim ({self.C}) < chunk ({spectral_chunk})"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patch = np.load(self.files[idx])
        if self.C == self.spectral_chunk:
            start = 0
        else:
            start = np.random.randint(0, self.C - self.spectral_chunk + 1)
        chunk = patch[:, :, start:start + self.spectral_chunk]
        chunk = np.transpose(chunk, (2, 0, 1)).astype(np.float32)
        chunk = _safe_normalize(chunk)

        mask = generate_mask_3d(chunk.shape, self.mask_ratio)
        masked = replace_with_neighbor_3d(chunk, mask, self.neighbor_radius)

        return (
            torch.from_numpy(masked).unsqueeze(0),
            torch.from_numpy(chunk).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )


# ─────────────────────────────────────────────────────────
# N2F: Noise2Fast (checkerboard split)
# ─────────────────────────────────────────────────────────
class N2FDataset2D(Dataset):
    """
    Noise2Fast 학습용. Band별 2D 처리.

    각 __getitem__은 (input, target) 한 쌍을 반환:
      - 같은 band의 patch를 짝/홀수 행(또는 열)으로 split
      - 4가지 pair 중 random 하나 선택

    DataLoader는 그냥 stack → (B, 1, H/2, W) 또는 (B, 1, H, W/2) 4D 텐서
    
    주의: 같은 batch 안에 H/2 와 W/2 형태가 섞이면 안 됨.
    → 한 epoch 안에서 한 종류만 사용하는 식으로 단순화하거나,
      batch마다 같은 형태가 되도록 설계 필요.
    
    현재 구현: pair_kind를 idx로 결정해서 같은 batch 안 섞임 방지.
    """
    def __init__(self, patch_dir, bands_per_sample=8):
        self.files = _list_npy(patch_dir)
        self.bands_per_sample = max(1, bands_per_sample)
        sample = np.load(self.files[0])
        self.H, self.W, self.C = sample.shape

    def __len__(self):
        # files * bands_per_sample
        # pair kind는 epoch 마다 섞이게 - 같은 batch 안에선 같게 유지
        return len(self.files) * self.bands_per_sample

    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        patch = np.load(self.files[file_idx])
        band_idx = np.random.randint(0, self.C)
        img = patch[:, :, band_idx].astype(np.float32)
        img = _safe_normalize(img)

        # 4가지 split pair 중 random 선택
        pairs = checkerboard_split(img)
        pair_idx = np.random.randint(0, len(pairs))
        sub_in, sub_target = pairs[pair_idx]

        # 모양이 (H/2, W) 또는 (H, W/2) 둘 중 하나
        # batch 안에서 섞이면 stack 실패하므로 collate에서 처리할 수도 있지만
        # 간단히: 항상 가로 split (top/bot) → (H/2, W) 형태로 통일
        # → pairs[0] 또는 pairs[1] 만 사용

        # 실용적으로: 짝수 idx는 row split, 홀수 idx는 column split 인데
        # 같은 batch 안에선 row만 또는 column만 → 단순화 위해 row split만 사용
        sub_in, sub_target = pairs[np.random.randint(0, 2)]  # row split만

        return (
            torch.from_numpy(np.ascontiguousarray(sub_in)).unsqueeze(0),
            torch.from_numpy(np.ascontiguousarray(sub_target)).unsqueeze(0),
        )


# ─────────────────────────────────────────────────────────
# N2N2: Neighbor2Neighbor (random subsample)
# ─────────────────────────────────────────────────────────
class N2N2Dataset2D(Dataset):
    """
    Neighbor2Neighbor 학습용. Band별 2D 처리.
    각 __getitem__: (g1, g2) 두 sub-image (H/2, W/2).
    """
    def __init__(self, patch_dir, bands_per_sample=8):
        self.files = _list_npy(patch_dir)
        self.bands_per_sample = max(1, bands_per_sample)
        sample = np.load(self.files[0])
        self.H, self.W, self.C = sample.shape

    def __len__(self):
        return len(self.files) * self.bands_per_sample

    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        patch = np.load(self.files[file_idx])
        band_idx = np.random.randint(0, self.C)
        img = patch[:, :, band_idx].astype(np.float32)
        img = _safe_normalize(img)

        g1, g2 = neighbor_subsample_2d(img)
        return (
            torch.from_numpy(np.ascontiguousarray(g1)).unsqueeze(0),
            torch.from_numpy(np.ascontiguousarray(g2)).unsqueeze(0),
        )


# ─────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────
def build_dataset(method, mode, patch_dir, **kwargs):
    """
    method: 'N2V' / 'N2F' / 'N2N2'
    mode:   '2D' / '3D'  (3D는 N2V만 지원)
    """
    if method == "N2V" and mode == "2D":
        return N2VDataset2D(patch_dir,
                            mask_ratio=kwargs.get("mask_ratio", 0.005),
                            bands_per_sample=kwargs.get("bands_per_sample", 8))
    if method == "N2V" and mode == "3D":
        return N2VDataset3D(patch_dir,
                            spectral_chunk=kwargs.get("spectral_chunk", 64),
                            mask_ratio=kwargs.get("mask_ratio", 0.005))
    if method == "N2F" and mode == "2D":
        return N2FDataset2D(patch_dir,
                            bands_per_sample=kwargs.get("bands_per_sample", 8))
    if method == "N2N2" and mode == "2D":
        return N2N2Dataset2D(patch_dir,
                             bands_per_sample=kwargs.get("bands_per_sample", 8))
    raise ValueError(f"Unsupported combination: method={method}, mode={mode}")

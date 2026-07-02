"""모델 입력 크기 자동 패딩 유틸리티 (회귀용)"""
import numpy as np


_PAD_MAP = {
    # SMP CNN — encoder stride 32
    "UNet":       32,
    "UNet++":     32,
    "MAnet":      32,
    "FPN":        32,
    "Linknet":    32,
    "PAN":        32,
    "DeepLabV3+": 32,
}
_DEFAULT_MULTIPLE = 32


def get_pad_multiple(model_name: str) -> int:
    return _PAD_MAP.get(model_name, _DEFAULT_MULTIPLE)


def pad_np(arr: np.ndarray, multiple=32, is_label=False, fill_value=0) -> tuple:
    if multiple is None or multiple <= 0:
        multiple = _DEFAULT_MULTIPLE
    h, w = arr.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return arr, h, w

    if is_label:
        # 라벨/농도맵: 0 또는 특정값으로 채움
        pad_width = ((0, pad_h), (0, pad_w), (0, 0)) if arr.ndim == 3 else ((0, pad_h), (0, pad_w))
        padded = np.pad(arr, pad_width, mode='constant', constant_values=fill_value)
    else:
        # 이미지: reflect
        pad_width = ((0, pad_h), (0, pad_w), (0, 0)) if arr.ndim == 3 else ((0, pad_h), (0, pad_w))
        padded = np.pad(arr, pad_width, mode='reflect')
    return padded, h, w


def pad_tensor(x, multiple=32) -> tuple:
    import torch.nn.functional as F
    if multiple is None or multiple <= 0:
        multiple = _DEFAULT_MULTIPLE
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_pad, h, w


def unpad_tensor(x, orig_h, orig_w):
    return x[..., :orig_h, :orig_w]

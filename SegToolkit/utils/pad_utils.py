"""
pad_utils.py — 모델 입력 크기 제약을 자동으로 처리하는 공통 유틸리티

학습(train) 때 패딩한 크기 정보를 _config.json 에 저장하고,
평가(eval) / 추론(infer) 때 동일하게 패딩 → 결과 언패딩하여
원본 해상도를 그대로 유지합니다.

사용 예:
    from utils.pad_utils import get_pad_multiple, pad_np, pad_tensor, unpad_tensor

    # numpy 단계 (데이터셋 로딩 시)
    padded_img, orig_h, orig_w = pad_np(img_arr, multiple=32)
    padded_lbl, _, _           = pad_np(lbl_arr, multiple=32, is_label=True)

    # tensor 단계 (추론 / 평가 시)
    t_pad, orig_h, orig_w = pad_tensor(t, multiple=32)
    out_crop              = unpad_tensor(out, orig_h, orig_w)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
#  모델별 필요 배수 자동 결정
# ─────────────────────────────────────────────────────────────
_PAD_MAP = {
    # SMP 계열 — encoder stride 32
    "UNet":       32,
    "UNet++":     32,
    "MAnet":      32,
    "DeepLabV3":  32,
    "DeepLabV3+": 32,
    "FPN":        32,
    "PSPNet":     32,
    "PAN":        32,
    "Linknet":    32,
    # Transformer 계열 — 내부 patch/stride
    "SegFormer-B0": 4,
    "SegFormer-B2": 4,
    "SegFormer-B4": 4,
    "MobileViT":    32,
    "Mask2Former":  32,
}
_DEFAULT_MULTIPLE = 32


def get_pad_multiple(model_name: str) -> int:
    """모델 이름으로 필요한 패딩 배수를 반환합니다."""
    return _PAD_MAP.get(model_name, _DEFAULT_MULTIPLE)


# ─────────────────────────────────────────────────────────────
#  numpy 패딩 (데이터셋 로딩 단계)
# ─────────────────────────────────────────────────────────────
def pad_np(arr: np.ndarray, multiple = 32,
           is_label: bool = False) -> tuple:
    """
    numpy 배열을 multiple 배수 크기로 패딩합니다.

    Args:
        arr:       HW 또는 HWC 형태의 numpy 배열
        multiple:  패딩 기준 배수 (기본 32, None이면 32 사용)
        is_label:  True이면 constant(0) 패딩, False이면 reflect 패딩

    Returns:
        (padded_arr, orig_h, orig_w)
    """
    if multiple is None or multiple <= 0:
        multiple = _DEFAULT_MULTIPLE
    h, w = arr.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return arr, h, w

    if is_label:
        # 라벨: 0으로 채움 (클래스 값 보존)
        if arr.ndim == 3:
            padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)),
                            mode='constant', constant_values=0)
        else:
            padded = np.pad(arr, ((0, pad_h), (0, pad_w)),
                            mode='constant', constant_values=0)
    else:
        # 이미지: reflect 패딩 (경계 아티팩트 최소화)
        if arr.ndim == 3:
            padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
    return padded, h, w


# ─────────────────────────────────────────────────────────────
#  tensor 패딩 / 언패딩 (추론·평가 단계)
# ─────────────────────────────────────────────────────────────
def pad_tensor(x, multiple = 32) -> tuple:
    """
    torch Tensor (N,C,H,W) 또는 (C,H,W) 를 multiple 배수로 패딩합니다.

    Returns:
        (padded_tensor, orig_h, orig_w)
    """
    import torch.nn.functional as F
    if multiple is None or multiple <= 0:
        multiple = _DEFAULT_MULTIPLE
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return x, h, w

    # F.pad 순서: (left, right, top, bottom)
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_pad, h, w


def unpad_tensor(x, orig_h: int, orig_w: int):
    """패딩된 tensor를 원본 크기로 크롭합니다."""
    return x[..., :orig_h, :orig_w]


def unpad_np(arr: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """패딩된 numpy 배열을 원본 크기로 크롭합니다."""
    return arr[..., :orig_h, :orig_w]

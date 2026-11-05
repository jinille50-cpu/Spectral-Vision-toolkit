"""
Core utilities for self-supervised denoising methods.

Includes:
  - N2V: blind-spot masking + neighbor replacement (existing)
  - N2F: checkerboard downsampling
  - N2N2: random neighbor sub-sampling
"""
import numpy as np
import torch


# ─────────────────────────────────────────────────────────
# N2V: blind-spot masking
# ─────────────────────────────────────────────────────────
def generate_mask_2d(shape, mask_ratio=0.005):
    H, W = shape
    n_mask = max(1, int(H * W * mask_ratio))
    mask = np.zeros((H, W), dtype=bool)
    idx = np.random.choice(H * W, n_mask, replace=False)
    mask.flat[idx] = True
    return mask


def generate_mask_3d(shape, mask_ratio=0.005):
    D, H, W = shape
    n_mask = max(1, int(D * H * W * mask_ratio))
    mask = np.zeros((D, H, W), dtype=bool)
    idx = np.random.choice(D * H * W, n_mask, replace=False)
    mask.flat[idx] = True
    return mask


def replace_with_neighbor_2d(image, mask, radius=2):
    H, W = image.shape
    out = image.copy()
    ys, xs = np.where(mask)
    n = len(ys)
    if n == 0:
        return out

    dy = np.random.randint(-radius, radius + 1, size=n)
    dx = np.random.randint(-radius, radius + 1, size=n)
    zero_mask = (dy == 0) & (dx == 0)
    dy[zero_mask] = 1
    ny = np.clip(ys + dy, 0, H - 1)
    nx = np.clip(xs + dx, 0, W - 1)

    out[ys, xs] = image[ny, nx]
    return out


def replace_with_neighbor_3d(image, mask, radius=2):
    D, H, W = image.shape
    out = image.copy()
    ds, ys, xs = np.where(mask)
    n = len(ds)
    if n == 0:
        return out

    dy = np.random.randint(-radius, radius + 1, size=n)
    dx = np.random.randint(-radius, radius + 1, size=n)
    zero_mask = (dy == 0) & (dx == 0)
    dy[zero_mask] = 1
    ny = np.clip(ys + dy, 0, H - 1)
    nx = np.clip(xs + dx, 0, W - 1)

    out[ds, ys, xs] = image[ds, ny, nx]
    return out


def n2v_loss(pred, target, mask):
    """Masked MSE: mask 위치에서만 loss 계산."""
    m = mask.float()
    s = m.sum()
    if s.item() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    diff = (pred - target) ** 2
    return (diff * m).sum() / s


# ─────────────────────────────────────────────────────────
# N2F: Checkerboard downsampling (Lequyer et al., 2021)
# ─────────────────────────────────────────────────────────
def checkerboard_split(image):
    """
    Noise2Fast 'checkerboard downsampling'.

    image: (H, W) np.float32, H/W must be even (else 1px crop)
    Returns 4 sub-images of shape (H/2, W) and (H, W/2) ...
    실제로는 4가지 패턴: 짝수/홀수 행/열 조합

    원 논문은 (H/2, W) 또는 (H, W/2) 형태로 4개 sub-image를 만들어
    인접 pair (예: sub1 ↔ sub2)를 학습 input/target으로 사용.

    Returns:
      list of (sub_in, sub_target) pairs (all shape: (H/2, W) or (H, W/2))
    """
    H, W = image.shape
    # 짝수로 자르기
    H = H - (H % 2)
    W = W - (W % 2)
    img = image[:H, :W]

    # 4가지 sub-image (논문 정의)
    # type A: rows 짝수 + columns 홀수 → (H/2, W/2)... 가 아니라
    # 논문은 stripe-style:
    #   sub1 = 짝수행 → (H/2, W)
    #   sub2 = 홀수행 → (H/2, W)
    #   sub3 = 짝수열 → (H, W/2)
    #   sub4 = 홀수열 → (H, W/2)
    # 인접 행/열은 같은 signal에 가깝고 noise는 독립 → N2N pair
    s_top = img[0::2, :]   # (H/2, W) 짝수 행
    s_bot = img[1::2, :]   # (H/2, W) 홀수 행
    s_lft = img[:, 0::2]   # (H, W/2) 짝수 열
    s_rgt = img[:, 1::2]   # (H, W/2) 홀수 열

    # 인접 pair 4개 반환 (양방향 학습)
    pairs = [
        (s_top, s_bot),
        (s_bot, s_top),
        (s_lft, s_rgt),
        (s_rgt, s_lft),
    ]
    return pairs


# ─────────────────────────────────────────────────────────
# N2N2: Random neighbor sub-sampling (Huang et al., CVPR 2021)
# ─────────────────────────────────────────────────────────
def neighbor_subsample_2d(image):
    """
    Neighbor2Neighbor random sub-sampler.
    각 2x2 block 안에서 random하게 2개 픽셀 선택 → 두 sub-image (H/2, W/2).

    image: (H, W) np.float32
    Returns: (g1, g2) where g1, g2: (H/2, W/2)

    구현:
      - 각 2x2 block에서 4개 위치 (00,01,10,11) 중 random 2개 선택
      - g1: 첫 번째, g2: 두 번째
      - 두 픽셀이 인접하도록 제약 (논문 가정)
    """
    H, W = image.shape
    H = H - (H % 2)
    W = W - (W % 2)
    img = image[:H, :W]

    # 2x2 block 단위
    h2, w2 = H // 2, W // 2

    # 각 block마다 (g1_pos, g2_pos) 결정
    # 인접 pair만 가능: (00,01), (00,10), (01,11), (10,11)
    # 즉 4가지 인접 pair → random 선택
    # 0=00, 1=01, 2=10, 3=11
    adjacent_pairs = np.array([
        [0, 1],  # 00, 01 (수평)
        [0, 2],  # 00, 10 (수직)
        [1, 3],  # 01, 11 (수직)
        [2, 3],  # 10, 11 (수평)
    ])
    pair_idx = np.random.randint(0, 4, size=(h2, w2))
    selected = adjacent_pairs[pair_idx]  # (h2, w2, 2)

    # 각 위치를 (dy, dx)로 변환
    pos_dy = np.array([0, 0, 1, 1])  # 00,01,10,11 → dy
    pos_dx = np.array([0, 1, 0, 1])  #              → dx

    g1 = np.empty((h2, w2), dtype=image.dtype)
    g2 = np.empty((h2, w2), dtype=image.dtype)

    yy, xx = np.meshgrid(np.arange(h2), np.arange(w2), indexing='ij')
    base_y = yy * 2
    base_x = xx * 2

    g1_dy = pos_dy[selected[:, :, 0]]
    g1_dx = pos_dx[selected[:, :, 0]]
    g2_dy = pos_dy[selected[:, :, 1]]
    g2_dx = pos_dx[selected[:, :, 1]]

    g1 = img[base_y + g1_dy, base_x + g1_dx]
    g2 = img[base_y + g2_dy, base_x + g2_dx]

    return g1, g2


def n2n2_regularizer(g1, g2, fg1, fg2):
    """
    Neighbor2Neighbor regularization term.
    
    Loss = ||fg1 - g2||² + γ * ||fg1 - g2 - (fg_full_g1 - fg_full_g2)||²
    
    여기선 단순화 버전:
    Loss_reg = ||fg1 - g2 - (g1 - g2)||²
            = ||(fg1 - g1)||²  (linear case 단순화)
    
    실제 구현:
      fg1 = model(g1)  
      fg2 = model(g2)  (참고용)
      reg = ((fg1 - g2) - (g1 - g2))^2  → 단순화
      
    이 함수는 (fg1, fg2, g1, g2) → reg loss 계산.
    """
    # ((f(g1) - g2) - (g1 - g2))² = (f(g1) - g1)²  (단순화 ver.)
    # 논문 정확한 form은 아니지만 단순/안정적 구현
    diff = (fg1 - g2) - (g1 - g2)
    return (diff ** 2).mean()


def n2n2_loss(fg1, g2, g1=None, fg2=None, gamma=1.0):
    """
    Neighbor2Neighbor full loss.
    
    Basic: MSE(fg1, g2)
    + Regularization (옵션): γ * ||fg1 - g2 - sg(fg2 - g1)||²
        sg = stop gradient
    
    γ=0이면 순수 N2N pair training과 동일.
    """
    main = ((fg1 - g2) ** 2).mean()
    if g1 is not None and fg2 is not None and gamma > 0:
        # symmetric regularization with stop-gradient
        reg = ((fg1 - g2 - (fg2.detach() - g1)) ** 2).mean()
        return main + gamma * reg
    return main

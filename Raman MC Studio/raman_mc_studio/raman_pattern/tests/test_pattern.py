import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pattern  # noqa: E402
from geometry import Geometry  # noqa: E402


# ---------------------------------------------------------------- geometry
def test_volume_labels_and_cube_placement():
    g = Geometry(depth_top=3.0)
    vol = g.build_volume()
    assert vol.shape == (g.nx, g.ny, g.nz) == (320, 160, 110)
    assert (vol[:, :, :g.z_surface_vox] == 0).all()        # 공기
    sx, sy, sz = g.cube_slices()
    assert (vol[sx, sy, sz] == 2).all()
    assert (vol == 2).sum() == 20 ** 3                      # 2 mm 큐브 = 20³ 복셀
    assert sz.start == g.z_surface_vox + 30                  # 윗면 깊이 3 mm
    # 큐브 중심 = x 0 mm, y 0 mm
    assert (sx.start + sx.stop) / 2 == pytest.approx(g.x_to_vox(0.0), abs=0.5)
    assert (sy.start + sy.stop) / 2 == pytest.approx(g.y_to_vox(0.0), abs=0.5)


def test_source_voxel_and_bounds():
    g = Geometry()
    assert g.source_voxel(0.0) == [g.x_to_vox(0.0), g.y_to_vox(0.0), g.z_surface_vox]
    assert g.source_voxel(16.0)[0] == g.x_to_vox(0.0) + 160
    with pytest.raises(ValueError):
        g.source_voxel(30.0)
    with pytest.raises(ValueError):
        Geometry(depth_top=9.0).build_volume()              # 큐브가 조직 밑으로 빠짐


# ---------------------------------------------------------------- pattern
def synthetic_table(positions, decay=0.3):
    """Φ(v; p) = exp(-decay·|p|) 인 가짜 표 (큐브 2×2×2 복셀)"""
    phi = np.exp(-decay * np.asarray(positions))[:, None, None, None]
    return np.repeat(np.repeat(np.repeat(phi, 2, 1), 2, 2), 2, 3).astype(np.float32)


def test_ghost_peak_sits_at_offset_equal_to_laser_distance():
    pos = np.arange(0, 16.01, 0.2)
    phi = synthetic_table(pos)
    lasers = [0, 2, 4, 6, 8]
    offsets = np.arange(0, 8.01, 0.2)
    M = pattern.signal_matrix(phi, pos, lasers, offsets, side=-1)
    rows = pattern.ghost_metrics(M, lasers, offsets)
    for r in rows:
        # 센서가 큐브 바로 위(offset = 레이저 거리)에 올 때 최대
        assert r['peak_offset'] == pytest.approx(r['x_laser'], abs=0.11)
    # 레이저 0 mm 행: offset 0 이 최대이고 단조 감소
    assert np.all(np.diff(M[0]) <= 0)


def test_away_side_is_monotonic_and_weaker():
    pos = np.arange(0, 16.01, 0.2)
    phi = synthetic_table(pos)
    offsets = np.arange(0, 8.01, 0.2)
    toward = pattern.signal_matrix(phi, pos, [4.0], offsets, side=-1)[0]
    away = pattern.signal_matrix(phi, pos, [4.0], offsets, side=+1)[0]
    assert np.all(np.diff(away) <= 0)
    assert away[1:].max() < toward.max()
    assert toward[0] == pytest.approx(away[0])             # offset 0 은 같은 센서


def test_normalize_and_lookup_errors():
    pos = np.arange(0, 2.01, 0.2)
    phi = synthetic_table(pos)
    M = pattern.signal_matrix(phi, pos, [0.0, 1.0], [0.0, 1.0], side=-1)
    N = pattern.normalize(M)
    assert N[0, 0] == pytest.approx(1.0)
    assert pattern.signal(phi, pos, 1.0, 1.0, side=-1) == pytest.approx(M[1, 1])
    with pytest.raises(ValueError):
        pattern.signal(phi, pos, 3.0, 0.0)                  # 표 밖


def test_depth_profile_sums_to_one():
    pos = np.arange(0, 4.01, 0.2)
    phi = synthetic_table(pos)
    prof = pattern.depth_profile(phi, pos, 2.0, 2.0)
    assert prof.shape == (2,)
    assert prof.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- background
def test_tissue_background_matches_bruteforce_and_splits():
    import background
    g = Geometry(x_extent=(-1.0, 2.0), y_extent=(-0.5, 0.5), tissue_mm=1.0, cube_mm=0.2, depth_top=0.3)
    rng = np.random.default_rng(0)
    F = rng.random((g.nx, g.ny, g.nz)).astype(np.float32)
    offsets = [0.0, 0.3, 1.0]
    B, sh, deep = background.tissue_background(F, g, offsets, depth_split=0.5)
    z0 = g.z_surface_vox
    for j, d in enumerate(offsets):
        s = int(round(d / g.voxel))
        brute = sum(float(F[i, y, z] * F[i - s, y, z])
                    for i in range(s, g.nx) for y in range(g.ny) for z in range(z0, g.nz))
        assert B[j] == pytest.approx(brute, rel=1e-5)
        assert sh[j] + deep[j] == pytest.approx(B[j], rel=1e-9)
    assert B[0] > B[1] > B[2]


# ---------------------------------------------------------------- fat strand / cone / signed table
def test_fat_strand_only_on_plus_x_side_and_cone_cfg():
    from geometry import FatStrand
    g = Geometry(x_extent=(-10.0, 10.0), fat_strand=FatStrand(x_center=3.0, depth_center=2.0,
                                                           diameter=1.5, wobble=0.5), source='cone')
    vol = g.build_volume()
    fat = vol == 3
    assert fat.any()
    xs = np.where(fat.any(axis=(1, 2)))[0]
    assert xs.min() > g.x_to_vox(1.0) and xs.max() < g.x_to_vox(5.0)    # +x 쪽, 1~5 mm 안
    assert not fat[:, :, :g.z_surface_vox].any()                        # 공기층엔 없음
    assert (vol[g.cube_slices()] == 2).all()                             # 큐브는 지방에 덮이지 않음
    # 같은 seed 면 같은 모양 (재현성)
    assert (g.fat_mask() == Geometry(x_extent=(-10.0, 10.0), fat_strand=g.fat_strand).fat_mask()).all()
    cfg = g.source_cfg()
    assert cfg['srctype'] == 'cone' and cfg['srcparam1'][0] == pytest.approx(np.arcsin(0.23))
    assert Geometry().source_cfg() == {}
    assert len(g.prop()) == 4


def test_signed_lookup_uses_sign_when_table_has_negatives():
    pos = np.arange(-2.0, 2.01, 0.5)
    phi = np.zeros((len(pos), 1, 1, 1), np.float32)
    phi[:, 0, 0, 0] = pos + 10            # 비대칭 표
    assert pattern._lookup(phi, pos, -1.5)[0, 0, 0] == pytest.approx(8.5)
    assert pattern._lookup(phi, pos, 1.5)[0, 0, 0] == pytest.approx(11.5)
    sym = np.arange(0, 2.01, 0.5)
    assert pattern._lookup(phi[4:], sym, -1.5)[0, 0, 0] == pattern._lookup(phi[4:], sym, 1.5)[0, 0, 0]


# ---------------------------------------------------------------- stack (box sum / kernel stack)
def test_box_sum_matches_bruteforce():
    import run_stack
    rng = np.random.default_rng(1)
    P = rng.random((9, 7, 5))
    S = run_stack.box_sum(P, 3)
    assert S.shape == (7, 5, 3)
    for i, j, k in [(0, 0, 0), (6, 4, 2), (3, 2, 1)]:
        assert S[i, j, k] == pytest.approx(P[i:i + 3, j:j + 3, k:k + 3].sum())


def test_kernel_stack_matches_bruteforce_and_reciprocity():
    """가짜 플루언스로 kernel_stack 을 정의식(큐브 복셀 합)과 직접 비교. 고스트(레이저 −d)와 직진(레이저 0)은 상반성으로 비슷."""
    import run_stack
    g = Geometry(x_extent=(-14.0, 14.0), y_extent=(-8.0, 8.0), tissue_mm=10.0)
    x0, y0, z0 = g.source_voxel(0.0)
    X = (np.arange(g.nx) - x0)[:, None, None] * g.voxel
    Y = (np.arange(g.ny) - y0)[None, :, None] * g.voxel
    Z = (np.arange(g.nz) - z0)[None, None, :] * g.voxel
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    F = (np.exp(-2.0 * R) / (R + 0.2) ** 2).astype(np.float32)
    pos, stack = run_stack.kernel_stack(F, g, depth_top=2.0, scan_mm=12.0, step_mm=0.3)
    assert stack.shape == (40, 41, 41)
    n = 20
    sx, sy, sz = g.cube_slices()                           # 큐브 창 (depth_top = 2.0)

    def brute(rl_x_mm, rl_y_mm, ch):
        s = int(round((ch - 8) * 0.2 / g.voxel))
        rx, ry = int(round(rl_x_mm / g.voxel)), int(round(rl_y_mm / g.voxel))
        tot = 0.0
        for v in range(sx.start, sx.stop):
            u = v - rx
            tot += (F[u, sy.start - ry:sy.stop - ry, sz] * F[u - s, sy.start - ry:sy.stop - ry, sz]).sum()
        return tot

    cy = int(np.argmin(np.abs(pos)))
    for (xl, yl, ch) in [(0.0, 0.0, 8), (0.0, 0.0, 28), (-4.2, 0.0, 28), (2.1, -1.5, 15), (-3.0, 3.0, 1)]:
        ix, iy = int(np.argmin(np.abs(pos - xl))), int(np.argmin(np.abs(pos - yl)))
        assert stack[ch - 1, iy, ix] == pytest.approx(brute(pos[ix], pos[iy], ch), rel=1e-4)
    # 표면 채널: 최대가 중심. 채널 28 (offset +4 mm): 고스트는 x ≈ −4 mm 쪽에 있고 직진과 같은 크기 (±15%)
    assert np.unravel_index(np.argmax(stack[7]), stack[7].shape) == (cy, cy)
    line = stack[27, cy]
    j = int(np.argmin(np.abs(pos + 4.2)))
    assert line[j] == pytest.approx(line[cy], rel=0.15)
    assert line[int(np.argmin(np.abs(pos - 4.2)))] < 0.1 * line[cy]      # 반대쪽엔 고스트 없음

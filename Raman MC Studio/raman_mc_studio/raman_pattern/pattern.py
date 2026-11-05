# -*- coding: utf-8 -*-
"""라만 신호 패턴 행렬 (GUI/PMCX 비의존, 순수 numpy).

입력: phi_table[k, ...] = 광원이 큐브 중심에서 x 방향 positions[k] (mm, ≥ 0) 떨어져 있을 때
      큐브 내부 복셀의 플루언스. 매질이 x = 0 에 대해 대칭이므로 음수 위치는 |p| 로 조회.

신호(x_L, d) = Σ_voxel Φ(v; 광원 x_L) · Φ(v; 광원 x_L + s·d)
    x_L : 레이저 위치 (큐브 중심 기준, mm)
    d   : CCD offset (레이저 기준, mm, ≥ 0)
    s   : +1 이면 센서가 큐브에서 멀어지는 쪽, -1 이면 큐브 쪽 (toward)
라만 팩터(산란 단면적·농도)는 큐브 안에서 상수이므로 비율에는 영향 없음.
"""
import numpy as np


def _lookup(phi_table, positions, p):
    """p 에 가장 가까운 저장 위치의 Φ. 표가 음수 위치 없이 저장돼 있으면(대칭 매질) |p| 로 조회."""
    p = float(p)
    if positions.min() >= 0:
        p = abs(p)
    k = int(np.argmin(np.abs(positions - p)))
    if abs(positions[k] - p) > 0.5 * np.min(np.diff(positions)) + 1e-9:
        raise ValueError(f"No stored fluence near |p| = {p:.2f} mm")
    return phi_table[k]


def signal(phi_table, positions, x_laser, offset, side=-1):
    """단일 (레이저 위치, offset) 신호. side=-1: 센서가 큐브 쪽, +1: 반대쪽."""
    ex = _lookup(phi_table, positions, x_laser)
    em = _lookup(phi_table, positions, x_laser + side * offset)
    return float(np.sum(ex * em, dtype=np.float64))


def signal_matrix(phi_table, positions, laser_positions, offsets, side=-1):
    """행 = 레이저 위치, 열 = offset"""
    M = np.zeros((len(laser_positions), len(offsets)))
    for i, xl in enumerate(laser_positions):
        for j, d in enumerate(offsets):
            M[i, j] = signal(phi_table, positions, xl, d, side)
    return M


def normalize(M, ref=None):
    """기준값(기본: 레이저 0 mm, offset 0)으로 나눈 비율"""
    ref = M[0, 0] if ref is None else ref
    return M / ref if ref > 0 else M


def ghost_metrics(M, laser_positions, offsets):
    """레이저 위치별: 최대 신호 offset, 최대값, 레이저 0 mm 행 최대값 대비 비율"""
    direct_peak = M[0].max()
    rows = []
    for i, xl in enumerate(laser_positions):
        j = int(np.argmax(M[i]))
        rows.append(dict(x_laser=float(xl), peak_offset=float(offsets[j]), peak=float(M[i, j]),
                         peak_vs_direct=float(M[i, j] / direct_peak) if direct_peak > 0 else np.nan,
                         total=float(M[i].sum()),
                         total_vs_direct=float(M[i].sum() / M[0].sum()) if M[0].sum() > 0 else np.nan))
    return rows


def depth_profile(phi_table, positions, x_laser, offset, side=-1):
    """큐브 안 깊이(z 복셀 층)별 기여 비율 → 큐브 윗면/아랫면 중 어디서 신호가 생기는지"""
    ex = _lookup(phi_table, positions, x_laser)
    em = _lookup(phi_table, positions, x_laser + side * offset)
    prod = (ex * em).sum(axis=(0, 1))           # z 축만 남김
    tot = prod.sum()
    return prod / tot if tot > 0 else prod

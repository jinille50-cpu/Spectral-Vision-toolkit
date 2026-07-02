"""
MCX backend using PMCX (Monte Carlo eXtreme) — TRAJECTORY-based.
================================================================
Reference:
  Fang Q, Boas DA. "Monte Carlo simulation of photon migration in 3D
  turbid media accelerated by graphics processing units."
  Optics Express 17(22), 20178-20190 (2009).

핵심 변경 (이전 버전 대비)
--------------------------------
이전 버전은 fluence map(광자 밀도)에서 '무관한 점들'을 샘플링해 가짜 path 를
만들었기 때문에, 광자별 정체성이 없어 probed-depth / direct-side 분석이 불가능했음.

이 버전은 MCX 의 trajectory 출력(debuglevel='M')을 사용해 **광자별 실제 궤적**을 받고,
photon-id 로 그룹핑하여:
  - 각 광자의 z_max(실제 최대 도달 깊이)
  - 표면 복귀(backscatter) 광자의 exit 위치
를 정확히 계산한다. → offset별 probed-depth, direct/side 분석이 유효해짐.

trajectory 좌표는 voxel(grid) 단위로 가정하고 unitinmm 로 mm 변환.
(검증: banana/Propagation 그림의 깊이 스케일이 layer 두께와 맞는지 확인)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Layer:
    name: str = "Tissue"
    thickness: float = 5.0
    mua: float = 0.03
    mus: float = 10.0
    g: float = 0.9
    n: float = 1.4

    @property
    def mut(self) -> float:
        return self.mua + self.mus

    @property
    def mus_prime(self) -> float:
        return self.mus * (1 - self.g) * 10


@dataclass
class SimulationConfig:
    layers: List[Layer] = field(default_factory=list)
    n_photons: int = 1_000_000
    source_pos: tuple = (0.0, 0.0, 0.0)
    source_dir: tuple = (0.0, 0.0, 1.0)
    detector_radius: float = 0.3
    max_offset: float = 8.0
    weight_threshold: float = 1e-4
    save_paths: bool = True
    n_paths_to_save: int = 100
    max_steps: int = 5000
    voxel_size: float = 0.1
    seed: int = 1648335518          # 고정 seed → 재현성. 0 이면 매번 다른 결과.
    traj_photon_cap: int = 100_000  # trajectory 기록용 광자 수 상한(메모리 보호)


@dataclass
class SimulationResult:
    detected_x: np.ndarray = None
    detected_y: np.ndarray = None
    detected_weight: np.ndarray = None
    detected_max_depth: np.ndarray = None      # 실제 z_max (mm)
    paths: List[np.ndarray] = field(default_factory=list)
    n_simulated: int = 0
    n_detected: int = 0
    n_transmitted: int = 0
    n_absorbed: int = 0
    total_thickness: float = 0.0


def simulate(config: SimulationConfig, progress_callback=None) -> SimulationResult:
    try:
        import pmcx
    except ImportError as e:
        raise RuntimeError(f"PMCX not installed: {e}")

    if not config.layers:
        raise ValueError("최소 1개 layer 필요")

    layers = config.layers
    total_thickness = sum(l.thickness for l in layers)
    voxel_size = config.voxel_size

    grid_xy_mm = config.max_offset * 2 + 4
    grid_z_mm = total_thickness + 2          # 표면 위 공기 여유
    nx = int(grid_xy_mm / voxel_size)
    ny = int(grid_xy_mm / voxel_size)
    nz = int(grid_z_mm / voxel_size)

    if progress_callback:
        progress_callback(0, config.n_photons)

    # 볼륨: 0=air, 1+=layers
    vol = np.zeros((nx, ny, nz), dtype='uint8')
    air_top_vox = int(1.0 / voxel_size)      # 표면 위 1mm 공기
    z_off = 0
    for i, layer in enumerate(layers):
        n_lvox = int(layer.thickness / voxel_size)
        z_start = air_top_vox + z_off
        z_end = min(z_start + n_lvox, nz)
        vol[:, :, z_start:z_end] = i + 1
        z_off += n_lvox

    prop = [[0, 0, 1, 1]]
    for layer in layers:
        prop.append([layer.mua, layer.mus, layer.g, layer.n])

    src_x = nx // 2
    src_y = ny // 2
    src_z = air_top_vox

    det_radius_vox = config.max_offset / voxel_size
    det_pos = [src_x, src_y, air_top_vox, det_radius_vox]

    # trajectory 기록은 광자 수에 비례해 메모리를 먹으므로 상한 적용
    n_traj = min(int(config.n_photons), int(config.traj_photon_cap))

    cfg = {
        'nphoton': n_traj,
        'vol': vol,
        'tstart': 0, 'tend': 5e-9, 'tstep': 5e-9,
        'srcpos': [src_x, src_y, src_z],
        'srcdir': list(config.source_dir),
        'prop': prop,
        'detpos': [det_pos],
        'savedetflag': 'dspxvw',
        'maxdetphoton': n_traj,
        'issrcfrom0': 1,
        'isreflect': 1,
        'issaveexit': 1,
        'issaveref': 0,
        'unitinmm': voxel_size,
        'autopilot': 1,
        # ── trajectory 출력 ──
        'debuglevel': 'M',
        'maxjumpdebug': 80_000_000,          # 충분히 크게(점 수 상한)
    }
    if config.seed and int(config.seed) != 0:
        cfg['seed'] = int(config.seed)        # 재현성

    if progress_callback:
        progress_callback(int(config.n_photons * 0.1), config.n_photons)

    try:
        res = pmcx.run(cfg)
    except Exception as e:
        raise RuntimeError(f"PMCX run failed: {e}")

    if progress_callback:
        progress_callback(int(config.n_photons * 0.6), config.n_photons)

    traj = res.get('traj', None)
    if traj is None or traj.size == 0:
        raise RuntimeError("PMCX returned no trajectory. "
                           "debuglevel='M' / maxjumpdebug 확인.")

    # traj: (6, N) → row0=id, row1=x, row2=y, row3=z, row4=weight (voxel 좌표)
    # ※ row0(id)은 정수가 아니라 per-photon 상수 float 로 인코딩됨.
    #    int 로 캐스팅하면 모두 0으로 뭉개지므로 float 그대로 그룹 키로 사용한다.
    ids = traj[0]
    x_mm = (traj[1] - src_x) * voxel_size
    y_mm = (traj[2] - src_y) * voxel_size
    z_mm = (traj[3] - air_top_vox) * voxel_size      # 0 = 조직 표면, +z = 깊이
    w_pt = traj[4] if traj.shape[0] > 4 else np.ones_like(ids, dtype=float)

    # photon-id 로 그룹핑 (안정 정렬: 그룹 내부 시간순 보존)
    order = np.argsort(ids, kind='stable')
    ids_s = ids[order]
    xs, ys, zs, ws = x_mm[order], y_mm[order], z_mm[order], w_pt[order]

    boundaries = np.where(np.diff(ids_s) != 0)[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(ids_s)])) - 1   # 각 그룹 마지막 index

    # ── 광자별 통계 (벡터화) ──
    z_max_per = np.maximum.reduceat(zs, starts)             # 각 광자 최대 깊이
    exit_x_per = xs[ends]                                   # 마지막 점 = exit
    exit_y_per = ys[ends]
    exit_z_per = zs[ends]
    exit_w_per = ws[ends]

    # 표면 복귀(backscatter=검출) 판정: 마지막 점이 표면 부근 & 조직에 들어갔던 광자
    surface_tol = 0.2
    det_mask = (exit_z_per <= surface_tol) & (z_max_per > 0.1)

    detected_x = exit_x_per[det_mask]
    detected_y = exit_y_per[det_mask]
    detected_w = np.clip(exit_w_per[det_mask], 1e-12, None)
    detected_d = z_max_per[det_mask]

    if progress_callback:
        progress_callback(int(config.n_photons * 0.85), config.n_photons)

    # ── 시각화용 실제 path 일부 추출 ──
    paths_list = []
    n_target = min(int(config.n_paths_to_save), 5000)
    n_groups = len(starts)
    if n_target > 0 and n_groups > 0:
        # backscatter 광자 우선으로 골고루 샘플
        cand = np.where(det_mask)[0]
        if cand.size < n_target:
            cand = np.arange(n_groups)
        pick = cand[np.linspace(0, cand.size - 1, min(n_target, cand.size)).astype(int)]
        for gi in pick:
            s, e = starts[gi], ends[gi] + 1
            if e - s >= 2:
                paths_list.append(np.column_stack([xs[s:e], ys[s:e], zs[s:e]]))

    if progress_callback:
        progress_callback(config.n_photons, config.n_photons)

    # n_transmitted: 바닥(z >= 두께) 도달 광자(근사)
    n_trans = int(np.sum(z_max_per >= total_thickness - voxel_size))

    return SimulationResult(
        detected_x=detected_x,
        detected_y=detected_y,
        detected_weight=detected_w,
        detected_max_depth=detected_d,
        paths=paths_list,
        n_simulated=n_traj,
        n_detected=int(det_mask.sum()),
        n_transmitted=n_trans,
        n_absorbed=max(0, n_groups - int(det_mask.sum()) - n_trans),
        total_thickness=total_thickness,
    )


def compute_offset_profile(result: SimulationResult,
                            n_bins: int = 40,
                            max_offset: float = 8.0):
    if result.detected_x is None or len(result.detected_x) == 0:
        return (np.linspace(0, max_offset, n_bins),
                np.zeros(n_bins), np.zeros(n_bins))

    r = np.sqrt(result.detected_x**2 + result.detected_y**2)
    bin_edges = np.linspace(0, max_offset, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    intensity = np.zeros(n_bins)
    mean_depth = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i+1])
        if mask.sum() > 0:
            w = result.detected_weight[mask]
            intensity[i] = w.sum()
            mean_depth[i] = (result.detected_max_depth[mask] * w).sum() / w.sum()
    return bin_centers, intensity, mean_depth

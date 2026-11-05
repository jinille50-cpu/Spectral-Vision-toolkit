import numpy as np
from dataclasses import dataclass, field
from typing import List

from .analysis import PathHist, N_BASE_GROUPS


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
    traj_photon_cap: int = 100_000  # batch당 광자 수(trajectory 메모리 보호)
    source_type: str = 'pencil'     # 'pencil' | 'gaussian' | 'flattop' | 'line'
    source_diameter: float = 0.0    # mm. gaussian: 1/e² 직경, flattop: 직경
    line_length: float = 0.0        # mm
    line_width: float = 0.0         # mm. 폭 프로파일 gaussian 이면 1/e² 전폭
    line_orient: str = 'parallel'   # offset 축(x)과 'parallel' | 'perpendicular'
    line_profile: str = 'uniform'   # 폭 방향 'uniform' | 'gaussian'


@dataclass
class SimulationResult:
    # 검출 광자별 (float32 / 정수형) — 궤적 자체는 저장하지 않음
    detected_x: np.ndarray = None
    detected_y: np.ndarray = None
    detected_weight: np.ndarray = None
    detected_max_depth: np.ndarray = None      # 실제 z_max (mm)
    detected_pathlen: np.ndarray = None        # 조직 내 광경로 길이 (mm, trajectory 점 간 거리 합)
    detected_nscat: np.ndarray = None          # 산란 횟수 (trajectory 점 수 - 2, uint16)
    detected_group: np.ndarray = None          # 광자 index % N_BASE_GROUPS (uint8, 불확도 그룹)
    path_hist: object = None                   # analysis.PathHist (띠 검출 광자 산란점 히스토그램)
    paths: List[np.ndarray] = field(default_factory=list)
    n_simulated: int = 0
    n_detected: int = 0
    n_transmitted: int = 0
    n_absorbed: int = 0
    total_thickness: float = 0.0


def _pmcx_source(config: SimulationConfig, src_x, src_y, src_z, voxel_size, half_grid_mm):
    """MCX srctype 설정 (단위: voxel). pencil 은 빈 dict → 기존 cfg 그대로."""
    t = config.source_type
    if t == 'pencil':
        return {}
    if t in ('gaussian', 'flattop'):
        R = config.source_diameter / 2
        if R > half_grid_mm:
            raise ValueError("광원 직경이 시뮬레이션 영역보다 큽니다.")
        # gaussian: srcparam1[0] = 1/e² 반경 (MCX: r = sqrt(-0.5 ln U)·w)
        # disk:     srcparam1[0] = 반경, 균일
        return {'srctype': 'gaussian' if t == 'gaussian' else 'disk',
                'srcparam1': [R / voxel_size, 0, 0, 0]}
    if t == 'line':
        L, W = config.line_length, config.line_width
        a = np.array([1.0, 0, 0]) if config.line_orient == 'parallel' else np.array([0, 1.0, 0])
        b = np.array([0, 1.0, 0]) if config.line_orient == 'parallel' else np.array([1.0, 0, 0])
        box_w = 2 * W if config.line_profile == 'gaussian' else W   # gaussian: ±W(=±4σ) 범위
        if max(L, box_w) / 2 > half_grid_mm:
            raise ValueError("line 광원이 시뮬레이션 영역보다 큽니다.")
        corner = (np.array([src_x, src_y, src_z], dtype=float)
                  - (L / 2) * a / voxel_size - (box_w / 2) * b / voxel_size)
        p1 = list(L * a / voxel_size)
        p2 = list(box_w * b / voxel_size)
        if config.line_profile == 'gaussian' and W > 0:
            # 폭 방향 1D gaussian 가중치 pattern (param1 방향 1칸, param2 방향 K칸)
            K = 101
            v = -W + (np.arange(K) + 0.5) * (2 * W / K)
            pat = np.exp(-8 * v**2 / W**2)
            pat = (pat / pat.mean()).astype(np.float32).reshape(1, K)
            return {'srctype': 'pattern', 'srcpos': list(corner),
                    'srcparam1': p1 + [1], 'srcparam2': p2 + [K], 'srcpattern': pat}
        # 균일 폭: 평행사변형 planar (시준)
        return {'srctype': 'planar', 'srcpos': list(corner),
                'srcparam1': p1 + [0], 'srcparam2': p2 + [0]}
    raise ValueError(f"Unknown source_type: {t}")


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

    # trajectory 기록은 광자 수에 비례해 메모리를 먹으므로,
    # 전체 광자를 batch(traj_photon_cap 단위)로 나눠 모두 시뮬레이션한다.
    n_total = int(config.n_photons)
    batch_size = max(1, int(config.traj_photon_cap))
    n_batches = (n_total + batch_size - 1) // batch_size

    base_cfg = {
        'vol': vol,
        'tstart': 0, 'tend': 5e-9, 'tstep': 5e-9,
        'srcpos': [src_x, src_y, src_z],
        'srcdir': list(config.source_dir),
        'prop': prop,
        'detpos': [det_pos],
        'savedetflag': 'dspxvw',
        'issrcfrom0': 1,
        'isreflect': 1,
        'issaveexit': 1,
        'issaveref': 0,
        'unitinmm': voxel_size,
        'autopilot': 1,
        **_pmcx_source(config, src_x, src_y, src_z, voxel_size, grid_xy_mm / 2),
        # ── trajectory 출력 ──
        'debuglevel': 'M',
        'maxjumpdebug': 80_000_000,          # batch당 점 수 상한
    }

    surface_tol = 0.2
    n_target = min(int(config.n_paths_to_save), 5000)

    det_x_parts, det_y_parts, det_w_parts, det_d_parts = [], [], [], []
    det_grp_parts, det_len_parts, det_ns_parts = [], [], []
    # 띠 검출기 폭 = Input 탭 "Detector strip width" (실행 시 고정)
    hist = PathHist(layers, config.max_offset, config.detector_radius)
    paths_list = []
    n_groups_total = 0
    n_trans_total = 0
    n_done = 0

    for b in range(n_batches):
        nph = min(batch_size, n_total - n_done)
        cfg = dict(base_cfg, nphoton=nph, maxdetphoton=nph)
        if config.seed and int(config.seed) != 0:
            cfg['seed'] = int(config.seed) + b   # batch마다 seed를 달리해 결과 중복 방지

        try:
            res = pmcx.run(cfg)
            res.pop('flux', None)
        except Exception as e:
            raise RuntimeError(f"PMCX run failed (batch {b + 1}/{n_batches}): {e}")

        traj = res.get('traj', None)
        if traj is None or traj.size == 0:
            raise RuntimeError("PMCX returned no trajectory. "
                               "debuglevel='M' / maxjumpdebug 확인.")

        # traj: (6, N) → row0=id, row1=x, row2=y, row3=z, row4=weight (voxel 좌표)
        # ※ row0(id)은 정수가 아니라 per-photon 상수 float 로 인코딩됨.
        #    int 로 캐스팅하면 모두 0으로 뭉개지므로 float 그대로 그룹 키로 사용한다.
        ids = traj[0]

        # photon-id 로 그룹핑 (안정 정렬: 그룹 내부 시간순 보존)
        # id 의 float 비트 = 정수 광자 번호(0..nph-1) → 16비트 기수 정렬 2회 (float 정렬보다 ~2배 빠름)
        iv = ids.view(np.uint32)
        order = np.argsort((iv & 0xFFFF).astype(np.uint16), kind='stable')
        order = order[np.argsort((iv[order] >> 16).astype(np.uint16), kind='stable')]
        ids_s = iv[order]
        xs = traj[1][order]; xs -= src_x; xs *= voxel_size
        ys = traj[2][order]; ys -= src_y; ys *= voxel_size
        zs = traj[3][order]; zs -= air_top_vox; zs *= voxel_size   # 0 = 조직 표면, +z = 깊이

        boundaries = np.where(np.diff(ids_s) != 0)[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(ids_s)])) - 1   # 각 그룹 마지막 index

        # ── 광자별 통계 (벡터화) ──
        z_max_per = np.maximum.reduceat(zs, starts)             # 각 광자 최대 깊이
        exit_x_per = xs[ends]                                   # 마지막 점 = exit
        exit_y_per = ys[ends]
        exit_z_per = zs[ends]
        exit_w_per = (traj[4][order[ends]] if traj.shape[0] > 4
                      else np.ones(len(ends), dtype=np.float32))

        # 표면 복귀(backscatter=검출) 판정: 마지막 점이 표면 부근 & 조직에 들어갔던 광자
        det_mask = (exit_z_per <= surface_tol) & (z_max_per > 0.1)

        det_x_parts.append(exit_x_per[det_mask])
        det_y_parts.append(exit_y_per[det_mask])
        det_w_parts.append(np.clip(exit_w_per[det_mask], 1e-12, None))
        det_d_parts.append(z_max_per[det_mask])

        # 광경로 길이 = trajectory 점 간 거리 합, 산란 횟수 = 점 수 - 2 (출발점·탈출점 제외)
        seg = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2 + np.diff(zs)**2)
        seg[ends[:-1]] = 0.0                                    # 광자 경계 구간 제거
        seg = np.append(seg, 0.0)
        len_per = np.add.reduceat(seg, starts)
        cnt_per = ends - starts + 1
        bgrp = ((np.arange(len(starts)) + n_done) % N_BASE_GROUPS).astype(np.uint8)
        det_grp_parts.append(bgrp[det_mask])
        det_len_parts.append(len_per[det_mask].astype(np.float32))
        det_ns_parts.append(np.clip(cnt_per[det_mask] - 2, 0, 65535).astype(np.uint16))

        # 띠 검출 광자의 산란점(출발점·탈출점 제외) → 히스토그램 누적 후 궤적 폐기
        side, ch = hist.photon_bins(exit_x_per, exit_y_per)
        ch = np.where(det_mask, ch, -1)
        strip_ph = np.where(ch >= 0)[0]
        if strip_ph.size:
            lens = cnt_per[strip_ph]
            ph = np.repeat(strip_ph, lens)
            pt = (np.repeat(starts[strip_ph] - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
                  + np.arange(lens.sum()))
            inner = (pt != starts[ph]) & (pt != ends[ph])
            ph, pt = ph[inner], pt[inner]
            hist.add(bgrp[ph], side[ph], ch[ph], xs[pt], zs[pt], exit_w_per[ph])

        n_groups = len(starts)
        n_groups_total += n_groups
        # 투과 = 마지막 점이 바닥면 (바닥에 닿았다가 내부반사로 돌아온 광자는 제외)
        n_trans_total += int(np.sum(exit_z_per >= total_thickness - surface_tol))

        # ── 시각화용 실제 path 일부 추출 (필요한 만큼만) ──
        n_need = n_target - len(paths_list)
        if n_need > 0 and n_groups > 0:
            # backscatter 광자 우선으로 골고루 샘플
            cand = np.where(det_mask)[0]
            if cand.size < n_need:
                cand = np.arange(n_groups)
            pick = cand[np.linspace(0, cand.size - 1,
                                    min(n_need, cand.size)).astype(int)]
            for gi in pick:
                s, e = starts[gi], ends[gi] + 1
                if e - s >= 2:
                    paths_list.append(np.column_stack([xs[s:e], ys[s:e], zs[s:e]]))

        n_done += nph
        if progress_callback:
            progress_callback(n_done, n_total)

    detected_x = np.concatenate(det_x_parts) if det_x_parts else np.array([])
    detected_y = np.concatenate(det_y_parts) if det_y_parts else np.array([])
    detected_w = np.concatenate(det_w_parts) if det_w_parts else np.array([])
    detected_d = np.concatenate(det_d_parts) if det_d_parts else np.array([])
    n_detected = int(len(detected_x))

    def _cat(parts, dtype):
        return np.concatenate(parts) if parts else np.array([], dtype=dtype)

    return SimulationResult(
        detected_x=detected_x,
        detected_y=detected_y,
        detected_weight=detected_w,
        detected_max_depth=detected_d,
        detected_pathlen=_cat(det_len_parts, np.float32),
        detected_nscat=_cat(det_ns_parts, np.uint16),
        detected_group=_cat(det_grp_parts, np.uint8),
        path_hist=hist,
        paths=paths_list,
        n_simulated=n_done,
        n_detected=n_detected,
        n_transmitted=n_trans_total,
        n_absorbed=max(0, n_groups_total - n_detected - n_trans_total),
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

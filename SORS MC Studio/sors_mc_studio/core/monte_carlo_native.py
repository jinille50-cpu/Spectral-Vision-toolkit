"""
Monte Carlo Photon Transport — Vectorized Version
==================================================
다층 매질에서 광자의 random walk를 numpy 벡터화로 시뮬레이션.

벡터화 전략:
  - N개 광자를 한 번에 처리 (각 광자가 독립이므로 가능)
  - alive[N] mask로 종료 여부 관리
  - 매 step마다 alive인 광자만 갱신
  - 종료 조건 (탈출/흡수/통과) 시 alive=False

성능:
  - 비벡터화: 1만 광자 × 100 step = 10만 iteration (Python loop)
  - 벡터화:   100 step × N차원 numpy 연산
  - 결과: 10~50배 빠름

알고리즘:
  1. Step: distance = -ln(rand) / mut
  2. 이동: pos += step * dir
  3. 경계 검사: z<0 (탈출) or z>thickness (통과)
  4. Absorption: weight *= (1 - mua/mut)
  5. Henyey-Greenstein 산란: cos_theta from rand
  6. 방향 갱신
  7. Russian roulette: weight < threshold
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

from .analysis import PathHist, N_BASE_GROUPS


@dataclass
class Layer:
    name: str = "Tissue"
    thickness: float = 5.0      # mm
    mua: float = 0.03           # mm⁻¹
    mus: float = 10.0           # mm⁻¹
    g: float = 0.9
    n: float = 1.4

    @property
    def mut(self) -> float:
        return self.mua + self.mus

    @property
    def mus_prime(self) -> float:
        """Reduced scattering coefficient in cm⁻¹"""
        return self.mus * (1 - self.g) * 10


@dataclass
class SimulationConfig:
    layers: List[Layer] = field(default_factory=list)
    n_photons: int = 100_000
    source_pos: tuple = (0.0, 0.0, 0.0)
    source_dir: tuple = (0.0, 0.0, 1.0)
    detector_radius: float = 0.3
    max_offset: float = 8.0
    weight_threshold: float = 1e-4
    save_paths: bool = True
    n_paths_to_save: int = 100
    max_steps: int = 5000
    seed: int = 1648335518          # 고정 seed → 재현성. 0 이면 매번 다른 결과.
    source_type: str = 'pencil'     # 'pencil' | 'gaussian' | 'flattop' | 'line'
    source_diameter: float = 0.0    # mm. gaussian: 1/e² 직경, flattop: 직경
    line_length: float = 0.0        # mm
    line_width: float = 0.0         # mm. 폭 프로파일 gaussian 이면 1/e² 전폭
    line_orient: str = 'parallel'   # offset 축(x)과 'parallel' | 'perpendicular'
    line_profile: str = 'uniform'   # 폭 방향 'uniform' | 'gaussian'
    batch_size: int = 100_000       # batch당 광자 수 (산란점 임시 기록 메모리 ∝ 이 값)


@dataclass
class SimulationResult:
    detected_x: np.ndarray = None
    detected_y: np.ndarray = None
    detected_weight: np.ndarray = None
    detected_max_depth: np.ndarray = None
    detected_pathlen: np.ndarray = None     # 조직 내 광경로 길이 (mm)
    detected_nscat: np.ndarray = None       # 산란 횟수 (uint16)
    detected_group: np.ndarray = None       # 광자 index % N_BASE_GROUPS (uint8, 불확도 그룹)
    path_hist: object = None                # analysis.PathHist (띠 검출 광자 산란점 히스토그램)
    paths: List[np.ndarray] = field(default_factory=list)
    n_simulated: int = 0
    n_detected: int = 0
    n_transmitted: int = 0
    n_absorbed: int = 0
    total_thickness: float = 0.0


def _build_layer_lookup(layers: List[Layer], voxel_resolution: float = 0.01):
    """
    z 좌표 → layer 인덱스 변환용 lookup table.
    매 step마다 z로부터 어느 layer인지 알아야 하므로 빠른 조회용 배열을 만듦.
    """
    total = sum(l.thickness for l in layers)
    n_voxels = int(np.ceil(total / voxel_resolution)) + 1
    z_to_layer = np.zeros(n_voxels, dtype=np.int32)
    cumul = 0.0
    for i, layer in enumerate(layers):
        start = int(cumul / voxel_resolution)
        cumul += layer.thickness
        end = int(cumul / voxel_resolution)
        z_to_layer[start:end + 1] = i
    return z_to_layer, voxel_resolution


def _sample_source_xy(config: SimulationConfig, N: int):
    """광원 형상에 따른 입사 위치 (x, y). pencil 은 난수를 쓰지 않음 (기존 결과 유지)."""
    x0, y0 = config.source_pos[0], config.source_pos[1]
    t = config.source_type
    if t == 'pencil':
        return np.full(N, x0, dtype=np.float64), np.full(N, y0, dtype=np.float64)
    if t in ('gaussian', 'flattop'):
        R = config.source_diameter / 2
        if t == 'gaussian':
            # MCX 와 동일: 강도 ∝ exp(-2r²/R²), R = 1/e² 반경
            r = R * np.sqrt(-0.5 * np.log(np.clip(np.random.rand(N), 1e-12, 1.0)))
        else:
            r = R * np.sqrt(np.random.rand(N))
        phi = 2 * np.pi * np.random.rand(N)
        return x0 + r * np.cos(phi), y0 + r * np.sin(phi)
    if t == 'line':
        u = config.line_length * (np.random.rand(N) - 0.5)
        if config.line_profile == 'gaussian':
            v = np.random.normal(0.0, config.line_width / 4, N)   # 1/e² 전폭 W → σ = W/4
        else:
            v = config.line_width * (np.random.rand(N) - 0.5)
        if config.line_orient == 'parallel':
            return x0 + u, y0 + v
        return x0 + v, y0 + u
    raise ValueError(f"Unknown source_type: {t}")


def simulate(config: SimulationConfig, progress_callback=None) -> SimulationResult:
    """
    광자를 batch_size 단위로 나눠 실행 (batch 안에서만 산란점을 기록 → 히스토그램 누적 후 폐기).
    광자 수 <= batch_size 이면 batch 1개 = 이전 버전과 같은 난수 순서.
    """
    if not config.layers:
        raise ValueError("최소 1개 layer 필요")
    N = int(config.n_photons)
    if config.seed and int(config.seed) != 0:
        np.random.seed(int(config.seed))
    # 띠 검출기 폭 = Input 탭 "Detector strip width" (실행 시 고정)
    hist = PathHist(config.layers, config.max_offset, config.detector_radius)
    batch = max(1, int(config.batch_size))
    parts, paths_list = [], []
    n_trans = n_abs = done = 0
    while done < N:
        nb = min(batch, N - done)
        n_save = (min(config.n_paths_to_save - len(paths_list), nb)
                  if config.save_paths else 0)
        cb = None
        if progress_callback:
            cb = (lambda d, t, _o=done: progress_callback(_o + d, N))
        r = _simulate_batch(config, nb, max(0, n_save), done, hist, cb)
        parts.append(r)
        paths_list += r['paths']
        n_trans += r['n_transmitted']
        n_abs += r['n_absorbed']
        done += nb

    def cat(k, dtype):
        return np.concatenate([r[k] for r in parts]).astype(dtype)

    det_x = cat('x', np.float32)
    return SimulationResult(
        detected_x=det_x,
        detected_y=cat('y', np.float32),
        detected_weight=cat('w', np.float32),
        detected_max_depth=cat('d', np.float32),
        detected_pathlen=cat('pathlen', np.float32),
        detected_nscat=np.clip(cat('nscat', np.int64), 0, 65535).astype(np.uint16),
        detected_group=cat('group', np.uint8),
        path_hist=hist,
        paths=paths_list,
        n_simulated=N,
        n_detected=len(det_x),
        n_transmitted=n_trans,
        n_absorbed=n_abs,
        total_thickness=sum(l.thickness for l in config.layers),
    )


def _simulate_batch(config: SimulationConfig, N, n_save, pid_offset, hist,
                    progress_callback=None):
    """
    벡터화 Monte Carlo simulation (광자 N 개 batch).

    핵심 트릭: 모든 광자를 동시에 한 step씩 진행시키되,
    종료된 광자는 alive=False로 처리하여 후속 연산에서 무시.
    """
    layers = config.layers
    n_layers = len(layers)
    total_thickness = sum(l.thickness for l in layers)

    # 각 layer의 광학 특성을 array로 (vectorized lookup)
    layer_mua = np.array([l.mua for l in layers])
    layer_mus = np.array([l.mus for l in layers])
    layer_mut = layer_mua + layer_mus
    layer_g = np.array([l.g for l in layers])

    # z → layer 인덱스 lookup
    z_to_layer, voxel_res = _build_layer_lookup(layers)

    # 광자 상태 array
    x, y = _sample_source_xy(config, N)
    z = np.full(N, config.source_pos[2], dtype=np.float64)
    ux = np.full(N, config.source_dir[0], dtype=np.float64)
    uy = np.full(N, config.source_dir[1], dtype=np.float64)
    uz = np.full(N, config.source_dir[2], dtype=np.float64)
    weight = np.ones(N, dtype=np.float64)
    max_depth = np.zeros(N, dtype=np.float64)
    alive = np.ones(N, dtype=bool)
    status = np.zeros(N, dtype=np.int8)  # 0=alive/escaped, 1=detected, 2=transmitted, 3=absorbed
    pathlen = np.zeros(N, dtype=np.float64)
    nscat = np.zeros(N, dtype=np.int32)
    log_parts = []                       # batch 안 산란점 (광자 index, x, z)

    # 경로 저장 (일부 광자만)
    save_path_ids = set()
    if n_save > 0:
        save_path_ids = set(np.random.choice(N, n_save, replace=False).tolist())
    path_records = {pid: [(x[pid], y[pid], z[pid])] for pid in save_path_ids}

    # 메인 루프 — 각 iteration은 살아있는 광자 전체를 한 step씩 진행
    for step_i in range(config.max_steps):
        if not alive.any():
            break

        alive_idx = np.where(alive)[0]
        n_alive = len(alive_idx)

        # 1. 현재 layer 인덱스 (각 광자별)
        z_alive = z[alive_idx]
        voxel_idx = np.clip((z_alive / voxel_res).astype(np.int32), 0, len(z_to_layer) - 1)
        layer_ids = z_to_layer[voxel_idx]

        # 2. 각 광자의 광학 특성
        mua_p = layer_mua[layer_ids]
        mus_p = layer_mus[layer_ids]
        mut_p = layer_mut[layer_ids]
        g_p = layer_g[layer_ids]

        # 3. Step size = -ln(rand) / mut
        r = np.random.rand(n_alive)
        r = np.clip(r, 1e-10, 1.0)
        step = -np.log(r) / mut_p

        # 4. 이동
        x[alive_idx] += step * ux[alive_idx]
        y[alive_idx] += step * uy[alive_idx]
        z[alive_idx] += step * uz[alive_idx]
        pathlen[alive_idx] += step

        # 최대 깊이 갱신
        new_z = z[alive_idx]
        old_max = max_depth[alive_idx]
        max_depth[alive_idx] = np.maximum(old_max, new_z)

        # 5. 경계 검사
        escaped = new_z < 0           # 표면 탈출
        transmitted = new_z > total_thickness  # 통과

        # 검출 (탈출 + 검출 범위 내)
        if escaped.any():
            esc_idx = alive_idx[escaped]
            pathlen[esc_idx] -= z[esc_idx] / uz[esc_idx]   # 표면 밖으로 나간 초과분 제외
            x_esc = x[esc_idx]
            y_esc = y[esc_idx]
            r_lateral = np.sqrt(x_esc**2 + y_esc**2)
            detected_mask = r_lateral <= config.max_offset
            det_idx = esc_idx[detected_mask]
            status[det_idx] = 1  # detected
            # 검출 안 된 탈출도 alive=False
            status[esc_idx[~detected_mask]] = 1  # 일단 detected 처리 (외부는 0으로 마스킹)
            alive[esc_idx] = False

        if transmitted.any():
            trans_idx = alive_idx[transmitted]
            status[trans_idx] = 2
            alive[trans_idx] = False

        # 살아남은 광자만 다음 단계
        still_alive = alive[alive_idx]
        alive_idx = alive_idx[still_alive]
        n_alive = len(alive_idx)
        if n_alive == 0:
            # 경로 저장 마무리
            for pid in save_path_ids:
                if pid not in path_records:
                    continue
                if alive[pid] == False and len(path_records[pid]) > 0:
                    path_records[pid].append((x[pid], y[pid], z[pid]))
            break

        # 6. Absorption: weight *= (1 - mua/mut)
        layer_ids_alive = layer_ids[still_alive]
        absorb_frac = layer_mua[layer_ids_alive] / layer_mut[layer_ids_alive]
        weight[alive_idx] *= (1 - absorb_frac)

        # 7. Russian roulette (weight < threshold)
        low_weight = weight[alive_idx] < config.weight_threshold
        if low_weight.any():
            low_idx = alive_idx[low_weight]
            # 10% 살리고 weight × 10, 90% 죽임
            survive = np.random.rand(len(low_idx)) < 0.1
            weight[low_idx[survive]] *= 10
            dead = low_idx[~survive]
            status[dead] = 3
            alive[dead] = False
            # 다시 alive 갱신
            still_alive2 = alive[alive_idx]
            alive_idx = alive_idx[still_alive2]
            n_alive = len(alive_idx)
            if n_alive == 0:
                break
            layer_ids_alive = layer_ids[still_alive][still_alive2]

        # 8. 산란 — Henyey-Greenstein
        nscat[alive_idx] += 1
        log_parts.append((alive_idx.astype(np.int32), x[alive_idx].astype(np.float32),
                          z[alive_idx].astype(np.float32)))
        g_p2 = layer_g[layer_ids_alive]
        r1 = np.random.rand(n_alive)
        r2 = np.random.rand(n_alive)

        # cos(theta) from HG distribution
        cos_theta = np.empty(n_alive)
        isotropic = np.abs(g_p2) < 1e-6
        if isotropic.any():
            cos_theta[isotropic] = 2 * r1[isotropic] - 1
        non_iso = ~isotropic
        if non_iso.any():
            g_ni = g_p2[non_iso]
            r1_ni = r1[non_iso]
            temp = (1 - g_ni**2) / (1 - g_ni + 2 * g_ni * r1_ni)
            cos_theta[non_iso] = (1 + g_ni**2 - temp**2) / (2 * g_ni)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        sin_theta = np.sqrt(1 - cos_theta**2)

        # azimuth
        psi = 2 * np.pi * r2
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # 방향 갱신 (MCML 공식)
        ux_a = ux[alive_idx]
        uy_a = uy[alive_idx]
        uz_a = uz[alive_idx]

        # near-vertical 광자 (|uz| ≈ 1) 처리
        near_vert = np.abs(uz_a) > 0.99999
        new_ux = np.empty(n_alive)
        new_uy = np.empty(n_alive)
        new_uz = np.empty(n_alive)

        if near_vert.any():
            new_ux[near_vert] = sin_theta[near_vert] * cos_psi[near_vert]
            new_uy[near_vert] = sin_theta[near_vert] * sin_psi[near_vert]
            new_uz[near_vert] = np.sign(uz_a[near_vert]) * cos_theta[near_vert]

        norm = ~near_vert
        if norm.any():
            uz_n = uz_a[norm]
            ux_n = ux_a[norm]
            uy_n = uy_a[norm]
            ct = cos_theta[norm]
            st = sin_theta[norm]
            cp = cos_psi[norm]
            sp = sin_psi[norm]
            denom = np.sqrt(1 - uz_n**2)
            new_ux[norm] = st * (ux_n*uz_n*cp - uy_n*sp) / denom + ux_n*ct
            new_uy[norm] = st * (uy_n*uz_n*cp + ux_n*sp) / denom + uy_n*ct
            new_uz[norm] = -st * cp * denom + uz_n*ct

        ux[alive_idx] = new_ux
        uy[alive_idx] = new_uy
        uz[alive_idx] = new_uz

        # 경로 저장 (매 5 step마다)
        if save_path_ids and step_i % 5 == 0:
            for pid in save_path_ids:
                if alive[pid]:
                    path_records[pid].append((x[pid], y[pid], z[pid]))

        # 진행 보고
        if progress_callback and step_i % 5 == 0:
            done_frac = 1.0 - n_alive / N
            progress_callback(int(done_frac * N), N)

    if progress_callback:
        progress_callback(N, N)

    # 결과 정리
    detected_mask = status == 1
    # 검출된 광자 중 max_offset 안에 들어온 것만
    if detected_mask.any():
        det_idx = np.where(detected_mask)[0]
        r_lateral = np.sqrt(x[det_idx]**2 + y[det_idx]**2)
        valid = r_lateral <= config.max_offset
        det_idx = det_idx[valid]
    else:
        det_idx = np.array([], dtype=np.int32)

    # 띠 검출 광자의 산란점 → 히스토그램 누적 (batch 기록은 여기서 폐기)
    group = ((det_idx + pid_offset) % N_BASE_GROUPS).astype(np.uint8)
    side_d, ch_d = hist.photon_bins(x[det_idx], y[det_idx])
    if log_parts and (ch_d >= 0).any():
        ph_ch = np.full(N, -1, dtype=np.int64)
        ph_side = np.zeros(N, dtype=np.int64)
        ph_grp = np.zeros(N, dtype=np.uint8)
        ph_ch[det_idx], ph_side[det_idx], ph_grp[det_idx] = ch_d, side_d, group
        lp = np.concatenate([p[0] for p in log_parts])
        keep = ph_ch[lp] >= 0
        lp = lp[keep]
        hist.add(ph_grp[lp], ph_side[lp], ph_ch[lp],
                 np.concatenate([p[1] for p in log_parts])[keep],
                 np.concatenate([p[2] for p in log_parts])[keep], weight[lp])
    del log_parts

    paths_list = []
    for pid in save_path_ids:
        if len(path_records[pid]) >= 2:
            paths_list.append(np.array(path_records[pid]))

    return dict(
        x=x[det_idx], y=y[det_idx], w=weight[det_idx], d=max_depth[det_idx],
        pathlen=pathlen[det_idx], nscat=nscat[det_idx], group=group,
        paths=paths_list,
        n_transmitted=int((status == 2).sum()),
        n_absorbed=int((status == 3).sum()),
    )


def compute_offset_profile(result: SimulationResult,
                            n_bins: int = 40,
                            max_offset: float = 8.0):
    """offset별 강도 + 평균 침투 깊이 계산"""
    if len(result.detected_x) == 0:
        return np.linspace(0, max_offset, n_bins), np.zeros(n_bins), np.zeros(n_bins)

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


if __name__ == "__main__":
    import time
    config = SimulationConfig(
        layers=[Layer(name="Meat", thickness=6.0, mua=0.03, mus=10.0, g=0.9, n=1.4)],
        n_photons=1_000_000,
        save_paths=True,
        n_paths_to_save=100,
    )
    t0 = time.time()
    result = simulate(config)
    dt = time.time() - t0
    print(f"Vectorized: {result.n_simulated:,} photons in {dt:.2f}s")
    print(f"  → {result.n_simulated/dt:,.0f} photons/sec")
    print(f"  Detected: {result.n_detected:,}")
    print(f"  Transmitted: {result.n_transmitted:,}")
    print(f"  Absorbed: {result.n_absorbed:,}")
    print(f"  Paths saved: {len(result.paths)}")

    centers, intensity, depth = compute_offset_profile(result)
    valid = depth > 0
    if valid.sum() > 2:
        slope = np.sum(centers[valid] * depth[valid]) / np.sum(centers[valid]**2)
        print(f"\nDepth/offset slope: {slope:.3f}")
        print(f"Propagation angle: {np.degrees(np.arctan(slope)):.1f}°")

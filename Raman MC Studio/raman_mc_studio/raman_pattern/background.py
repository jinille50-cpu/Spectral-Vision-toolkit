# -*- coding: utf-8 -*-
"""조직 자체 라만 배경 B(d).

레이저 위치에 관계없이 조직 전체가 라만을 내므로, offset d 센서의 배경은
    B(d) = Σ_tissue voxel Φ(v; 광원 0) · Φ(v; 광원 d)
균질 슬래브에서는 Φ(v; d) = Φ(v − d·x̂; 0) 이라 전체 볼륨 플루언스 한 번이면 모든 d 가 나온다.
(PET 큐브가 배경 광장에 주는 섭동은 무시 — 큐브 부피는 조직의 0.5% 미만)

PET 신호 / 조직 배경 비율은 단위 복셀당 라만 효율이 같다고 놓은 값이다.
실제 PET 와 조직의 라만 단면적 비 r 을 알면 비율에 r 을 곱하면 된다.
"""
import numpy as np

from geometry import Geometry


def full_fluence(geo: Geometry, nphoton: int, seed: int = 7):
    """광원 0 mm, 큐브 없는 균질 슬래브의 전체 볼륨 플루언스 (nx, ny, nz)"""
    import pmcx
    cfg = dict(
        nphoton=int(nphoton), vol=geo.build_volume(with_cube=False),
        tstart=0, tend=5e-9, tstep=5e-9,
        srcpos=geo.source_voxel(0.0), srcdir=[0, 0, 1],
        prop=geo.prop(), issrcfrom0=1, isreflect=1,
        unitinmm=geo.voxel, autopilot=1, seed=int(seed), outputtype='fluence',
    )
    return np.array(pmcx.run(cfg)['flux'][..., 0], dtype=np.float32)


def tissue_background(F, geo: Geometry, offsets, depth_split=None):
    """
    F: 전체 볼륨 플루언스 (광원 x = 0).  반환 B[d] (offset 별), 그리고
    depth_split (mm) 이 주어지면 (표면층 몫, 깊은 몫) 도 같이 반환.
    x 축으로 d 만큼 민 사본과의 곱을 조직 복셀에서 합산. 격자 밖으로 나간 부분은 0 처리.
    """
    z0 = geo.z_surface_vox
    T = F[:, :, z0:].astype(np.float64)               # 조직 영역만
    nx = T.shape[0]
    B = np.zeros(len(offsets))
    shallow = np.zeros(len(offsets))
    zs = None
    if depth_split is not None:
        zs = int(round(depth_split / geo.voxel))
    for j, d in enumerate(offsets):
        s = int(round(d / geo.voxel))
        if s >= nx:
            continue
        # Φ(v; d) = Φ(v − d) : index i 의 값은 T[i − s]
        prod = T[s:] * T[:nx - s]
        B[j] = prod.sum()
        if zs is not None:
            shallow[j] = prod[:, :, :zs].sum()
    if zs is None:
        return B
    return B, shallow, B - shallow


def check_translation_symmetry(F, geo: Geometry, d_mm: float):
    """표가 x 대칭이라는 가정 점검: Φ(x_s + a) 와 Φ(x_s − a) 의 상대 차이 (조직 상층 1 mm)"""
    xs = geo.x_to_vox(0.0)
    a = int(round(d_mm / geo.voxel))
    z0, z1 = geo.z_surface_vox, geo.z_surface_vox + int(round(1.0 / geo.voxel))
    y = geo.y_to_vox(0.0)
    r, l = F[xs + a, y, z0:z1].sum(), F[xs - a, y, z0:z1].sum()
    return abs(r - l) / max(r, l)

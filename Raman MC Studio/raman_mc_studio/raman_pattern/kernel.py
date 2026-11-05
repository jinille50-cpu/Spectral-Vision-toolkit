# -*- coding: utf-8 -*-
"""PMCX 로 큐브 내부 플루언스 표를 만든다.

광원 위치 p 마다 한 번씩 실행하고 큐브 복셀의 플루언스만 저장 (npz 캐시).
여기광과 라만 방출광의 광학 계수는 같다고 가정 (상반성으로 방출 경로 = 센서 자리 광원 플루언스).
"""
import os
import time

import numpy as np

from geometry import Geometry


def run_fluence(geo: Geometry, p_mm: float, nphoton: int, seed: int = 1,
                src_radius_mm: float = 0.0):
    """광원 p_mm 에서 한 번 실행 → 큐브 내부 플루언스 (n, n, n)"""
    import pmcx
    vol = geo.build_volume()
    cfg = dict(
        nphoton=int(nphoton), vol=vol,
        tstart=0, tend=5e-9, tstep=5e-9,
        srcpos=geo.source_voxel(p_mm), srcdir=[0, 0, 1],
        prop=geo.prop(), issrcfrom0=1, isreflect=1,
        unitinmm=geo.voxel, autopilot=1, seed=int(seed), outputtype='fluence',
    )
    cfg.update(geo.source_cfg())
    if src_radius_mm > 0 and geo.source != 'cone':
        cfg['srctype'] = 'disk'
        cfg['srcparam1'] = [src_radius_mm / geo.voxel, 0, 0, 0]
    res = pmcx.run(cfg)
    F = res['flux'][..., 0]
    sx, sy, sz = geo.cube_slices()
    return np.array(F[sx, sy, sz], dtype=np.float32)


def fluence_table(geo: Geometry, positions, nphoton, cache_path=None, seed=1,
                  src_radius_mm=0.0, log=print):
    """positions (mm, 음수 가능) 마다 run_fluence → (K, n, n, n). cache_path 가 있으면 저장/재사용."""
    positions = np.asarray(positions, dtype=float)
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        if (z['positions'].shape == positions.shape and np.allclose(z['positions'], positions)
                and float(z['depth_top']) == geo.depth_top and int(z['nphoton']) == int(nphoton)):
            log(f"  cache hit: {os.path.basename(cache_path)}")
            return z['phi']
    phi = None
    t0 = time.time()
    for k, p in enumerate(positions):
        F = run_fluence(geo, p, nphoton, seed=seed + k, src_radius_mm=src_radius_mm)
        if phi is None:
            phi = np.zeros((len(positions),) + F.shape, dtype=np.float32)
        phi[k] = F
        if k % 10 == 0 or k == len(positions) - 1:
            log(f"  depth {geo.depth_top:.1f} mm  {k + 1}/{len(positions)}  "
                f"p = {p:.1f} mm  mean Φ = {F.mean():.3e}  ({time.time() - t0:.0f} s)")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, phi=phi, positions=positions, depth_top=geo.depth_top,
                            nphoton=nphoton, voxel=geo.voxel, cube_mm=geo.cube_mm,
                            tissue=np.array([geo.tissue[k] for k in ('mua', 'mus', 'g', 'n')]),
                            pet=np.array([geo.pet[k] for k in ('mua', 'mus', 'g', 'n')]))
    return phi

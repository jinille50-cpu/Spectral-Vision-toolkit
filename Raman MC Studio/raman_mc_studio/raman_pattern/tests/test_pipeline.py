# -*- coding: utf-8 -*-
"""config.json 파이프라인 검증: 설정 왕복, kernel_stack 이 run_stack 과 같은지, 기여율 합 = 1, 점 측정 파서."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RP = os.path.dirname(HERE)
sys.path.insert(0, RP)
import pipeline as pl  # noqa: E402
import point_data  # noqa: E402


def small_cfg(tmp_path):
    cfg = pl.load_config()
    cfg["geometry"].update(x_extent_mm=[-6.0, 6.0], y_extent_mm=[-4.0, 4.0], tissue_mm=6.0, voxel_mm=0.2)
    cfg["case1"]["depths_mm"] = [1, 2]
    cfg["case2"].update(laser_beside_mm=[0, 2], scan_mm=6.0, step_mm=0.3)
    cfg["ccd"].update(n_channels=40, channel_mm=0.2, laser_channel=8)
    cfg["fluence_cache"] = None
    cfg["output_dir"] = str(tmp_path)
    cfg["_path"] = str(tmp_path / "cfg.json")
    return cfg


def fake_fluence(geo):
    """해석적 가짜 플루언스: 광원에서 거리의 지수 감쇠 (PMCX 없이 커널 논리만 검사)"""
    x0, y0, z0 = geo.source_voxel(0.0)
    X, Y, Z = np.meshgrid(np.arange(geo.nx), np.arange(geo.ny), np.arange(geo.nz), indexing="ij")
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2) * geo.voxel
    return np.exp(-r / 1.5).astype(np.float32)


def test_config_roundtrip(tmp_path):
    cfg = small_cfg(tmp_path)
    p = pl.save_config(cfg)
    back = json.load(open(p, encoding="utf-8"))
    assert back["medium"]["mua"] == cfg["medium"]["mua"]
    assert "_path" not in back
    assert pl.fluence_tag(cfg) == pl.fluence_tag(pl.load_config(p))


def test_scan_positions_symmetric():
    pos = pl.scan_positions(20.0, 0.3)                   # 앱 기본값: 0 을 지나고 ±9.9 mm 대칭
    assert len(pos) == 67 and np.isclose(pos[33], 0.0) and np.isclose(pos[0], -9.9) and np.isclose(pos[-1], 9.9)
    assert np.allclose(pos, -pos[::-1])
    assert np.isclose(pl.scan_positions(20.0, 0.5)[-1], 10.0)     # 폭/2 가 간격의 배수면 끝점 포함


def test_kernel_stack_matches_run_stack(tmp_path):
    cfg = small_cfg(tmp_path)
    # 간격을 복셀(0.2)의 배수로: 0.3 이면 ±2.7 mm 같은 점이 복셀 경계 한가운데라 반올림 방향이 부동소수점 오차에 좌우됨
    cfg["case2"].update(scan_mm=5.6, step_mm=0.4)
    geo = pl.make_geometry(cfg)
    F = fake_fluence(geo)
    import run_stack
    pos_a, st_a = pl.kernel_stack(cfg, F, geo, 1.0)
    pos_b, st_b = run_stack.kernel_stack(F, geo, 1.0, cfg["case2"]["scan_mm"], cfg["case2"]["step_mm"])
    assert np.allclose(pos_a, pos_b)
    assert np.allclose(st_a, st_b)


def test_case1_case2_shapes_and_contribution(tmp_path):
    cfg = small_cfg(tmp_path)
    geo = pl.make_geometry(cfg)
    F = fake_fluence(geo)
    r1 = pl.case1(cfg, F, geo)
    r2 = pl.case2(cfg, F, geo)
    lc = cfg["ccd"]["laser_channel"] - 1
    for d in r1["depths"]:
        assert r1["direct_norm"][d][lc] == pytest.approx(1.0)
        assert np.all(np.diff(r1["direct_norm"][d][lc:]) <= 1e-9)        # 지수 감쇠 가짜 플루언스면 단조 감소
        assert np.allclose(r2["contribution"][d].sum(0), 1.0)
        assert r2["patterns"][d][2][lc] < r2["patterns"][d][0][lc]        # 옆 레이저는 레이저 채널에서 더 작음
    rows = [r for r in r2["summary"] if r["laser_beside_mm"] == 2]
    assert all(r["peak_offset_mm"] > 0 for r in rows)
    written = pl.save_results(cfg, r1, r2, pl.tissue_bg(cfg, F, geo))
    assert len(written) == 5 and all(os.path.exists(p) for p in written)


def test_case3_matches_make_paper_cases(tmp_path):
    cfg = small_cfg(tmp_path)
    geo = pl.make_geometry(cfg)
    F = fake_fluence(geo)
    r3 = pl.case3(cfg, F, geo)
    import make_paper_cases as mpc
    bins = [tuple(b) for b in cfg["case2"]["contribution_bins_mm"]]
    stacks = {}
    for d in r3["depths"]:
        pos, stacks[d] = pl.kernel_stack(cfg, F, geo, float(d))
    _, c3, cut = mpc.contributions(stacks, pos, bins, cfg)
    for d in r3["depths"]:
        assert np.allclose(r3["contribution"][d].sum(0), 1.0)
        assert np.allclose(r3["contribution"][d], c3[d])                  # 명령줄 스크립트의 Case 3 과 같은 값
        assert np.allclose(r3["outside_fraction"][d], cut[d])
    written = pl.save_results(cfg, r3=r3)
    assert len(written) == 1 and os.path.exists(written[0])


def test_parse_name():
    assert point_data.parse_name("3mm_10 binning_10s 2025 11월 20 155.csv") == (3.0, "2025-11-20", 155)
    assert point_data.parse_name("2mm_10binning_10s 2025 11월 18 32.csv") == (2.0, "2025-11-18", 32)
    assert point_data.parse_name("nodepth.csv")[0] is None


def test_point_data_synthetic(tmp_path):
    """합성 CSV 폴더: 채널 × 밴드 구조를 파서가 복원하고 batch 묶음이 맞는지"""
    nb, nch = 200, 10
    wl = np.linspace(860, 980, nb)
    shift = 1e7 / 830 - 1e7 / wl
    pet = np.exp(-((shift - 1615) / 6) ** 2) + 0.6 * np.exp(-((shift - 1725) / 8) ** 2)
    tis = np.exp(-((shift - 1440) / 12) ** 2) + 0.8 * np.exp(-((shift - 1655) / 20) ** 2)
    rng = np.random.default_rng(0)
    k = 0
    for depth, pet_amp in ((1, 1000.0), (6, 60.0)):
        for b in range(2):
            for f in range(3):
                Y = np.zeros((nch, nb))
                for c in range(nch):
                    decay = np.exp(-abs(c - 5) * 0.3)
                    Y[c] = 500 + pet_amp * decay * pet + 300 * tis * np.exp(-abs(c - 5) * 0.6) + rng.normal(0, 1, nb)
                arr = np.column_stack([np.tile(wl, nch), Y.ravel()])
                k += 1
                np.savetxt(tmp_path / f"{depth}mm_test 2025 11월 18 {k}.csv", arr, delimiter=",")
    cfg = pl.load_config()
    cfg["point_data"].update(folder=str(tmp_path), laser_channel=6, frames_per_batch=3, pet_ref_depth_mm=1, tissue_ref_depth_mm=6)
    cfg["ccd"]["channel_mm"] = 0.2
    shift2, groups = point_data.load_frames(str(tmp_path))
    assert len(groups) == 2 and all(len(v) == 6 for v in groups.values())
    assert np.allclose(shift2, shift)
    res = point_data.measured_profiles(cfg)
    assert set(res["pet_norm"]) == {1, 6}
    assert res["pet_norm"][1][0.0][0] == pytest.approx(1.0)
    assert res["pet_norm"][1][0.2][0] < 1.0

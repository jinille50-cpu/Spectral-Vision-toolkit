# -*- coding: utf-8 -*-
"""1포인트 SORS 점 측정 CSV 폴더 → PET 단독 진폭의 채널 프로파일 (깊이별 batch 평균 ± SD).

CSV 1개 = 1프레임: 열 0 파장(nm), 열 1 카운트, 행 = 채널 × 밴드 (채널 순, 파장이 다시 시작하면 다음 채널).
파일명에서 깊이("<d>mm") 와 날짜("<yyyy> <m>월 <d>") 를 읽고, 같은 (깊이, 날짜) 안에서 파일 번호 순으로
frames_per_batch 개씩 묶어 batch 로 본다 (batch 안에서는 프레임 중앙값).
처리: AsLS 배경 제거 → PET/조직 기준 스펙트럼(같은 데이터) → NNLS 분리 → PET 진폭.
"""
import glob
import os
import re

import numpy as np


def asls(y, lam=1e5, p=0.001, niter=10):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + lam * D @ D.T).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def parse_name(name):
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm", name)
    depth = float(m.group(1)) if m else None
    m = re.search(r"(\d{4})\s*(\d{1,2})월\s*(\d{1,2})", name)
    day = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else "all"
    m = re.search(r"(\d+)\.csv$", name)
    idx = int(m.group(1)) if m else 0
    return depth, day, idx


def load_frames(folder, laser_nm=830.0, log=None):
    """반환 shift(cm-1, nb), groups{(depth, day): [(idx, Y[ch, nb]), ...]}"""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"CSV 없음: {folder}")
    d0 = np.loadtxt(files[0], delimiter=",")
    wl = d0[:, 0]
    restarts = np.where(np.diff(wl) < 0)[0] + 1
    nb = int(restarts[0]) if len(restarts) else len(wl)
    nch = len(wl) // nb
    shift = 1e7 / laser_nm - 1e7 / wl[:nb]
    groups = {}
    for f in files:
        depth, day, idx = parse_name(os.path.basename(f))
        if depth is None:
            continue
        y = np.loadtxt(f, delimiter=",")[:, 1].reshape(nch, nb)
        groups.setdefault((depth, day), []).append((idx, y))
    if log:
        log(f"점 측정 로드: 파일 {len(files)}, 채널 {nch}, 밴드 {nb}, 그룹 {len(groups)}")
    return shift, groups


def corrected_batches(shift, groups, frames_per_batch, lam, p):
    """batch 마다 프레임 중앙값 → 채널별 AsLS. 반환 list of (depth, day, batch_idx, spectra[ch, nb])"""
    out = []
    for (depth, day), items in sorted(groups.items()):
        items = sorted(items)
        for b in range(len(items) // frames_per_batch):
            Y = np.stack([y for _, y in items[b * frames_per_batch:(b + 1) * frames_per_batch]]).astype(float)
            M = np.median(Y, 0)
            out.append((depth, day, b + 1, np.stack([M[c] - asls(M[c], lam, p) for c in range(M.shape[0])])))
    return out


def _local_fit(shift, y, ref, lo, hi):
    s = (shift >= lo) & (shift <= hi)
    A = np.stack([ref[s], np.ones(s.sum()), shift[s] - shift[s].mean()], 1)
    return np.linalg.lstsq(A, y[s], rcond=None)[0][0]


def measured_profiles(cfg, log=None):
    """cfg['point_data'] 설정으로 전체 처리. 반환 dict:
       pet_norm{depth: {offset_mm: (mean, sd)}} (레이저 채널 = 1),
       pet_total{depth: {offset_mm: (mean, sd)}} (채널 총합 = 1),
       tissue_norm(offset_mm, mean), batches[(depth, day, b)], offset_mm"""
    from scipy.optimize import nnls
    pd_ = cfg["point_data"]
    lc = int(pd_["laser_channel"]) - 1
    # 상대 경로는 이 파일(raman_pattern) 폴더 기준. 절대 경로면 join 이 그대로 돌려줌
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), pd_["folder"])
    shift, groups = load_frames(folder, pd_.get("laser_nm", 830.0), log)
    batches = corrected_batches(shift, groups, int(pd_["frames_per_batch"]), float(pd_["asls_lambda"]), float(pd_["asls_p"]))
    spectra = np.stack([b[3] for b in batches])
    meta = [(b[0], b[1], b[2]) for b in batches]
    depths = sorted({m[0] for m in meta})
    days = sorted({m[1] for m in meta})
    nch = spectra.shape[1]
    off = (np.arange(1, nch + 1) - lc - 1) * cfg["ccd"]["channel_mm"]

    def sel(depth, day=None):
        idx = [i for i, m in enumerate(meta) if m[0] == depth and (day is None or m[1] == day)]
        return spectra[idx]

    pet_d = float(pd_.get("pet_ref_depth_mm", depths[0]))
    tis_d = float(pd_.get("tissue_ref_depth_mm", depths[-1]))
    PET0 = sel(pet_d)[:, lc].mean(0)
    T = {}
    for day in days:
        y = sel(tis_d, day)
        if len(y) == 0:
            continue
        y = y[:, lc].mean(0)
        a = _local_fit(shift, y, PET0, 1590, 1632)
        T[day] = np.clip(y - a * PET0, 0, None)
    tkey = max(T, key=lambda k: T[k].max())
    g = _local_fit(shift, PET0, T[tkey], 1425, 1460)
    PET = np.clip(PET0 - g * T[tkey], 0, None)
    lo, hi = pd_["fit_range_cm1"]
    fit = (shift >= lo) & (shift <= hi)
    if pd_.get("exclude_cm1"):
        e0, e1 = pd_["exclude_cm1"]
        fit &= ~((shift >= e0) & (shift <= e1))
    A = np.stack([PET[fit]] + [T[k][fit] for k in sorted(T)], 1)
    amp = np.zeros((len(spectra), nch, A.shape[1]))
    for i in range(len(spectra)):
        for c in range(nch):
            amp[i, c] = nnls(A, spectra[i, c][fit])[0]
    pet = amp[..., 0]
    tis = amp[..., 1:].sum(-1)
    out = dict(offset_mm=off, batches=meta, pet_norm={}, pet_total={}, laser_ok={}, depths=depths)
    i16 = lc + int(round(1.6 / cfg["ccd"]["channel_mm"]))
    for d in depths:
        idx = [i for i, m in enumerate(meta) if m[0] == d]
        P = pet[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            Pn = P / P[:, lc:lc + 1]
        Pt = P / P[:, lc:].sum(1, keepdims=True)
        key = int(d) if float(d).is_integer() else d
        # 레이저 채널 정규화가 믿을 만한가: 모든 batch 에서 레이저 채널 PET > 0 이고 offset 1.6 mm 의 batch 간 편차가 평균의 절반 이하
        with np.errstate(invalid="ignore"):
            spread = np.nanstd(Pn[:, i16]) / np.nanmean(Pn[:, i16]) if i16 < nch else 0.0
        out["laser_ok"][key] = bool(np.all(P[:, lc] > 0) and np.isfinite(spread) and spread <= 0.5)
        out["pet_norm"][key] = {round(float(o), 1): (float(np.nanmean(Pn[:, c])), float(np.nanstd(Pn[:, c]))) for c, o in enumerate(off)}
        out["pet_total"][key] = {round(float(o), 1): (float(Pt[:, c].mean()), float(Pt[:, c].std())) for c, o in enumerate(off)}
        if log:
            log(f"측정 {key} mm: batch {len(idx)}, 레이저 채널 PET 진폭 중앙값 {np.median(P[:, lc]):.0f}, "
                f"offset 1.6 mm 정규화 {np.nanmean(Pn[:, i16]):.2f}, 레이저 채널 정규화 {'가능' if out['laser_ok'][key] else '불가(총합 정규화 사용)'}")
    with np.errstate(divide="ignore", invalid="ignore"):
        Tn = tis / tis[:, lc:lc + 1]
    out["tissue_norm"] = (off, np.nanmean(Tn, 0))
    return out

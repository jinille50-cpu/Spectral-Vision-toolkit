# -*- coding: utf-8 -*-
"""설정 파일 하나(config.json)로 돌아가는 라만 신호 패턴 파이프라인 (GUI 없음).

Case 1 (1D, z)      : 레이저가 PET 바로 위, 깊이별 CCD 채널 패턴 (+ 점 측정 비교)
Case 2 (2D, x-z)    : PET 중심을 지나는 x 라인 스캔. 레이저가 PET 옆 s mm 일 때 채널 패턴, |x| 구간별 기여율
Case 3 (3D, x-y-z)  : XY 전체 래스터 스캔. 레이저 위치 반경 r = sqrt(x^2 + y^2) 구간별 기여율
조직 배경 B(d)

사용:
    from raman_pattern import pipeline as pl
    cfg = pl.load_config()                 # raman_pattern/config.json
    F, geo = pl.get_fluence(cfg, log=print)
    r1 = pl.case1(cfg, F, geo)             # dict
    r2 = pl.case2(cfg, F, geo)
    r3 = pl.case3(cfg, F, geo)
    pl.save_results(cfg, r1, r2, bg=pl.tissue_bg(cfg, F, geo), r3=r3)
"""
import csv
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


def setup_font():
    """그림 한글 폰트 (앱 main 과 같은 규칙, 수학 기호는 DejaVu Sans 로 대체)"""
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if f in names:
            matplotlib.rcParams["font.family"] = [f, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


setup_font()


# ---------------------------------------------------------------- 설정
def load_config(path=None):
    path = path or CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_path"] = path
    return cfg


def save_config(cfg, path=None):
    path = path or cfg.get("_path") or CONFIG_PATH
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return path


def _abs(cfg, p):
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def channel_offsets(cfg):
    c = cfg["ccd"]
    return (np.arange(1, c["n_channels"] + 1) - c["laser_channel"]) * c["channel_mm"]


# ---------------------------------------------------------------- 기하와 플루언스
def make_geometry(cfg, depth_top=2.0):
    from geometry import Geometry
    g, m, s = cfg["geometry"], cfg["medium"], cfg["source"]
    return Geometry(voxel=g["voxel_mm"], x_extent=tuple(g["x_extent_mm"]), y_extent=tuple(g["y_extent_mm"]),
                    air_mm=g["air_mm"], tissue_mm=g["tissue_mm"], cube_mm=g["cube_mm"], depth_top=depth_top,
                    tissue=dict(mua=m["mua"], mus=m["mus"], g=m["g"], n=m["n"]),
                    source=s["type"], na=s["na"])


def fluence_tag(cfg):
    key = json.dumps({k: cfg[k] for k in ("medium", "geometry", "source")}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:10]


def fluence_cache_path(cfg):
    p = cfg.get("fluence_cache")
    if p:
        p = _abs(cfg, p)
        if os.path.exists(p):
            return p
    return os.path.join(HERE, "results", "cache", f"fluence_{fluence_tag(cfg)}.npy")


def get_fluence(cfg, log=None, force=False):
    """캐시가 있으면 읽고, 없으면 PMCX 로 1회 계산 후 저장. 반환 (F, geo)"""
    geo = make_geometry(cfg)
    path = fluence_cache_path(cfg)
    if os.path.exists(path) and not force:
        F = np.load(path)
        if F.shape == (geo.nx, geo.ny, geo.nz):
            if log:
                log(f"플루언스 캐시 사용: {path} {F.shape}")
            return F, geo
        if log:
            log(f"캐시 격자 {F.shape} 가 설정 {(geo.nx, geo.ny, geo.nz)} 과 달라 다시 계산")
        path = os.path.join(HERE, "results", "cache", f"fluence_{fluence_tag(cfg)}.npy")
    from run_stack import full_fluence
    n = int(cfg["source"]["nphoton"])
    if log:
        log(f"PMCX 플루언스 계산: 광자 {n:,}, 격자 {(geo.nx, geo.ny, geo.nz)} ...")
    F = full_fluence(geo, n, seed=int(cfg["source"].get("seed", 11)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, F)
    if log:
        log(f"저장: {path}")
    return F, geo


# ---------------------------------------------------------------- 감도 커널 (채널 설정을 config 에서 받음)
def _box_sum(P, n):
    c = np.cumsum(np.cumsum(np.cumsum(np.pad(P, 1)[:-1, :-1, :-1], 0), 1), 2)
    return (c[n:, n:, n:] - c[:-n, n:, n:] - c[n:, :-n, n:] - c[n:, n:, :-n]
            + c[:-n, :-n, n:] + c[:-n, n:, :-n] + c[n:, :-n, :-n] - c[:-n, :-n, :-n])


def scan_positions(scan_mm, step_mm):
    """레이저 스캔 위치 (PET 중심 기준, mm). 0 을 지나는 대칭 격자 k * step (|k * step| ≤ 폭/2)"""
    n = int(np.floor(scan_mm / 2 / step_mm + 1e-6))         # 1e-6: 10.0 이 9.9999… 로 계산돼 하나 빠지는 것 방지
    return np.arange(-n, n + 1) * step_mm


def kernel_stack(cfg, F, geo, depth_top):
    """stack[ch, iy, ix]: 레이저 raster 위치 (ix, iy) 에서 채널 ch 의 PET 라만 신호"""
    c2, ccd = cfg["case2"], cfg["ccd"]
    n = int(round(geo.cube_mm / geo.voxel))
    x0, y0, _ = geo.source_voxel(0.0)
    z0 = geo.z_to_vox(depth_top)
    Fz = F[:, :, z0:z0 + n]
    pos = scan_positions(c2["scan_mm"], c2["step_mm"])
    ix = x0 - n // 2 - np.round(pos / geo.voxel).astype(int)
    iy = y0 - n // 2 - np.round(pos / geo.voxel).astype(int)
    nch = ccd["n_channels"]
    stack = np.zeros((nch, len(pos), len(pos)), np.float64)
    for ch in range(nch):
        s = int(round((ch + 1 - ccd["laser_channel"]) * ccd["channel_mm"] / geo.voxel))
        Fs = np.zeros_like(Fz)
        if s >= 0:
            Fs[s:] = Fz[:Fz.shape[0] - s]
        else:
            Fs[:s] = Fz[-s:]
        S = _box_sum(Fz * Fs, n)[:, :, 0]
        stack[ch] = S[np.ix_(ix, iy)].T
    return pos, stack


def tissue_bg(cfg, F, geo):
    import background as bgmod
    offs = np.abs(channel_offsets(cfg))
    B = bgmod.tissue_background(F, geo, offs)
    return dict(offset_mm=channel_offsets(cfg), B=B / B[cfg["ccd"]["laser_channel"] - 1])


# ---------------------------------------------------------------- Case 1
def case1(cfg, F, geo, log=None, measured=None):
    """깊이별 직진 패턴. measured: point_data.measured_profiles() 결과 dict(선택).
    측정의 레이저 채널 값이 잡음(평균이 유한하지 않거나 SD/평균 > 0.5)이면 총합 정규화를 MC 총합에 맞춰 넣는다."""
    off = channel_offsets(cfg)
    lc = cfg["ccd"]["laser_channel"] - 1
    out = dict(offset_mm=off, depths=[], direct={}, direct_norm={}, measured={}, measured_note={})
    for d in cfg["case1"]["depths_mm"]:
        pos, st = kernel_stack(cfg, F, geo, float(d))
        cy = int(np.argmin(np.abs(pos)))
        direct = st[:, cy, cy]
        out["depths"].append(d)
        out["direct"][d] = direct
        out["direct_norm"][d] = direct / direct[lc]
        if log:
            sel = [lc + int(round(o / cfg["ccd"]["channel_mm"])) for o in (1.6, 3.2, 4.8, 6.4)]
            log(f"Case 1 깊이 {d} mm: offset 1.6/3.2/4.8/6.4 = " + " / ".join(f"{100 * out['direct_norm'][d][i]:.1f}%" for i in sel))
    if measured:
        for d in out["depths"]:
            pn = measured.get("pet_norm", {}).get(d)
            pt = measured.get("pet_total", {}).get(d)
            if not pn:
                continue
            if measured.get("laser_ok", {}).get(d, True):
                out["measured"][d] = pn
                out["measured_note"][d] = "normalized to laser channel"
            elif pt:
                mc_tot = out["direct"][d] / out["direct"][d][lc:].sum()
                k = 1.0 / mc_tot[lc]
                out["measured"][d] = {o: (m * k, s * k) for o, (m, s) in pt.items()}
                out["measured_note"][d] = "channel-sum normalized, scaled to MC sum (laser channel at noise level)"
                if log:
                    log(f"측정 {d} mm: 레이저 채널이 잡음이라 총합 정규화로 표시")
    return out


# ---------------------------------------------------------------- Case 2
def case2(cfg, F, geo, log=None):
    off = channel_offsets(cfg)
    lc = cfg["ccd"]["laser_channel"] - 1
    c2 = cfg["case2"]
    bins = [tuple(b) for b in c2["contribution_bins_mm"]]
    out = dict(offset_mm=off, depths=[], sides=c2["laser_beside_mm"], patterns={}, contribution={},
               ghost_fraction={}, summary=[], bins=bins, bin_labels=contribution_bin_labels(bins, "|x|"))
    for d in cfg["case1"]["depths_mm"]:
        pos, st = kernel_stack(cfg, F, geo, float(d))
        cy = int(np.argmin(np.abs(pos)))
        row = st[:, cy, :]                                   # [ch, laser x]
        ref = st[lc, cy, cy]
        pats = {}
        for s in c2["laser_beside_mm"]:
            j = int(np.argmin(np.abs(pos + s)))              # 레이저가 −s (offset 축이 PET 쪽)
            pats[s] = st[:, cy, j] / ref
            k = int(np.argmax(pats[s]))
            out["summary"].append(dict(depth_mm=d, laser_beside_mm=s, peak_pct=100 * pats[s][k], peak_channel=k + 1,
                                       peak_offset_mm=off[k], total_pct=100 * pats[s].sum() / pats[0].sum() if 0 in pats else np.nan,
                                       same_channel_ratio=pats[s][lc] / pats[0][lc] if 0 in pats else np.nan))
        ax_ = np.abs(pos)
        tot = row.sum(1)
        fr = np.stack([row[:, (ax_ >= lo) & (ax_ < hi)].sum(1) / tot for lo, hi in bins])
        far = np.abs(pos) > c2["ghost_outside_mm"]
        gf = row[:, far].sum(1) / tot
        out["depths"].append(d)
        out["patterns"][d] = pats
        out["contribution"][d] = fr
        out["ghost_fraction"][d] = gf
        if log:
            i16 = lc + int(round(1.6 / cfg["ccd"]["channel_mm"]))
            log(f"Case 2 깊이 {d} mm: 2 mm 옆 최대 {100 * pats.get(2, pats[max(pats)])[np.argmax(pats.get(2, pats[max(pats)]))]:.1f}%, "
                f"offset 1.6 mm 바로 위 몫 {100 * fr[0, i16]:.1f}%")
    return out


# ---------------------------------------------------------------- Case 3
CASE3_R_MAX = 9.0     # 반경 9 mm 원 안만 합산 (Case 2 라인 스캔 ±9 mm 와 같은 거리 범위, make_paper_cases.py 와 같음)


def contribution_bin_labels(bins, sym):
    """기여율 구간 범례 (영문). sym: Case 2 는 "|x|", Case 3 은 "r" """
    lab = []
    for lo, hi in bins:
        if lo == 0:
            lab.append(f"{sym} < {hi:g} mm (on top)")
        elif hi >= CASE3_R_MAX:
            lab.append(f"{sym} ≥ {lo:g} mm")
        else:
            lab.append(f"{lo:g} ≤ {sym} < {hi:g} mm")
    return lab


def case3(cfg, F, geo, log=None):
    """XY 전체 래스터 (kernel_stack 의 XY 전체) 에서 레이저-PET 거리 r 구간별 기여율.
    스캔 폭, 간격, 기여율 구간은 Case 2 설정을 x, y 양방향에 그대로 쓴다."""
    off = channel_offsets(cfg)
    lc = cfg["ccd"]["laser_channel"] - 1
    bins = [tuple(b) for b in cfg["case2"]["contribution_bins_mm"]]
    out = dict(offset_mm=off, depths=[], contribution={}, outside_fraction={}, bins=bins,
               bin_labels=contribution_bin_labels(bins, "r"))
    for d in cfg["case1"]["depths_mm"]:
        pos, st = kernel_stack(cfg, F, geo, float(d))          # st[ch, iy, ix]
        X, Y = np.meshgrid(pos, pos)
        R = np.hypot(X, Y)                                     # hypot = sqrt(X**2 + Y**2)
        inside = R <= CASE3_R_MAX + 1e-9
        tot = st[:, inside].sum(1)
        out["depths"].append(d)
        out["contribution"][d] = np.stack([st[:, inside & (R >= lo) & (R < hi)].sum(1) / tot for lo, hi in bins])
        out["outside_fraction"][d] = 1 - tot / st.reshape(st.shape[0], -1).sum(1)   # 원 밖(모서리) 몫, 참고용
        if log:
            i16 = lc + int(round(1.6 / cfg["ccd"]["channel_mm"]))
            log(f"Case 3 깊이 {d} mm: offset 1.6 mm 바로 위(r < {bins[0][1]:g} mm) 몫 {100 * out['contribution'][d][0, i16]:.1f}%")
    return out


# ---------------------------------------------------------------- 저장
def save_results(cfg, r1=None, r2=None, bg=None, out_dir=None, log=None, r3=None):
    out_dir = out_dir or _abs(cfg, cfg.get("output_dir", "results/app"))
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def w(name, header, rows):
        p = os.path.join(out_dir, name)
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            cw = csv.writer(f)
            cw.writerow(header)
            cw.writerows(rows)
        written.append(p)

    if r1:
        rows = []
        m = r1.get("measured") or {}
        for d in r1["depths"]:
            for c, o in enumerate(r1["offset_mm"]):
                mm = m.get(d, {}).get(round(float(o), 1), (np.nan, np.nan))
                rows.append([d, c + 1, f"{o:.1f}", f"{r1['direct_norm'][d][c]:.6g}", f"{r1['direct'][d][c]:.6g}",
                             "" if np.isnan(mm[0]) else f"{mm[0]:.6g}", "" if np.isnan(mm[1]) else f"{mm[1]:.6g}"])
        w("case1_direct_pattern.csv", ["depth_mm", "channel", "offset_mm", "MC_direct_norm", "MC_direct_raw",
                                       "measured_mean_norm", "measured_sd"], rows)
    if r2:
        rows = []
        for d in r2["depths"]:
            for s, p in r2["patterns"][d].items():
                for c, o in enumerate(r2["offset_mm"]):
                    rows.append([d, s, c + 1, f"{o:.1f}", f"{p[c]:.6g}"])
        w("case2_side_patterns.csv", ["depth_mm", "laser_beside_mm", "channel", "offset_mm", "signal_over_direct_laser_ch"], rows)
        w("case2_summary_table.csv", ["depth_mm", "laser_beside_mm", "peak_%", "peak_channel", "peak_offset_mm", "total_%", "same_channel_ratio"],
          [[r["depth_mm"], r["laser_beside_mm"], f"{r['peak_pct']:.1f}", r["peak_channel"], f"{r['peak_offset_mm']:.1f}",
            f"{r['total_pct']:.1f}", f"{r['same_channel_ratio']:.3f}"] for r in r2["summary"]])
        rows = []
        for d in r2["depths"]:
            for c, o in enumerate(r2["offset_mm"]):
                rows.append([d, c + 1, f"{o:.1f}"] + [f"{v:.4f}" for v in r2["contribution"][d][:, c]] + [f"{r2['ghost_fraction'][d][c]:.4f}"])
        w("case2_contribution_by_laser_position.csv",
          ["depth_mm", "channel", "offset_mm"] + [f"frac_{lo}_{hi}mm" for lo, hi in r2["bins"]] + ["ghost_fraction_outside"], rows)
    if r3:
        rows = []
        for d in r3["depths"]:
            for c, o in enumerate(r3["offset_mm"]):
                rows.append([d, c + 1, f"{o:.1f}"] + [f"{v:.4f}" for v in r3["contribution"][d][:, c]]
                            + [f"{r3['outside_fraction'][d][c]:.4f}"])
        w("case3_contribution_by_laser_radius.csv",
          ["depth_mm", "channel", "offset_mm"] + [f"frac_r{lo}_{hi}mm" for lo, hi in r3["bins"]]
          + [f"excluded_outside_r{CASE3_R_MAX:g}mm"], rows)
    if bg:
        w("tissue_background.csv", ["channel", "offset_mm", "B_norm"],
          [[c + 1, f"{o:.1f}", f"{v:.6g}"] for c, (o, v) in enumerate(zip(bg["offset_mm"], bg["B"]))])
    if log:
        for p in written:
            log(f"저장: {p}")
    return written


# ---------------------------------------------------------------- 그림 (matplotlib Figure 에 그림; GUI 와 파일 저장 공용)
def plot_case1(fig, r1, cfg, depth=None):
    fig.clear()
    depths = [depth] if depth else r1["depths"]
    ax = fig.add_subplot(111)
    from matplotlib import colormaps
    cm = colormaps["viridis"]
    for i, d in enumerate(depths):
        col = "#1565c0" if depth else cm(i / max(1, len(r1["depths"]) - 1))
        ax.semilogy(r1["offset_mm"], r1["direct_norm"][d], "-", color=col, lw=2, label=f"MC {d} mm")
        m = r1.get("measured", {}).get(d)
        if m:
            om = np.array(sorted(m))
            mu = np.array([m[o][0] for o in om])
            sd = np.array([m[o][1] for o in om])
            ok = np.isfinite(mu) & (mu > 0)
            if depth:
                note = r1.get("measured_note", {}).get(d, "")
                ax.errorbar(om[ok], mu[ok], yerr=np.minimum(sd[ok], mu[ok] * 0.95), fmt="o", color="k", ms=4, capsize=2,
                            label=f"measured {d} mm (batch mean ± SD; {note})" if note else f"measured {d} mm (mean ± SD)")
            else:
                ax.plot(om[ok], mu[ok], "o", color=col, ms=3.5)
    ax.axvline(0, color="red", ls=":", lw=1)
    ax.set_xlim(-1.4, 6.6)
    ax.set_ylim(5e-3, 2)
    ax.set_xlabel("Offset d (mm), laser position = 0")
    ax.set_ylabel("PET Raman signal / signal at laser channel")
    ax.set_title("Case 1 (1D): laser fixed above PET" + (f", PET depth {depth} mm" if depth else " (lines: MC, dots: measured)"))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    _plain_log_ticks(ax)


def plot_case2_patterns(fig, r2, depth):
    fig.clear()
    ax = fig.add_subplot(111)
    cols = ["#000000", "#1565c0", "#2e7d32", "#ef6c00", "#c62828", "#6a1b9a"]
    for i, (s, p) in enumerate(r2["patterns"][depth].items()):
        ax.semilogy(r2["offset_mm"], np.clip(p, 1e-4, None), "-", color=cols[i % len(cols)], lw=2,
                    label="laser above PET (direct)" if s == 0 else f"laser {s} mm beside PET")
    ax.axvline(0, color="red", ls=":", lw=1)
    ax.set_xlim(-1.4, 6.6)
    ax.set_ylim(1e-3, 2)
    ax.set_xlabel("Offset d (mm), laser position = 0")
    ax.set_ylabel("PET signal / direct laser channel")
    ax.set_title(f"Case 2 (2D, x–z): channel pattern by laser position, PET depth {depth} mm")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    _plain_log_ticks(ax)


def plot_case2_contribution(fig, r2, depth):
    fig.clear()
    ax = fig.add_subplot(111)
    cols = ["#000000", "#1565c0", "#2e7d32", "#ef6c00", "#c62828", "#6a1b9a"]
    ax.stackplot(r2["offset_mm"], r2["contribution"][depth], labels=r2["bin_labels"], colors=cols[:len(r2["bins"])], alpha=0.85)
    ax.set_xlim(0, r2["offset_mm"].max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Offset d (mm)")
    ax.set_ylabel("Fraction of channel signal")
    ax.set_title(f"Case 2 (2D, x–z): contribution by laser position, PET depth {depth} mm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", title="laser–PET distance", title_fontsize=8)


def plot_case3_contribution(fig, r3, depth):
    fig.clear()
    ax = fig.add_subplot(111)
    cols = ["#000000", "#1565c0", "#2e7d32", "#ef6c00", "#c62828", "#6a1b9a"]
    ax.stackplot(r3["offset_mm"], r3["contribution"][depth], labels=r3["bin_labels"], colors=cols[:len(r3["bins"])], alpha=0.85)
    ax.set_xlim(0, r3["offset_mm"].max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Offset d (mm)")
    ax.set_ylabel("Fraction of channel signal")
    ax.set_title(f"Case 3 (3D, x–y–z): contribution by laser position, PET depth {depth} mm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", title="laser–PET distance", title_fontsize=8)


def plot_background(fig, bg, measured_bg=None):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.semilogy(bg["offset_mm"], bg["B"], "-", color="#1565c0", lw=2, label="MC tissue background B(d)")
    if measured_bg is not None:
        om, mu = measured_bg
        ax.semilogy(om, mu, "ko", ms=3.5, label="measured tissue amplitude (mean)")
    ax.axvline(0, color="red", ls=":", lw=1)
    ax.set_xlim(-1.4, 6.6)
    ax.set_ylim(1e-3, 2)
    ax.set_xlabel("Offset d (mm)")
    ax.set_ylabel("Tissue signal / signal at laser channel")
    ax.set_title("Tissue Raman background vs offset")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    _plain_log_ticks(ax)


def _plain_log_ticks(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def save_figures(cfg, r1=None, r2=None, bg=None, out_dir=None, log=None, r3=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    out_dir = out_dir or _abs(cfg, cfg.get("output_dir", "results/app"))
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    def save(fig, name):
        p = os.path.join(out_dir, name)
        fig.savefig(p, dpi=170)
        paths.append(p)

    if r1:
        fig = Figure(figsize=(7.5, 5.2), constrained_layout=True)
        plot_case1(fig, r1, cfg)
        save(fig, "case1_direct_all_depths.png")
        for d in r1["depths"]:
            fig = Figure(figsize=(6.4, 4.6), constrained_layout=True)
            plot_case1(fig, r1, cfg, depth=d)
            save(fig, f"case1_direct_depth{d}mm.png")
    if r2:
        for d in r2["depths"]:
            fig = Figure(figsize=(6.4, 4.6), constrained_layout=True)
            plot_case2_patterns(fig, r2, d)
            save(fig, f"case2_side_patterns_depth{d}mm.png")
            fig = Figure(figsize=(6.4, 4.6), constrained_layout=True)
            plot_case2_contribution(fig, r2, d)
            save(fig, f"case2_contribution_depth{d}mm.png")
    if r3:
        for d in r3["depths"]:
            fig = Figure(figsize=(6.4, 4.6), constrained_layout=True)
            plot_case3_contribution(fig, r3, d)
            save(fig, f"case3_contribution_depth{d}mm.png")
    if bg:
        fig = Figure(figsize=(6.4, 4.6), constrained_layout=True)
        plot_background(fig, bg)
        save(fig, "tissue_background.png")
    if log:
        log(f"그림 {len(paths)}장 저장: {out_dir}")
    return paths


# ---------------------------------------------------------------- 명령줄
def main():
    import argparse
    ap = argparse.ArgumentParser(description="라만 신호 패턴 파이프라인 (config.json)")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--no-measured", action="store_true", help="점 측정 비교 생략")
    a = ap.parse_args()
    cfg = load_config(a.config)
    F, geo = get_fluence(cfg, log=print)
    measured = None
    if not a.no_measured and os.path.isdir(_abs(cfg, cfg["point_data"]["folder"])):
        import point_data
        measured = point_data.measured_profiles(cfg, log=print)
    r1 = case1(cfg, F, geo, log=print, measured=measured)
    r2 = case2(cfg, F, geo, log=print)
    r3 = case3(cfg, F, geo, log=print)
    bg = tissue_bg(cfg, F, geo)
    save_results(cfg, r1, r2, bg, log=print, r3=r3)
    save_figures(cfg, r1, r2, bg, log=print, r3=r3)


if __name__ == "__main__":
    main()

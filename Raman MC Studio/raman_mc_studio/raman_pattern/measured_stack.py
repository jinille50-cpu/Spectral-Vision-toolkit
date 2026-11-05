# -*- coding: utf-8 -*-
"""측정 Hcube (Y, X, 40 offset × 1310 band) 에서 offset 별 PET 점 세기(직진, 고스트)를 뽑아 MC 스택과 비교.

PET 피크: 1725 cm⁻¹ (C=O) = 밴드 1151 (cnn_depth.py 의 RGB_BANDS 와 동일), 중심 ±5 평균.
배경: 피크 양옆 (±40 밴드, ±5 평균) 직선 보간을 뺀 피크 높이.
픽셀 0.3 mm (50 픽셀 = 15 mm), 표면 offset = 8번 채널.

실행:  python measured_stack.py --data "C:/Users/User/Documents/Python Scripts/SORS 3D Analyzer" --depths 1 2 3 5
결과:  results/stack/measured_metrics.csv, fig13_measured_stack_<depth>mm.png, fig12_mc_vs_measured.png
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "stack")
N_CH, SURFACE_CH, PX_MM = 40, 8, 0.3
PEAK_BAND, HALF, SIDE = 1151, 5, 40


def load_peak_stack(path):
    """(Y, X, 40) PET 1725 피크 높이 이미지 스택"""
    import scipy.io as sio
    H = sio.loadmat(path)["Hcube"]
    Y, X, C = H.shape
    nb = C // N_CH
    H4 = H.reshape(Y, X, N_CH, nb)

    def band_mean(b):
        return H4[..., b - HALF:b + HALF + 1].mean(-1, dtype=np.float64)

    peak = band_mean(PEAK_BAND)
    base = 0.5 * (band_mean(PEAK_BAND - SIDE) + band_mean(PEAK_BAND + SIDE))
    # 조직 라만 배경의 채널별 세기: 조직 CH2 1440 cm-1 (밴드 879) 피크 높이의 XY 중앙값
    # (CCD 암전류 수준 ~1300 카운트가 있어 밴드 합은 쓸 수 없음. PET 픽셀은 5% 미만이라 중앙값에 영향 없음)
    tb = 879
    tissue = band_mean(tb) - 0.5 * (band_mean(tb - SIDE) + band_mean(tb + SIDE))
    bg = np.median(tissue, axis=(0, 1))
    return peak - base, bg


def spot_metrics(stack):
    """채널별 직진 점(표면 채널 최대 위치)과 고스트 점(같은 행, |dx| ≥ 4 px 의 최대) 세기와 이동량"""
    from scipy.ndimage import uniform_filter
    sm = np.stack([uniform_filter(stack[..., c], 3) for c in range(N_CH)], axis=-1)
    y0, x0 = np.unravel_index(np.argmax(sm[..., SURFACE_CH - 1]), sm.shape[:2])
    ref = sm[y0, x0, SURFACE_CH - 1]
    rows = []
    for c in range(N_CH):
        img = sm[..., c]
        direct = img[y0 - 1:y0 + 2, x0 - 1:x0 + 2].max()
        line = img[y0 - 1:y0 + 2, :].max(axis=0)
        dx = np.arange(img.shape[1]) - x0
        d_mm = (c + 1 - SURFACE_CH) * 0.2
        far = np.abs(dx) >= 4
        if abs(d_mm) >= 1.5 and far.any():
            j = int(np.where(far)[0][np.argmax(line[far])])
            ghost, shift = line[j], dx[j] * PX_MM
        else:
            ghost, shift = np.nan, np.nan
        rows.append(dict(ch=c + 1, offset_mm=(c + 1 - SURFACE_CH) * 0.2, direct=direct / ref, ghost=ghost / ref,
                         ghost_shift_mm=shift, total=img.clip(0).sum() / sm[..., SURFACE_CH - 1].clip(0).sum()))
    return rows, (y0, x0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:/Users/User/Documents/Python Scripts/SORS 3D Analyzer")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 5])
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meas = {}
    for d in a.depths:
        files = sorted(glob.glob(os.path.join(a.data, f"Hcube{d}mm_*.mat")))
        if not files:
            print(f"no Hcube{d}mm file"); continue
        stack, bg = load_peak_stack(files[0])
        rows, (y0, x0) = spot_metrics(stack)
        for r, b in zip(rows, bg):
            r["background"] = b / bg[SURFACE_CH - 1]
        meas[d] = rows
        print(f"{os.path.basename(files[0])}: direct spot at (y {y0}, x {x0}); "
              + ", ".join(f"ch{r['ch']}: d {r['direct']:.2f} g {r['ghost']:.2f} @{r['ghost_shift_mm']:+.1f}mm"
                          for r in [rows[7], rows[14], rows[19], rows[24], rows[29], rows[39]]))
        vmax = np.percentile(stack[..., SURFACE_CH - 1], 99.5)
        fig, axes = plt.subplots(4, 10, figsize=(20, 8.6), constrained_layout=True)
        for c, ax in enumerate(axes.ravel()):
            im = ax.imshow(stack[..., c] / vmax, cmap="jet", vmin=0, vmax=1, origin="upper")
            ax.set_title(f"Offset {c + 1}", fontsize=9, color="red" if c + 1 == SURFACE_CH else "black")
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="PET 1725 peak / surface-offset peak")
        fig.suptitle(f"Measured: {os.path.basename(files[0])} (PET top {d} mm), band {PEAK_BAND} ± {HALF}, linear baseline")
        fig.savefig(os.path.join(RESULTS, f"fig13_measured_stack_{d}mm.png"), dpi=110)
        plt.close(fig)

    with open(os.path.join(RESULTS, "measured_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["depth_mm", "channel", "offset_mm", "direct_peak", "ghost_peak", "ghost_over_direct",
                    "ghost_shift_mm", "image_total", "tissue_background"])
        for d, rows in meas.items():
            for r in rows:
                w.writerow([d, r["ch"], f"{r['offset_mm']:.1f}", f"{r['direct']:.4g}", f"{r['ghost']:.4g}",
                            f"{r['ghost'] / r['direct']:.4g}" if r["direct"] > 0 else "",
                            f"{r['ghost_shift_mm']:.2f}", f"{r['total']:.4g}", f"{r['background']:.4g}"])

    # MC 와 비교 (run_stack.py 결과가 있으면)
    mc_csv = os.path.join(RESULTS, "stack_metrics.csv")
    if not os.path.exists(mc_csv):
        print("no MC stack_metrics.csv yet; measured only"); return
    mc = {}
    for r in csv.DictReader(open(mc_csv, encoding="utf-8")):
        mc.setdefault(float(r["depth_top_mm"]), []).append(r)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    ks = sorted(meas)
    for i, d in enumerate(ks):
        c = cmap(i / max(1, len(ks) - 1))
        rows = meas[d]
        ch = [r["ch"] for r in rows]
        axes[0].semilogy(ch, [r["direct"] for r in rows], "o", color=c, ms=4, label=f"meas {d} mm")
        axes[1].semilogy(ch, [r["ghost"] for r in rows], "o", color=c, ms=4)
        axes[2].plot(ch, [abs(r["ghost_shift_mm"]) for r in rows], "o", color=c, ms=4)
        if float(d) in mc:
            m = mc[float(d)]
            mch = [int(x["channel"]) for x in m]
            axes[0].semilogy(mch, [float(x["direct_peak"]) for x in m], "-", color=c, lw=2, label=f"MC {d} mm")
            axes[1].semilogy(mch, [float(x["ghost_peak"]) if x["ghost_peak"] != "nan" else np.nan for x in m],
                             "-", color=c, lw=2)
            axes[2].plot(mch, [abs(float(x["ghost_pos_mm"])) for x in m], "-", color=c, lw=2)
    for ax in axes:
        ax.axvline(SURFACE_CH, color="red", ls=":", lw=1)
        ax.set_xlabel("Offset channel (8 = laser position)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Direct spot / surface-offset peak"); axes[0].set_title("Direct spot (laser above PET)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("Ghost spot / surface-offset peak"); axes[1].set_title("Ghost spot")
    axes[2].set_ylabel("|Ghost shift| (mm)"); axes[2].set_title("Ghost shift from PET")
    xs = np.arange(1, N_CH + 1)
    axes[2].plot(xs, np.abs(xs - SURFACE_CH) * 0.2, "k:", lw=1, label="= |offset|")
    axes[2].legend(fontsize=8)
    fig.suptitle("Measured (dots) vs Monte Carlo (lines): 2 mm PET cube in pork muscle")
    fig.savefig(os.path.join(RESULTS, "fig12_mc_vs_measured.png"), dpi=150)
    print("saved fig12")

    # fig14: 조직 배경의 채널 의존성 (측정 vs MC) 과 직진 점의 측정/MC 비율 → 공통 인자(광학계 감도) 여부
    sys.path.insert(0, HERE)
    import background as bgmod
    from geometry import Geometry
    geo = Geometry(x_extent=(-16.0, 16.0), y_extent=(-8.0, 8.0), tissue_mm=20.0, source="cone")
    cache = os.path.join(RESULTS, "fluence_bg_cone_8M.npy")
    F = np.load(cache) if os.path.exists(cache) else bgmod.full_fluence(geo, 8_000_000)
    if not os.path.exists(cache):
        np.save(cache, F)
    offs = np.abs((np.arange(1, N_CH + 1) - SURFACE_CH) * 0.2)
    Bmc = bgmod.tissue_background(F, geo, offs)
    Bmc = Bmc / Bmc[SURFACE_CH - 1]
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    xs = np.arange(1, N_CH + 1)
    for i, d in enumerate(ks):
        c = cmap(i / max(1, len(ks) - 1))
        b1.semilogy(xs, [r["background"] for r in meas[d]], "o", color=c, ms=4, label=f"meas {d} mm")
        if float(d) in mc:
            mcd = {int(x["channel"]): float(x["direct_peak"]) for x in mc[float(d)]}
            ratio = [r["direct"] / mcd[r["ch"]] if mcd.get(r["ch"], 0) > 0 else np.nan for r in meas[d]]
            b2.semilogy(xs, ratio, "o-", color=c, ms=4, lw=1, label=f"{d} mm")
    b1.semilogy(xs, Bmc, "k-", lw=2, label="MC tissue background")
    b1.set_ylabel("Tissue background / channel 8"); b1.set_title("Tissue background vs channel (image median)")
    b2.set_ylabel("Measured direct / MC direct"); b2.set_title("Direct spot: measured ÷ MC (common factor?)")
    for ax in (b1, b2):
        ax.axvline(SURFACE_CH, color="red", ls=":", lw=1); ax.set_xlabel("Offset channel (8 = laser position)")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.savefig(os.path.join(RESULTS, "fig14_background_and_ratio.png"), dpi=150)
    print("saved fig14; measured background at ch 1/8/15/20/30/40:",
          ", ".join(f"{meas[ks[-1]][i]['background']:.2f}" for i in (0, 7, 14, 19, 29, 39)),
          "| MC:", ", ".join(f"{Bmc[i]:.2f}" for i in (0, 7, 14, 19, 29, 39)))


if __name__ == "__main__":
    main()

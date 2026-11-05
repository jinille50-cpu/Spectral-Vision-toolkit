# -*- coding: utf-8 -*-
"""깊이별 플루언스 표 생성 → 라만 신호 패턴 행렬 → 그림/표 저장.

실행:  python run_all.py [--depths 1 2 3 4 5 6] [--nphoton 8000000] [--quick]
결과:  results/  (cache/*.npz, pattern_*.npz, *.csv, *.png)
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pattern  # noqa: E402
from geometry import Geometry  # noqa: E402
from kernel import fluence_table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LASERS = [0.0, 2.0, 4.0, 6.0, 8.0]          # 레이저 위치 (큐브 중심 기준, mm)
OFFSETS = np.round(np.arange(0, 8.01, 0.2), 1)   # CCD 40 채널 (0.2 mm)
POSITIONS = np.round(np.arange(0, 16.01, 0.2), 1)  # 플루언스 표 광원 위치


def compute(depth, nphoton, log=print):
    geo = Geometry(depth_top=depth)
    cache = os.path.join(RESULTS, "cache", f"phi_depth{depth:.1f}_n{int(nphoton)}.npz")
    phi = fluence_table(geo, POSITIONS, nphoton, cache_path=cache, log=log)
    toward = pattern.signal_matrix(phi, POSITIONS, LASERS, OFFSETS, side=-1)
    away = pattern.signal_matrix(phi, POSITIONS, LASERS, OFFSETS, side=+1)
    ref = toward[0, 0]
    return dict(depth=depth, toward=toward, away=away, ref=ref,
                toward_n=pattern.normalize(toward, ref), away_n=pattern.normalize(away, ref),
                ghost=pattern.ghost_metrics(toward, LASERS, OFFSETS),
                depth_prof_direct=pattern.depth_profile(phi, POSITIONS, 0.0, 0.0),
                depth_prof_ghost=pattern.depth_profile(phi, POSITIONS, 4.0, 4.0, side=-1))


def save_tables(res_list):
    with open(os.path.join(RESULTS, "ghost_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["depth_top_mm", "x_laser_mm", "peak_offset_mm", "peak_vs_direct_peak",
                    "row_total_vs_direct_total", "peak_norm_to_ref"])
        for r in res_list:
            for g in r["ghost"]:
                w.writerow([r["depth"], g["x_laser"], g["peak_offset"], f"{g['peak_vs_direct']:.4g}",
                            f"{g['total_vs_direct']:.4g}", f"{g['peak'] / r['ref']:.4g}"])
    for r in res_list:
        for side in ("toward", "away"):
            p = os.path.join(RESULTS, f"pattern_depth{r['depth']:.1f}_{side}.csv")
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["x_laser_mm \\ offset_mm"] + [f"{d:.1f}" for d in OFFSETS])
                for xl, row in zip(LASERS, r[side + "_n"]):
                    w.writerow([f"{xl:.1f}"] + [f"{v:.4g}" for v in row])
    np.savez_compressed(os.path.join(RESULTS, "patterns.npz"),
                        depths=[r["depth"] for r in res_list], lasers=LASERS, offsets=OFFSETS,
                        toward=np.stack([r["toward"] for r in res_list]),
                        away=np.stack([r["away"] for r in res_list]))


def make_figures(res_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    depths = [r["depth"] for r in res_list]
    cmap = plt.get_cmap("viridis")

    # 1. 깊이별 heatmap (센서가 큐브 쪽)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    vmin = max(1e-6, min(r["toward_n"][r["toward_n"] > 0].min() for r in res_list))
    for ax, r in zip(axes.ravel(), res_list):
        im = ax.imshow(np.clip(r["toward_n"], vmin, None), aspect="auto", origin="lower",
                       extent=[OFFSETS[0] - 0.1, OFFSETS[-1] + 0.1, -1, len(LASERS) * 2 - 1],
                       norm=LogNorm(vmin=vmin, vmax=1.0), cmap="magma")
        ax.set_yticks(LASERS)
        ax.set_xlabel("CCD offset from laser (mm)")
        ax.set_ylabel("Laser distance from PET (mm)")
        ax.set_title(f"PET top at {r['depth']:.0f} mm")
        for g in r["ghost"]:
            ax.plot(g["peak_offset"], g["x_laser"], "w+", ms=10, mew=1.5)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Signal / (laser 0 mm, offset 0)")
    fig.suptitle("Raman signal pattern of a 2 mm PET cube (sensors on the PET side); + = row maximum")
    fig.savefig(os.path.join(RESULTS, "fig1_pattern_heatmaps.png"), dpi=150)
    plt.close(fig)

    # 2. 깊이별 곡선: 레이저 위치마다 offset 프로파일 (실선 = 큐브 쪽, 점선 = 반대쪽)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True, sharey=True)
    for ax, r in zip(axes.ravel(), res_list):
        for i, xl in enumerate(LASERS):
            c = cmap(i / (len(LASERS) - 1))
            ax.semilogy(OFFSETS, r["toward_n"][i], "-", color=c, lw=2, label=f"laser {xl:.0f} mm")
            ax.semilogy(OFFSETS, r["away_n"][i], ":", color=c, lw=1.5)
        ax.set_title(f"PET top at {r['depth']:.0f} mm")
        ax.set_xlabel("CCD offset from laser (mm)")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("Signal / reference")
    axes[1, 0].set_ylabel("Signal / reference")
    axes[0, 0].legend(title="solid: sensor toward PET\ndotted: away from PET", fontsize=8)
    fig.savefig(os.path.join(RESULTS, "fig2_offset_profiles.png"), dpi=150)
    plt.close(fig)

    # 3. 요약: 고스트 최대값 / 직진 최대값 vs 레이저 거리, 깊이별
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for k, r in enumerate(res_list):
        c = cmap(k / max(1, len(res_list) - 1))
        xs = [g["x_laser"] for g in r["ghost"]]
        a1.semilogy(xs, [g["peak_vs_direct"] for g in r["ghost"]], "o-", color=c,
                    label=f"{r['depth']:.0f} mm")
        a2.semilogy(xs, [g["total_vs_direct"] for g in r["ghost"]], "o-", color=c)
    a1.set_xlabel("Laser distance from PET (mm)")
    a1.set_ylabel("Ghost peak / direct peak")
    a1.set_title("Peak signal when the laser is beside the PET")
    a1.legend(title="PET top depth")
    a1.grid(alpha=0.3, which="both")
    a2.set_xlabel("Laser distance from PET (mm)")
    a2.set_ylabel("Row total / direct row total")
    a2.set_title("Total signal over 40 offsets")
    a2.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(RESULTS, "fig3_ghost_vs_direct.png"), dpi=150)
    plt.close(fig)

    # 4. 직진 행(레이저 0 mm): 깊이별 offset 프로파일 + 큐브 안 깊이 기여
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for k, r in enumerate(res_list):
        c = cmap(k / max(1, len(res_list) - 1))
        a1.semilogy(OFFSETS, r["toward"][0] / res_list[0]["ref"], "-", color=c, lw=2,
                    label=f"{r['depth']:.0f} mm")
        zc = r["depth"] + (np.arange(len(r["depth_prof_direct"])) + 0.5) * 0.1
        a2.plot(zc, r["depth_prof_direct"], "-", color=c, lw=2)
        a2.plot(zc, r["depth_prof_ghost"], ":", color=c, lw=1.5)
    a1.set_xlabel("CCD offset from laser (mm)")
    a1.set_ylabel("Signal / (1 mm case, offset 0)")
    a1.set_title("Laser directly above PET: offset profile by depth")
    a1.legend(title="PET top depth")
    a1.grid(alpha=0.3, which="both")
    a2.set_xlabel("Depth inside tissue (mm)")
    a2.set_ylabel("Fraction of signal per 0.1 mm layer of the cube")
    a2.set_title("Where in the cube the signal is generated\n(solid: laser above, dotted: laser 4 mm beside, offset 4)")
    a2.grid(alpha=0.3)
    fig.savefig(os.path.join(RESULTS, "fig4_direct_row_and_depth_origin.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--nphoton", type=int, default=8_000_000)
    ap.add_argument("--quick", action="store_true", help="1e6 photons, depth 2 only (pipeline check)")
    a = ap.parse_args()
    if a.quick:
        a.depths, a.nphoton = [2.0], 1_000_000
    os.makedirs(RESULTS, exist_ok=True)
    log_path = os.path.join(RESULTS, "run.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    res_list = [compute(d, a.nphoton, log) for d in a.depths]
    save_tables(res_list)
    make_figures(res_list)
    log("ghost summary (toward side):")
    for r in res_list:
        for g in r["ghost"]:
            log(f"  depth {r['depth']:.0f} mm  laser {g['x_laser']:.0f} mm → peak at offset "
                f"{g['peak_offset']:.1f} mm, peak/direct = {g['peak_vs_direct']:.3g}, "
                f"total/direct = {g['total_vs_direct']:.3g}")
    log(f"done → {RESULTS}")


if __name__ == "__main__":
    main()

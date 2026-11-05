# -*- coding: utf-8 -*-
"""지방 줄기 비대칭 실험 + 스캔 라인 이미지.

근육 슬래브의 +x 쪽에만 무작위 지방 원기둥(y 방향)을 넣고, 레이저를 +x(지방 쪽)와 −x(근육만) 양쪽으로
옮기며 PET 신호 패턴을 비교한다. 광원은 NA 0.23 원뿔 (2" f = 100 mm achromat, 초점 100 µm).

실행:  python run_fat.py [--depths 1 2 4 6] [--nphoton 8000000] [--quick]
결과:  results/fat/
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pattern  # noqa: E402
from geometry import FatStrand, Geometry  # noqa: E402
from kernel import fluence_table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "fat")
LASERS = [0.0, 2.0, 4.0, 6.0, 8.0]
OFFSETS = np.round(np.arange(0, 8.01, 0.2), 1)
POSITIONS = np.round(np.arange(-16.0, 16.01, 0.2), 1)        # 비대칭 매질 → 양쪽 다 저장
SCAN = np.round(np.arange(-8.0, 8.01, 0.2), 1)              # 스캔 라인 이미지의 레이저 위치


def make_geometry(depth):
    return Geometry(x_extent=(-24.0, 24.0), depth_top=depth, fat_strand=FatStrand(), source='cone')


def compute(depth, nphoton, log=print):
    geo = make_geometry(depth)
    cache = os.path.join(RESULTS, "cache", f"phi_fat_cone_depth{depth:.1f}_n{int(nphoton)}.npz")
    phi = fluence_table(geo, POSITIONS, nphoton, cache_path=cache, log=log)
    # 레이저가 지방 쪽(+x)에 있고 센서가 PET 쪽(−x 방향)으로 뻗음 → 빛이 지방을 지나 PET 에 닿음
    fat_side = pattern.signal_matrix(phi, POSITIONS, LASERS, OFFSETS, side=-1)
    # 레이저가 근육 쪽(−x)에 있고 센서가 PET 쪽(+x 방향) → 지방 없는 기준
    muscle_side = pattern.signal_matrix(phi, POSITIONS, [-x for x in LASERS], OFFSETS, side=+1)
    ref = muscle_side[0, 0]
    # 스캔 라인 이미지: CCD 방향 고정. toward −x: I(x_L, d) = S(x_L, x_L − d); toward +x: S(x_L, x_L + d)
    img_minus = np.array([[pattern.signal(phi, POSITIONS, xl, d, side=-1) for xl in SCAN] for d in OFFSETS])
    img_plus = np.array([[pattern.signal(phi, POSITIONS, xl, d, side=+1) for xl in SCAN] for d in OFFSETS])
    return dict(depth=depth, geo=geo, fat=fat_side / ref, muscle=muscle_side / ref, ref=ref,
                ghost_fat=pattern.ghost_metrics(fat_side, LASERS, OFFSETS),
                ghost_muscle=pattern.ghost_metrics(muscle_side, LASERS, OFFSETS),
                img_minus=img_minus / ref, img_plus=img_plus / ref)


def save_tables(res_list):
    with open(os.path.join(RESULTS, "fat_vs_muscle_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["depth_top_mm", "x_laser_mm", "ghost_peak_offset_muscle", "ghost_peak_offset_fat",
                    "peak_vs_direct_muscle", "peak_vs_direct_fat", "fat_over_muscle_peak",
                    "fat_over_muscle_total"])
        for r in res_list:
            for gm, gf in zip(r["ghost_muscle"], r["ghost_fat"]):
                w.writerow([r["depth"], gm["x_laser"], gm["peak_offset"], gf["peak_offset"],
                            f"{gm['peak_vs_direct']:.4g}", f"{gf['peak_vs_direct']:.4g}",
                            f"{gf['peak'] / gm['peak']:.4g}", f"{gf['total'] / gm['total']:.4g}"])
    np.savez_compressed(os.path.join(RESULTS, "fat_patterns.npz"),
                        depths=[r["depth"] for r in res_list], lasers=LASERS, offsets=OFFSETS, scan=SCAN,
                        fat=np.stack([r["fat"] for r in res_list]),
                        muscle=np.stack([r["muscle"] for r in res_list]),
                        img_minus=np.stack([r["img_minus"] for r in res_list]),
                        img_plus=np.stack([r["img_plus"] for r in res_list]))


def make_figures(res_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    cmap = plt.get_cmap("viridis")
    n = len(res_list)

    # fig6: 기하 (x-z 단면 y = 0, −4, +4 mm 와 x-y 위에서 본 지방)
    geo = res_list[0]["geo"]
    vol = geo.build_volume()
    ext_xz = [geo.x_extent[0], geo.x_extent[1], geo.air_mm + geo.tissue_mm - geo.air_mm, -geo.air_mm]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    for ax, y in zip(axes[:3], (-4.0, 0.0, 4.0)):
        ax.imshow(vol[:, geo.y_to_vox(y), :].T, origin="upper", aspect="equal", cmap="tab10", vmin=0, vmax=9,
                  extent=[geo.x_extent[0], geo.x_extent[1], geo.tissue_mm, -geo.air_mm])
        ax.set_xlim(-10, 10)
        ax.set_title(f"x-z section at y = {y:+.0f} mm")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("depth (mm)")
    top = vol.max(axis=2).T
    axes[3].imshow(top, origin="lower", aspect="equal", cmap="tab10", vmin=0, vmax=9,
                   extent=[geo.x_extent[0], geo.x_extent[1], geo.y_extent[0], geo.y_extent[1]])
    axes[3].set_xlim(-10, 10)
    axes[3].set_title("top view (max label along z)")
    axes[3].set_xlabel("x (mm)")
    axes[3].set_ylabel("y (mm)")
    fig.suptitle(f"Geometry: PET cube (label 2) at x = 0, fat strand (label 3) on +x side, "
                 f"PET top at {res_list[0]['depth']:.0f} mm; laser moves along x")
    fig.savefig(os.path.join(RESULTS, "fig6_geometry.png"), dpi=130)
    plt.close(fig)

    # fig7: 깊이별 offset 프로파일, 지방 쪽(실선) vs 근육 쪽(점선)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, res_list):
        for i, xl in enumerate(LASERS):
            c = cmap(i / (len(LASERS) - 1))
            ax.semilogy(OFFSETS, r["fat"][i], "-", color=c, lw=2, label=f"laser {xl:.0f} mm")
            ax.semilogy(OFFSETS, r["muscle"][i], ":", color=c, lw=1.5)
        ax.set_title(f"PET top at {r['depth']:.0f} mm")
        ax.set_xlabel("CCD offset from laser (mm)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Signal / (muscle side, laser 0, offset 0)")
    axes[0].legend(title="solid: laser on fat side\ndotted: laser on muscle side", fontsize=8)
    fig.savefig(os.path.join(RESULTS, "fig7_fat_vs_muscle_profiles.png"), dpi=150)
    plt.close(fig)

    # fig8: 스캔 라인 이미지 (행 = offset, 열 = 레이저 위치), CCD 방향 두 가지
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7.5), constrained_layout=True)
    axes = axes.reshape(2, n)
    vmin = 1e-4
    for j, r in enumerate(res_list):
        for i, (key, title) in enumerate((("img_minus", "CCD toward −x (over the fat → PET)"),
                                          ("img_plus", "CCD toward +x (over PET → fat)"))):
            ax = axes[i, j]
            im = ax.imshow(np.clip(r[key], vmin, None), aspect="auto", origin="lower",
                           extent=[SCAN[0] - 0.1, SCAN[-1] + 0.1, OFFSETS[0] - 0.1, OFFSETS[-1] + 0.1],
                           norm=LogNorm(vmin=vmin, vmax=1.0), cmap="magma")
            ax.axvline(0, color="w", lw=0.8, ls="--")
            ax.axvspan(2.25, 3.75, color="cyan", alpha=0.15)
            ax.set_xlabel("Laser position x_L (mm)   [PET at 0, fat around +3]")
            ax.set_ylabel("CCD offset (mm)")
            ax.set_title(f"PET top {r['depth']:.0f} mm — {title}", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Signal / reference")
    fig.suptitle("Scan-line image of the PET cube: direct trace at x_L = 0, ghost trace along x_L = ∓offset")
    fig.savefig(os.path.join(RESULTS, "fig8_scanline_images.png"), dpi=150)
    plt.close(fig)

    # fig9: 비대칭 요약
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for k, r in enumerate(res_list):
        c = cmap(k / max(1, n - 1))
        xs = LASERS
        a1.plot(xs, [gf["peak"] / gm["peak"] for gf, gm in zip(r["ghost_fat"], r["ghost_muscle"])],
                "o-", color=c, label=f"{r['depth']:.0f} mm")
        a2.plot(xs, [gf["peak_offset"] - gm["peak_offset"] for gf, gm in zip(r["ghost_fat"], r["ghost_muscle"])],
                "o-", color=c)
    a1.axhline(1, color="gray", lw=1)
    a1.set_xlabel("Laser distance from PET (mm)")
    a1.set_ylabel("Ghost peak: fat side / muscle side")
    a1.set_title("Ghost strength change caused by the fat strand")
    a1.legend(title="PET top depth")
    a1.grid(alpha=0.3)
    a2.axhline(0, color="gray", lw=1)
    a2.set_xlabel("Laser distance from PET (mm)")
    a2.set_ylabel("Ghost peak offset shift, fat − muscle (mm)")
    a2.set_title("Ghost position change caused by the fat strand")
    a2.grid(alpha=0.3)
    fig.savefig(os.path.join(RESULTS, "fig9_fat_asymmetry.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=float, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--nphoton", type=int, default=8_000_000)
    ap.add_argument("--quick", action="store_true")
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
    log("fat strand summary (ghost peak, fat side / muscle side; peak offset fat vs muscle):")
    for r in res_list:
        for gf, gm in zip(r["ghost_fat"], r["ghost_muscle"]):
            log(f"  depth {r['depth']:.0f} mm  laser {gf['x_laser']:.0f} mm → fat/muscle peak "
                f"{gf['peak'] / gm['peak']:.3f}, peak offset {gf['peak_offset']:.1f} vs {gm['peak_offset']:.1f} mm")
    log(f"done → {RESULTS}")


if __name__ == "__main__":
    main()

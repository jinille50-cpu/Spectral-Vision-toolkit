# -*- coding: utf-8 -*-
"""조직 라만 배경 B(d) 계산 → PET 신호 / 조직 배경 비율 그림.

run_all.py 가 만든 results/patterns.npz 가 있어야 한다.
실행:  python run_background.py [--nphoton 8000000] [--split 0.5]
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import background  # noqa: E402
from geometry import Geometry  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nphoton", type=int, default=8_000_000)
    ap.add_argument("--split", type=float, default=0.5, help="표면층 두께 (mm)")
    a = ap.parse_args()

    pat = np.load(os.path.join(RESULTS, "patterns.npz"))
    depths, lasers, offsets = pat["depths"], pat["lasers"], pat["offsets"]
    toward = pat["toward"]                                  # (n_depth, n_laser, n_offset)

    geo = Geometry()
    cache = os.path.join(RESULTS, "cache", f"fluence_full_n{a.nphoton}.npy")
    if os.path.exists(cache):
        F = np.load(cache)
    else:
        F = background.full_fluence(geo, a.nphoton)
        np.save(cache, F)
    sym = background.check_translation_symmetry(F, geo, 4.0)
    print(f"x-symmetry check at ±4 mm: relative difference {sym:.3%}")

    B, Bsh, Bdeep = background.tissue_background(F, geo, offsets, depth_split=a.split)
    ratio = toward / B[None, None, :]                       # PET / tissue (단위 라만 효율 동일 가정)

    with open(os.path.join(RESULTS, "background_ratio.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["offset_mm", "tissue_B_norm", f"shallow_fraction_{a.split}mm"]
                   + [f"PET/tissue_depth{d:.0f}_laser{l:.0f}" for d in depths for l in lasers])
        for j, d in enumerate(offsets):
            w.writerow([f"{d:.1f}", f"{B[j] / B[0]:.4g}", f"{Bsh[j] / B[j]:.4g}"]
                       + [f"{ratio[i, k, j]:.4g}" for i in range(len(depths)) for k in range(len(lasers))])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.semilogy(offsets, B / B[0], "k-", lw=2, label="tissue background B(d)")
    ax.semilogy(offsets, Bsh / B[0], "k--", lw=1.5, label=f"from top {a.split} mm")
    ax.semilogy(offsets, Bdeep / B[0], "k:", lw=1.5, label=f"from below {a.split} mm")
    ax.set_xlabel("CCD offset from laser (mm)")
    ax.set_ylabel("Background / B(0)")
    ax.set_title("Tissue Raman background vs offset")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for i, d in enumerate(depths):
        ax.semilogy(offsets, ratio[i, 0], "-", color=cmap(i / max(1, len(depths) - 1)), lw=2,
                    label=f"PET top {d:.0f} mm")
    ax.set_xlabel("CCD offset from laser (mm)")
    ax.set_ylabel("PET signal / tissue background")
    ax.set_title("Laser directly above PET (direct)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    i2 = int(np.argmin(np.abs(depths - 2.0)))
    for k, l in enumerate(lasers):
        ax.semilogy(offsets, ratio[i2, k], "-", color=cmap(k / max(1, len(lasers) - 1)), lw=2,
                    label=f"laser {l:.0f} mm beside")
    ax.set_xlabel("CCD offset from laser (mm)")
    ax.set_ylabel("PET signal / tissue background")
    ax.set_title(f"PET top at {depths[i2]:.0f} mm: direct vs ghost rows")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(RESULTS, "fig5_pet_over_tissue_background.png"), dpi=150)
    print("saved fig5 and background_ratio.csv")
    print("background shallow fraction by offset (0,2,4,6,8 mm):",
          ", ".join(f"{Bsh[j] / B[j]:.2f}" for j in [0, 10, 20, 30, 40]))


if __name__ == "__main__":
    main()

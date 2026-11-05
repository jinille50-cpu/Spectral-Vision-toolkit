# -*- coding: utf-8 -*-
"""측정과 같은 형태의 offset 별 XY 이미지 스택 (50 × 50 raster × offset 40) 을 MC 로 합성.

측정 구성: CCD 공간축 40채널(0.2 mm), 8번 채널이 레이저 자리(표면 offset). 채널 k 의 offset = (k − 8) × 0.2 mm.
라만 신호(r_L, d) = Σ_큐브 Φ0(v − r_L) · Φ0(v − r_L − d·x̂)  (균질 매질의 평행이동 대칭 + 상반성)
→ 곱 필드 P_d(u) = Φ0(u) Φ0(u − d x̂) 를 offset 마다 만들고, 큐브 크기 창의 합(박스 필터)으로 모든 레이저 위치를 한 번에 계산.

실행:  python run_stack.py [--depths 1 2 5] [--nphoton 20000000] [--step 0.3] [--n 50]
결과:  results/stack/
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import Geometry  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "stack")
N_CH, SURFACE_CH, CH_MM = 40, 8, 0.2


def full_fluence(geo: Geometry, nphoton: int, seed: int = 11):
    import pmcx
    cfg = dict(nphoton=int(nphoton), vol=geo.build_volume(with_cube=False),
               tstart=0, tend=1e-8, tstep=1e-8, srcpos=geo.source_voxel(0.0), srcdir=[0, 0, 1],
               prop=geo.prop(), issrcfrom0=1, isreflect=1, unitinmm=geo.voxel, autopilot=1,
               seed=int(seed), outputtype='fluence', **geo.source_cfg())
    return np.array(pmcx.run(cfg)['flux'][..., 0], dtype=np.float32)


def box_sum(P, n):
    """3D 누적합으로 n×n×n 창 합. 반환 S[i, j, k] = Σ P[i:i+n, j:j+n, k:k+n]"""
    c = np.pad(P, ((1, 0), (1, 0), (1, 0))).cumsum(0).cumsum(1).cumsum(2)
    return (c[n:, n:, n:] - c[:-n, n:, n:] - c[n:, :-n, n:] - c[n:, n:, :-n]
            + c[:-n, :-n, n:] + c[:-n, n:, :-n] + c[n:, :-n, :-n] - c[:-n, :-n, :-n])


def kernel_stack(F, geo: Geometry, depth_top, scan_mm, step_mm):
    """
    F: 광원이 (x0, y0) 표면에 있을 때의 전체 플루언스. 반환 stack[ch, iy, ix] (라만 신호, 레이저 raster 위치 기준).
    레이저 위치 r_L 에서 큐브(윗면 depth_top, 중심 (0, 0))의 복셀 v 에 대해 Φ0(v − r_L): 광원을 원점에 둔 F 를
    큐브 쪽으로 −r_L 만큼 민 것 = F 에서 큐브 창을 +r_L 만큼 옮긴 것과 같다.
    """
    n = int(round(geo.cube_mm / geo.voxel))
    x0, y0, _ = geo.source_voxel(0.0)
    z0 = geo.z_to_vox(depth_top)
    Fz = F[:, :, z0:z0 + n]                                   # 큐브 깊이 층만 (nx, ny, n)
    half = scan_mm / 2
    pos = np.arange(-half, half + 1e-9, step_mm)
    # 레이저가 r_L 에 있으면 Φ0(v − r_L): 원점 광원 F 에서 큐브 창을 −r_L 만큼 옮긴 것 → 창 시작 = 원점 − r_L − n/2
    ix = x0 - n // 2 - np.round(pos / geo.voxel).astype(int)
    iy = y0 - n // 2 - np.round(pos / geo.voxel).astype(int)
    stack = np.zeros((N_CH, len(pos), len(pos)), np.float64)
    for ch in range(N_CH):
        s = int(round((ch + 1 - SURFACE_CH) * CH_MM / geo.voxel))     # 센서 위치 = 레이저 + s 복셀
        # P(u) = Φ0(u) Φ0(u − s x̂): 센서 자리 광원의 플루언스는 F 를 +s 만큼 민 것
        Fs = np.zeros_like(Fz)
        if s >= 0:
            Fs[s:] = Fz[:Fz.shape[0] - s]
        else:
            Fs[:s] = Fz[-s:]
        P = Fz * Fs
        S = box_sum(P, n)[:, :, 0]                                  # z 창은 큐브 두께 전체 하나
        stack[ch] = S[np.ix_(ix, iy)].T                             # [iy, ix]
    return pos, stack


def spot_metrics(stack, pos, depth_top):
    """채널마다 직진 점(레이저가 큐브 위, 중심)과 고스트 점(x 방향으로 밀린 최대) 세기와 위치"""
    rows = []
    cy = int(np.argmin(np.abs(pos)))
    ref = stack[SURFACE_CH - 1, cy, cy]
    for ch in range(N_CH):
        line = stack[ch, cy]                                        # y = 0 줄, x 방향
        direct = line[cy]
        d_mm = (ch + 1 - SURFACE_CH) * CH_MM
        # 고스트: 레이저가 −d 쪽에 있을 때 (센서가 큐브 위). 직진 점에서 1.5 mm 이상 떨어진 쪽의 최대
        far = np.abs(pos + d_mm) <= 1.0 if abs(d_mm) > 1.5 else np.zeros_like(pos, bool)
        if far.any():
            j = np.where(far)[0][np.argmax(line[far])]
            ghost, gpos = line[j], pos[j]
        else:
            ghost, gpos = np.nan, np.nan
        rows.append(dict(ch=ch + 1, offset_mm=d_mm, direct=direct / ref, ghost=ghost / ref,
                         ghost_pos_mm=gpos, line_max=line.max() / ref, total=stack[ch].sum() / stack[SURFACE_CH - 1].sum()))
    return rows


def make_figures(res_list, pos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for r in res_list:
        stack = r["stack"]
        vmax = stack[SURFACE_CH - 1].max()
        fig, axes = plt.subplots(4, 10, figsize=(20, 8.6), constrained_layout=True)
        for ch, ax in enumerate(axes.ravel()):
            im = ax.imshow(stack[ch] / vmax, cmap="jet", vmin=0, vmax=1, origin="lower",
                           extent=[pos[0], pos[-1], pos[0], pos[-1]])
            ax.set_title(f"Offset {ch + 1}", fontsize=9, color="red" if ch + 1 == SURFACE_CH else "black")
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Signal / surface-offset peak")
        fig.suptitle(f"MC Raman image stack: 2 mm PET cube, top at {r['depth']:.0f} mm, "
                     f"laser raster {len(pos)}×{len(pos)} ({pos[1] - pos[0]:.1f} mm step), "
                     f"offset channel 8 = laser position; channel k offset = (k − 8) × 0.2 mm")
        fig.savefig(os.path.join(RESULTS, f"fig10_stack_depth{r['depth']:.0f}mm.png"), dpi=110)
        plt.close(fig)
        # 같은 스택을 로그 스케일로 (약한 고스트도 보이게)
        fig, axes = plt.subplots(4, 10, figsize=(20, 8.6), constrained_layout=True)
        for ch, ax in enumerate(axes.ravel()):
            im = ax.imshow(np.log10(np.clip(stack[ch] / vmax, 1e-3, 1)), cmap="jet", vmin=-3, vmax=0,
                           origin="lower", extent=[pos[0], pos[-1], pos[0], pos[-1]])
            ax.set_title(f"Offset {ch + 1}", fontsize=9, color="red" if ch + 1 == SURFACE_CH else "black")
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="log10(Signal / surface-offset peak)")
        fig.suptitle(f"Same stack, log scale — PET top at {r['depth']:.0f} mm")
        fig.savefig(os.path.join(RESULTS, f"fig10_stack_depth{r['depth']:.0f}mm_log.png"), dpi=110)
        plt.close(fig)

    # 채널별 직진/고스트 곡선
    cmap = plt.get_cmap("viridis")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    for k, r in enumerate(res_list):
        c = cmap(k / max(1, len(res_list) - 1))
        m = r["metrics"]
        ch = [x["ch"] for x in m]
        a1.semilogy(ch, [x["direct"] for x in m], "-", color=c, lw=2, label=f"PET top {r['depth']:.0f} mm")
        a1.semilogy(ch, [x["ghost"] for x in m], "--", color=c, lw=2)
        a2.plot(ch, [x["ghost"] / x["direct"] if x["direct"] > 0 else np.nan for x in m], "-", color=c, lw=2)
        a3.plot(ch, [-x["ghost_pos_mm"] for x in m], "-", color=c, lw=2)
    for a in (a1, a2, a3):
        a.axvline(SURFACE_CH, color="red", ls=":", lw=1)
        a.set_xlabel("Offset channel (8 = laser position)")
        a.grid(alpha=0.3, which="both")
    a1.set_ylabel("Spot peak / surface-offset peak")
    a1.set_title("Direct spot (solid) and ghost spot (dashed)")
    a1.legend(fontsize=8)
    a2.set_ylabel("Ghost / direct (same channel)")
    a2.set_title("Ghost relative to direct in the same image")
    a2.set_ylim(0, 1.5)
    a3.set_ylabel("Ghost shift from PET (mm)")
    a3.set_title("Ghost position vs channel")
    xs = np.arange(1, N_CH + 1)
    a3.plot(xs, (xs - SURFACE_CH) * CH_MM, "k:", lw=1, label="= offset")
    a3.legend(fontsize=8)
    fig.savefig(os.path.join(RESULTS, "fig11_stack_metrics.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=float, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--nphoton", type=int, default=20_000_000)
    ap.add_argument("--step", type=float, default=0.3, help="raster 간격 (mm)")
    ap.add_argument("--n", type=int, default=50, help="raster 픽셀 수 (한 변)")
    ap.add_argument("--source", default="cone", choices=["cone", "pencil"])
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    scan_mm = (a.n - 1) * a.step
    margin = scan_mm / 2 + 2.0 + N_CH * CH_MM + 6.0
    geo = Geometry(x_extent=(-margin, margin), y_extent=(-scan_mm / 2 - 6.0, scan_mm / 2 + 6.0),
                   tissue_mm=20.0, source=a.source)
    cache = os.path.join(RESULTS, f"fluence_full_{a.source}_n{a.nphoton}_{geo.nx}x{geo.ny}x{geo.nz}.npy")
    if os.path.exists(cache):
        F = np.load(cache)
    else:
        print(f"MC run: grid {geo.nx}×{geo.ny}×{geo.nz}, {a.nphoton:,} photons", flush=True)
        F = full_fluence(geo, a.nphoton)
        np.save(cache, F)

    res_list = []
    for d in a.depths:
        pos, stack = kernel_stack(F, geo, d, scan_mm, a.step)
        res_list.append(dict(depth=d, stack=stack, metrics=spot_metrics(stack, pos, d)))
        print(f"depth {d:.0f} mm: stack {stack.shape} done", flush=True)
    np.savez_compressed(os.path.join(RESULTS, "stacks.npz"), pos=pos,
                        depths=[r["depth"] for r in res_list], stacks=np.stack([r["stack"] for r in res_list]))
    with open(os.path.join(RESULTS, "stack_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["depth_top_mm", "channel", "offset_mm", "direct_peak", "ghost_peak", "ghost_over_direct",
                    "ghost_pos_mm", "image_total"])
        for r in res_list:
            for m in r["metrics"]:
                w.writerow([r["depth"], m["ch"], f"{m['offset_mm']:.1f}", f"{m['direct']:.4g}",
                            f"{m['ghost']:.4g}", f"{m['ghost'] / m['direct']:.4g}" if m["direct"] > 0 else "",
                            f"{m['ghost_pos_mm']:.2f}", f"{m['total']:.4g}"])
    make_figures(res_list, pos)
    for r in res_list:
        m = r["metrics"]
        pick = [m[i] for i in (0, 7, 14, 19, 24, 29, 34, 39)]
        print(f"depth {r['depth']:.0f} mm  ch(direct%, ghost%): " +
              ", ".join(f"{x['ch']}({x['direct'] * 100:.0f}, {x['ghost'] * 100:.0f})" for x in pick))
    print("done →", RESULTS)


if __name__ == "__main__":
    main()

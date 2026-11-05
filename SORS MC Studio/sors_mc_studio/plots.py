"""MC 결과 그래프 (어두운 테마). 계산은 core.analysis 그대로, 여기서는 그리기만."""
from __future__ import annotations

import numpy as np

from . import theme
from .core import analysis as A
from .widgets import MplCanvas, outside_legend

SIDE_SETS = {'single': {'R': ('R',)}, 'symmetric': {'S': ('R', 'L')},
             'both': {'L': ('L',), 'R': ('R',)}}
SIDE_TAG = {'L': ' (−x)', 'R': '', 'S': ''}


def _integer_offset_targets(max_offset):
    t = [0.1] + list(range(2, int(np.floor(max_offset)) + 1, 2))
    return [x for x in t if x <= max_offset]


def _layer_bands(ax, layers, x0, x1):
    """배경에 층 띠와 경계선 (깊이 축이 아래로 증가하는 그래프용)"""
    z = 0.0
    for i, l in enumerate(layers):
        ax.axhspan(z, z + l.thickness, color=theme.layer_color(i), alpha=0.07, lw=0, zorder=0)
        z += l.thickness
        ax.axhline(z, color=theme.layer_color(i), lw=0.8, ls=(0, (4, 4)), alpha=0.5, zorder=1)
        ax.text(x1, z, f"{l.name} 바닥 {z:g} mm ", color=theme.layer_color(i), fontsize=8,
                ha='right', va='bottom', alpha=0.9, zorder=2, clip_on=True)
    ax.axhline(0, color=theme.MUTED, lw=1.2, zorder=1)


class BananaCanvas(MplCanvas):
    """offset 별 주 경로: 검출 광자 산란점 깊이 중앙값 선 + 25~75 백분위 밴드"""

    def plot(self, result, ctx):
        mode, max_offset = ctx['mode'], ctx['max_offset']
        side_sets = SIDE_SETS[mode]
        targets = [t for t in _integer_offset_targets(max_offset) if t >= 1.0]
        data, rows = [], []
        for side, sides in side_sets.items():
            sgn = -1.0 if side == 'L' else 1.0
            for i, t in enumerate(targets):
                xc, (med, p25, p75), n_ph = A.banana_main_path(result, sides, float(t))
                rows += [[side, t, sgn * x, a, b, c, n_ph] for x, a, b, c in zip(xc, med, p25, p75)]
                if np.isfinite(med).sum() >= 3:
                    m, off, _ = A.side_select(result, mode, side, ctx['strip_width'])
                    m &= np.abs(off - t) <= 0.5
                    zmed = float(A._wpercentile(result.detected_max_depth[m],
                                                result.detected_weight[m], 0.5))
                    data.append(dict(offset=sgn * t, x=sgn * xc, z=med, p25=p25, p75=p75,
                                     depth=zmed, n=n_ph, side=side, ti=i))
        self.export_tables = {'main_path': (['side', 'offset_mm', 'x_mm', 'median_depth_mm',
                                             'p25_depth_mm', 'p75_depth_mm', 'n_photons'], rows)}
        if not data:
            self.message("띠 검출기에 들어온 광자가 너무 적습니다.\n광자 수나 검출 띠 폭을 늘려 다시 실행하세요.")
            return
        colors = theme.offset_colors(len(targets))
        xmax = max(abs(b['offset']) for b in data) * 1.12
        xlo = -xmax if mode == 'both' else -0.6
        with self.style():
            self.reset_axes()
            ax = self.ax
            _layer_bands(ax, ctx['layers'], xlo, xmax)
            for b in data:
                ok = np.isfinite(b['z'])
                c = colors[b['ti']]
                ax.fill_between(b['x'][ok], b['p25'][ok], b['p75'][ok], color=c, alpha=0.16, lw=0)
                ax.plot(b['x'][ok], b['z'][ok], '-', color=c, lw=2.6,
                        label=(f"{abs(b['offset']):.0f} mm{SIDE_TAG[b['side']]}  "
                               f"z_max 중앙값 {b['depth']:.2f} mm  N={b['n']:,}"))
                ax.plot([b['offset']], [0], marker='v', ms=9, color=c, mec=theme.BG0, zorder=12)
            ax.plot([0], [-0.05], marker='v', ms=13, color=theme.SOURCE_COLOR, mec=theme.BG0,
                    zorder=15, label='광원')
            # 깊이 축: 밴드 아래 끝 + 여유, 매질 두께보다 깊게는 보이지 않게
            zmax = max(np.nanmax(b['p75']) for b in data)
            zlim = min(ctx['total'], max(zmax * 1.3, 0.5))
            ax.set_xlim(xlo, xmax)
            ax.set_ylim(zlim, -0.35)
            ax.set_xlabel("Lateral x (mm)")
            ax.set_ylabel("Depth z (mm)")
            ax.set_title("주 경로: 산란점 깊이 중앙값과 25–75 백분위")
            outside_legend(ax, self.show_legend)
        self.draw_idle()


class AngleCanvas(MplCanvas):
    """깊이별 평균 |x| 로 구한 평균 전파 각도"""

    last_angles = None

    def plot(self, result, ctx):
        mode, max_offset, T = ctx['mode'], ctx['max_offset'], ctx['total']
        angles, curves, curve_rows, fit_rows = {}, {}, [], []
        for side, sides in SIDE_SETS[mode].items():
            zc, mx, valid, slope, angle = A.propagation_curve(result, sides, T)
            curve_rows += [[side, z, m] for z, m in zip(zc[valid], mx[valid])]
            if valid.sum() >= 3:
                angles[side] = angle
                curves[side] = (zc[valid], mx[valid])
                fit_rows.append([side, slope, angle])
        self.export_tables = {'mean_lateral_vs_depth': (['side', 'depth_mm', 'mean_abs_x_mm'], curve_rows),
                              'fit': (['side', 'slope_x_per_z', 'angle_deg'], fit_rows)}
        self.last_angles = angles
        if not angles:
            self.message("전파 각도를 계산할 검출 광자가 부족합니다.")
            return
        with self.style():
            self.reset_axes()
            ax = self.ax
            _layer_bands(ax, ctx['layers'], -max_offset, max_offset)
            for ref in (15, 30, 45, 60, 75):
                r = np.radians(ref)
                xe, ze = (T * np.tan(r), T) if T * np.tan(r) < max_offset else (max_offset, max_offset / np.tan(r))
                for s in (1, -1):
                    ax.plot([0, s * xe], [0, ze], ':', color=theme.MUTED, lw=0.9, alpha=0.6, zorder=2)
                ax.text(xe, ze, f" {ref}°", color=theme.MUTED, fontsize=8, ha='left', va='top', zorder=3)
            for side, angle in angles.items():
                col = theme.SIDE_COLORS[side]
                zc, mx = curves[side]
                r = np.radians(angle)
                xe, ze = (T * np.tan(r), T) if T * np.tan(r) < max_offset else (max_offset, max_offset / np.tan(r))
                signs = (-1,) if side == 'L' else ((1, -1) if side == 'S' else (1,))
                for k, s in enumerate(signs):
                    ax.plot(s * mx, zc, 'o', ms=3, color=col, alpha=0.55, zorder=8)
                    ax.plot([0, s * xe], [0, ze], '-', color=col, lw=2.6, zorder=10,
                            label=f"평균 전파 각도{SIDE_TAG[side] if mode == 'both' else ''}  {angle:.1f}°"
                            if k == 0 else None)
            ax.plot([0], [-0.05], marker='v', ms=13, color=theme.SOURCE_COLOR, mec=theme.BG0,
                    zorder=15, label='광원')
            ax.set_xlim(-max_offset, max_offset)
            ax.set_ylim(T, -0.4)
            ax.set_xlabel("Lateral x (mm)")
            ax.set_ylabel("Depth z (mm)")
            ax.set_title("깊이별 평균 |x| (점)와 원점 통과 직선 (각도 = 깊이 방향 기준)")
            outside_legend(ax, self.show_legend)
        self.draw_idle()


class OffsetCanvas(MplCanvas):
    """0.2 mm 채널별 검출 강도, 그룹 평균 ± SD"""

    def plot(self, result, ctx):
        mode, G = ctx['mode'], ctx['n_groups']
        centers, gp = A.grouped_profile(result, mode, ctx['strip_width'], ctx['max_offset'], G)
        rows = []
        with self.style():
            self.reset_axes()
            ax = self.ax
            for side, p in gp.items():
                m, sd = A.mean_sd(p['intensity'])
                rows += [[side, c, a, b] for c, a, b in zip(centers, m, sd)]
                xs = -centers if side == 'L' else centers
                col = theme.SIDE_COLORS[side]
                tag = {'R': '+x', 'L': '−x', 'S': '|x| 양쪽 평균'}[side]
                if np.isfinite(sd).any():
                    ax.fill_between(xs, m - sd, m + sd, color=col, alpha=0.22, lw=0)
                ax.plot(xs, m, '-o', color=col, lw=2, ms=3.5, label=f"{tag}  (± SD, 그룹 {G}개)")
                if mode != 'both' and np.nanmax(m) > 0:
                    k = int(np.nanargmax(m))
                    ax.axvline(centers[k], color=theme.WARN, ls=':', lw=1.2,
                               label=f"최대 {centers[k]:.1f} mm")
            ax.set_xlabel("Offset along detector strip (mm)")
            ax.set_ylabel("Detected weight per launched photon")
            ax.set_title("검출 띠 0.2 mm 채널별 신호")
            outside_legend(ax, self.show_legend)
        self.export_tables = {'offset_profile': (['side', 'offset_mm', 'intensity_per_photon',
                                                  'intensity_sd'], rows)}
        self.draw_idle()


class DepthCanvas(MplCanvas):
    """(좌) offset 별 z_max 분포, (우) 깊이 구간별 직진/측면 비율"""

    def __init__(self):
        super().__init__(ncols=2)
        self.last_table_text = ""

    def plot(self, result, ctx, central_radius=1.0):
        mode, max_offset, T = ctx['mode'], ctx['max_offset'], ctx['total']
        m, exit_x, _ = A.side_select(result, mode, 'R' if mode == 'single' else 'S', ctx['strip_width'])
        exit_x, z_max = exit_x[m], result.detected_max_depth[m]
        self.export_tables = {}
        if z_max.size < 20:
            self.message(f"띠 검출 광자가 {z_max.size}개뿐입니다. 광자 수를 늘려 다시 실행하세요.")
            return
        rows, data, pos, means = [], [], [], []
        for t in [t for t in [0.1, 1, 2, 3, 4, 5, 6, 7, 8] if t <= max_offset]:
            d = z_max[np.abs(exit_x - t) <= 0.5]
            if d.size >= 8:
                dv = d if d.size <= 3000 else d[np.linspace(0, d.size - 1, 3000).astype(int)]
                data.append(dv); pos.append(t); means.append(d.mean())
                rows.append((t, d.mean(), np.median(d), d.std(), float(np.mean(d >= 0.5 * T)), d.size))
        edges = np.arange(0, T + 1e-6, 1.0)
        cen, sid, lbl, ratio_rows = [], [], [], []
        for d0, d1 in zip(edges[:-1], edges[1:]):
            mm = (z_max >= d0) & (z_max < d1)
            n = int(mm.sum())
            if n >= 8:
                c = float(np.mean(np.abs(exit_x[mm]) < central_radius))
                cen.append(c * 100); sid.append((1 - c) * 100); lbl.append(f"{d0:.0f}–{d1:.0f}")
                ratio_rows.append((d0, d1, c * 100, (1 - c) * 100, n))
        with self.style():
            self.reset_axes()
            ax1, ax2 = self.axes
            if data:
                vp = ax1.violinplot(data, positions=pos, widths=0.6, showmeans=False, showextrema=False)
                for body in vp['bodies']:
                    body.set_facecolor(theme.ACCENT); body.set_edgecolor(theme.ACCENT); body.set_alpha(0.28)
                ax1.plot(pos, means, '-o', color=theme.TEXT, lw=1.4, ms=4, label='평균 z_max')
                outside_legend(ax1, self.show_legend)
            ax1.set_xlabel("Offset (mm)")
            ax1.set_ylabel("Probed depth z_max (mm)")
            ax1.set_title("offset별 도달 깊이 분포" + ("  (−x/+x 합침)" if mode == 'both' else ""))
            ax1.invert_yaxis()
            if cen:
                xp = np.arange(len(lbl))
                ax2.bar(xp, cen, color=theme.ACCENT, alpha=0.85, label=f'직진 (|x| < {central_radius:g} mm)')
                ax2.bar(xp, sid, bottom=cen, color='#6aa3ff', alpha=0.55, label='측면')
                ax2.set_xticks(xp); ax2.set_xticklabels(lbl)
                outside_legend(ax2, self.show_legend)
            ax2.set_xlabel("Probed depth bin (mm)")
            ax2.set_ylabel("Fraction of returning photons (%)")
            ax2.set_title("깊이 구간별 직진 대 측면")
            ax2.set_ylim(0, 100)
            ax2.grid(True, axis='y'); ax2.grid(False, axis='x')
        lines = ["[offset별 z_max]  offset  mean  median  std  frac(z>=T/2)  N"]
        lines += [f"  {r[0]:5.1f}  {r[1]:5.2f}  {r[2]:5.2f}  {r[3]:4.2f}  {r[4]:5.2f}  {r[5]}" for r in rows]
        lines += ["[깊이 구간별 직진/측면]  depth  direct%  side%  N"]
        lines += [f"  {r[0]:.0f}-{r[1]:.0f}  {r[2]:6.1f}  {r[3]:6.1f}  {r[4]}" for r in ratio_rows]
        self.last_table_text = "\n".join(lines)
        self.export_tables = {
            'depth_per_offset': (['offset_mm', 'mean_depth_mm', 'median_depth_mm', 'std_mm',
                                  'frac_depth_ge_half_T', 'n_photons'], rows),
            'direct_vs_side': (['depth_from_mm', 'depth_to_mm', 'direct_pct', 'side_pct', 'n_photons'],
                               ratio_rows)}
        self.draw_idle()

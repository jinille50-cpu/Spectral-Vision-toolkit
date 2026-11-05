"""
라인스캔 검출 분석 + 논문용 정량 지표
======================================
검출기: 광원 중심을 지나는 x축(offset 축) 위의 띠, |y| <= strip_width/2.
offset 채널 간격 bin_width (기본 0.2 mm).

검출 방향 mode:
  'single'    : +x 쪽만 (offset = x)
  'both'      : -x / +x 를 따로 (side 'L', 'R', offset = |x|)
  'symmetric' : |x| 로 합침, 강도는 양쪽 평균 (offset = |x|)

불확도: 광자 index % N_BASE_GROUPS(20) 로 기본 그룹을 저장해 두고, 분석 그룹 수 G (20 의
        약수) 로 묶어 그룹별 지표의 평균 ± 표준편차.
        (각 그룹은 N/G 광자짜리 독립 시뮬레이션과 같음)

광자 궤적은 저장하지 않음. 시뮬레이션 중 띠 검출 광자의 산란점을 PathHist 에 누적하고,
Banana 주 경로 / 전파 각도 / 깊이별 FWHM 은 이 히스토그램에서 계산.
"""

import numpy as np

MODES = ('single', 'both', 'symmetric')
N_BASE_GROUPS = 20                  # 저장 단위 그룹 수. 분석 그룹 수는 이 값의 약수
GROUP_CHOICES = (1, 2, 4, 5, 10, 20)
CH_WIDTH = 0.2                      # offset 채널 간격 (mm)
_SIDE_IDX = {'R': 0, 'L': 1}


class PathHist:
    """
    띠 검출기(|y_exit| <= strip_width/2, |x_exit| < max_offset)에 검출된 광자의 산란점
    (x, z) 누적 히스토그램.
      H[기본 그룹, side, offset 채널, z, x]   side 0 = +x 쪽, 1 = -x 쪽 (x 를 뒤집어 +x 기준)
    가중치 = 검출 가중치 / μt(z): 산란점 하나가 평균 1/μt 길이의 경로를 대표하므로
    층마다 μs 가 달라도 '경로 길이' 밀도가 됨.
    칸 크기: z, x 모두 0.1 mm (축당 100 칸을 넘으면 그만큼 넓힘).
    """

    def __init__(self, layers, max_offset, strip_width):
        T = sum(l.thickness for l in layers)
        self.strip_width = float(strip_width)
        self.max_offset = float(max_offset)
        self.n_ch = int(np.ceil(max_offset / CH_WIDTH - 1e-9))
        self.hz = max(0.1, T / 100)
        self.hx = max(0.1, (max_offset + 2.0) / 100)
        nz = int(np.ceil(T / self.hz - 1e-9))
        nx = int(np.ceil((max_offset + 2.0) / self.hx - 1e-9))
        self.z_edges = np.arange(nz + 1) * self.hz
        self.x_edges = -1.0 + np.arange(nx + 1) * self.hx      # [-1, max_offset + 1]
        self._bounds = np.cumsum([l.thickness for l in layers])
        self._mut = np.array([l.mua + l.mus for l in layers])
        self.H = np.zeros((N_BASE_GROUPS, 2, self.n_ch, nz, nx), np.float32)

    @property
    def z_centers(self):
        return (self.z_edges[:-1] + self.z_edges[1:]) / 2

    @property
    def x_centers(self):
        return (self.x_edges[:-1] + self.x_edges[1:]) / 2

    def photon_bins(self, exit_x, exit_y):
        """광자별 (side, 채널). 띠 밖이거나 max_offset 밖이면 채널 = -1"""
        side = (exit_x < 0).astype(np.int64)
        ch = np.floor(np.abs(exit_x) / CH_WIDTH).astype(np.int64)
        ok = (np.abs(exit_y) <= self.strip_width / 2) & (ch < self.n_ch)
        return side, np.where(ok, ch, -1)

    def add(self, group, side, ch, x, z, w):
        """산란점 단위 배열 누적 (x 는 원래 부호, w = 광자의 검출 가중치)"""
        nz, nx = self.H.shape[3], self.H.shape[4]
        xm = np.where(side == 1, -x, x)
        iz = np.floor(z / self.hz).astype(np.int64)
        ix = np.floor((xm - self.x_edges[0]) / self.hx).astype(np.int64)
        ok = (ch >= 0) & (iz >= 0) & (iz < nz) & (ix >= 0) & (ix < nx)
        if not ok.any():
            return
        zl = z[ok]
        ww = w[ok] / self._mut[np.clip(np.searchsorted(self._bounds, zl), 0, len(self._mut) - 1)]
        flat = ((((group[ok].astype(np.int64) * 2 + side[ok]) * self.n_ch + ch[ok]) * nz
                 + iz[ok]) * nx + ix[ok])
        self.H += np.bincount(flat, ww, self.H.size).reshape(self.H.shape).astype(np.float32)

    def window(self, sides, offset, halfwidth):
        """sides ⊂ {'R','L'}, offset ± halfwidth 채널 합 → (기본 그룹, nz, nx)"""
        lo = max(0, int(round((offset - halfwidth) / CH_WIDTH)))
        hi = min(self.n_ch, int(round((offset + halfwidth) / CH_WIDTH)))
        si = [_SIDE_IDX[s] for s in sides]
        return self.H[:, si, lo:hi].sum(axis=(1, 2), dtype=np.float64)


def _group_of(base_group, n_groups):
    """기본 그룹(0..19) → 분석 그룹 (G 는 20 의 약수)"""
    G = max(1, int(n_groups))
    if N_BASE_GROUPS % G:
        raise ValueError(f"그룹 수는 {GROUP_CHOICES} 중 하나여야 합니다.")
    return base_group % G


def _combine_groups(Hb, n_groups):
    """(기본 그룹, ...) → (G, ...)"""
    G = max(1, int(n_groups))
    return np.stack([Hb[_group_of(np.arange(N_BASE_GROUPS), G) == g].sum(axis=0)
                     for g in range(G)])


def _hist_percentiles(p, edges, qs):
    """히스토그램 p (칸 경계 edges) 의 백분위 (칸 내부 선형)"""
    tot = p.sum()
    if tot <= 0:
        return [np.nan] * len(qs)
    cdf = np.concatenate(([0.0], np.cumsum(p) / tot))
    return [float(np.interp(q, cdf, edges)) for q in qs]


def _sides(mode):
    return ('L', 'R') if mode == 'both' else (('S',) if mode == 'symmetric' else ('R',))


def side_select(result, mode, side, strip_width):
    """해당 side 의 검출 광자 mask, offset 값, 강도 scale"""
    x = result.detected_x
    strip = np.abs(result.detected_y) <= strip_width / 2
    if side == 'R':
        return strip & (x >= 0), x, 1.0
    if side == 'L':
        return strip & (x < 0), -x, 1.0
    return strip, np.abs(x), 0.5          # symmetric: 양쪽 합 / 2


def _wpercentile(v, w, q):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    return np.interp(np.asarray(q) * w.sum(), c, v)


def offset_profile(result, mode, strip_width, max_offset, n_photons,
                   bin_width=0.2, sel_extra=None):
    """
    side 별 offset 프로파일.
    반환: centers, {side: dict(intensity, mean_depth, median_depth, p10, p90,
                               mean_pathlen, mean_nscat, count)}
    intensity = 검출 가중치 합 / 입사 광자 수
    """
    edges = np.arange(0, max_offset + 1e-9, bin_width)
    centers = (edges[:-1] + edges[1:]) / 2
    nb = len(centers)
    out = {}
    for side in _sides(mode):
        m, off, scale = side_select(result, mode, side, strip_width)
        if sel_extra is not None:
            m = m & sel_extra
        off = off[m]
        w = result.detected_weight[m]
        d = result.detected_max_depth[m]
        pl = result.detected_pathlen[m]
        ns = result.detected_nscat[m].astype(float)
        idx = np.digitize(off, edges) - 1
        ok = (idx >= 0) & (idx < nb)
        idx, w, d, pl, ns = idx[ok], w[ok], d[ok], pl[ok], ns[ok]
        wsum = np.bincount(idx, w, nb)
        cnt = np.bincount(idx, minlength=nb)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_d = np.bincount(idx, w * d, nb) / wsum
            mean_pl = np.bincount(idx, w * pl, nb) / wsum
            mean_ns = np.bincount(idx, w * ns, nb) / wsum
        med, p10, p90 = (np.full(nb, np.nan) for _ in range(3))
        for i in np.where(cnt >= 5)[0]:
            b = idx == i
            med[i], p10[i], p90[i] = _wpercentile(d[b], w[b], [0.5, 0.1, 0.9])
        out[side] = dict(intensity=wsum * scale / n_photons, mean_depth=mean_d,
                         median_depth=med, p10_depth=p10, p90_depth=p90,
                         mean_pathlen=mean_pl, mean_nscat=mean_ns, count=cnt)
    return centers, out


def grouped_profile(result, mode, strip_width, max_offset, n_groups, bin_width=0.2):
    """그룹별 offset_profile → {side: {key: (G, nbins) array}}"""
    G = max(1, int(n_groups))
    g_of = _group_of(result.detected_group, G)
    n_per = result.n_simulated / G
    per = []
    for g in range(G):
        c, p = offset_profile(result, mode, strip_width, max_offset, n_per,
                              bin_width, sel_extra=(g_of == g))
        per.append(p)
    stacked = {s: {k: np.vstack([p[s][k] for p in per]) for k in per[0][s]}
               for s in per[0]}
    return c, stacked


def mean_sd(a):
    """그룹 축(0) 평균 ± 표준편차 (그룹 1개면 sd = nan)"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # 전부 nan 인 bin
        m = np.nanmean(a, axis=0)
        sd = np.nanstd(a, axis=0, ddof=1) if a.shape[0] > 1 else np.full(m.shape, np.nan)
    return m, sd


def linfit(x, y):
    """y = a x + b, 반환 (a, b, R²)"""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x, y, 1)
    ss_res = np.sum((y - (a * x + b))**2)
    ss_tot = np.sum((y - y.mean())**2)
    return a, b, (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _fwhm(xc, h):
    """히스토그램 반치폭 (peak 에서 바깥으로 선형보간)"""
    if h.max() <= 0:
        return np.nan
    k = int(np.argmax(h))
    half = h[k] / 2
    i = k
    while i > 0 and h[i] > half:
        i -= 1
    j = k
    while j < len(h) - 1 and h[j] > half:
        j += 1
    if h[i] > half or h[j] > half:
        return np.nan                                     # 범위 밖으로 잘림
    xl = np.interp(half, [h[i], h[i + 1]], [xc[i], xc[i + 1]])
    xr = np.interp(half, [h[j], h[j - 1]], [xc[j], xc[j - 1]])
    return xr - xl


def _hist_sides(mode):
    """히스토그램에서 합칠 side: single 은 +x 만, both/symmetric 은 양쪽(+x 기준으로 뒤집힘)"""
    return ('R',) if mode == 'single' else ('R', 'L')


def banana_main_path(result, sides, offset, halfwidth=0.5, n_xbins=10, min_photons=20):
    """
    offset 에서 검출된 광자(띠 검출기, |exit| 가 offset ± halfwidth)의 주 경로.
    0 ~ offset 을 n_xbins 구간으로 나눠, 구간마다 산란점 깊이(경로 길이 기준) 분포의
    중앙값과 25~75 백분위. PathHist 에서 계산.
    반환: x_centers, (median, p25, p75) (데이터 없는 구간 nan), 사용한 광자 수
    """
    hist = result.path_hist
    edges = np.linspace(0.0, offset, n_xbins + 1)
    xc = (edges[:-1] + edges[1:]) / 2
    med, p25, p75 = (np.full(len(xc), np.nan) for _ in range(3))
    # 광자 수: 광자별 배열에서 같은 창
    n_ph = 0
    for s in sides:
        m, off, _ = side_select(result, 'both', s, hist.strip_width)
        n_ph += int(np.sum(m & (np.abs(off - offset) <= halfwidth)))
    if n_ph < min_photons:
        return xc, (med, p25, p75), n_ph
    Hw = hist.window(sides, offset, halfwidth).sum(axis=0)          # (nz, nx)
    hx = hist.x_centers
    for k in range(len(xc)):
        cols = (hx >= edges[k]) & (hx < edges[k + 1])
        if cols.any():
            med[k], p25[k], p75[k] = _hist_percentiles(Hw[:, cols].sum(axis=1),
                                                       hist.z_edges, [0.5, 0.25, 0.75])
    return xc, (med, p25, p75), n_ph


def propagation_curve(result, sides, total_thickness):
    """
    띠 검출 광자(모든 offset 채널)의 깊이별 평균 |x| 와 원점 통과 직선 기울기.
    반환: z_centers, mean_abs_x, valid, slope(|x|/z), angle_deg(깊이 방향 기준)
    """
    hist = result.path_hist
    si = [_SIDE_IDX[s] for s in sides]
    Hz = hist.H[:, si].sum(axis=(0, 1, 2), dtype=np.float64)       # (nz, nx)
    zc = hist.z_centers
    wsum = Hz.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_abs_x = (Hz * np.abs(hist.x_centers)).sum(axis=1) / wsum
    valid = (wsum > 0) & (zc >= 0.2) & (zc <= 0.95 * total_thickness)
    if valid.sum() < 3:
        return zc, mean_abs_x, valid, 0.0, 0.0
    slope = np.sum(zc[valid] * mean_abs_x[valid]) / np.sum(zc[valid]**2)
    return zc, mean_abs_x, valid, slope, np.degrees(np.arctan(slope))


def fwhm_by_depth(result, mode, offsets, n_groups, depth_step=0.5, offset_halfwidth=0.5):
    """
    깊이 z 에서 검출 광자(띠 검출기, 해당 offset 창)의 산란점 x 분포 FWHM.
    'single' 은 +x 광자만, 'both'/'symmetric' 은 -x 광자를 뒤집어 합침. PathHist 에서 계산.
    반환: z_centers, {offset: (G, nz) array}
    """
    hist = result.path_hist
    G = max(1, int(n_groups))
    k = max(1, int(round(depth_step / hist.hz)))                    # z 칸 묶음
    nzb = hist.H.shape[3] // k
    zc = (np.arange(nzb) + 0.5) * k * hist.hz
    out = {t: np.full((G, nzb), np.nan) for t in offsets}
    for t in offsets:
        Hg = _combine_groups(hist.window(_hist_sides(mode), t, offset_halfwidth), G)
        Hg = Hg[:, :nzb * k].reshape(G, nzb, k, -1).sum(axis=2)    # (G, nzb, nx)
        for g in range(G):
            for j in range(nzb):
                out[t][g, j] = _fwhm(hist.x_centers, Hg[g, j])
    return zc, out


def compute_metrics(result, mode, strip_width, max_offset, n_groups,
                    fit_range=(1.0, 6.0), seg_width=1.0, bin_width=0.2):
    """
    모든 지표를 section 별 표로 반환.
    section = dict(index_name, index, cols={name: (mean, sd)})
    """
    G = max(1, int(n_groups))
    centers, gp = grouped_profile(result, mode, strip_width, max_offset, G, bin_width)
    sections = {}
    sides = list(gp.keys())

    # 1. offset별 침투 깊이 / 2. 광경로·산란 / 3. 감쇠 강도
    for title, keys in [('Depth per offset', ['mean_depth', 'median_depth', 'p10_depth', 'p90_depth']),
                        ('Signal per offset', ['intensity']),
                        ('Path & scattering', ['mean_pathlen', 'mean_nscat'])]:
        cols = {}
        for s in sides:
            for k in keys:
                name = k if len(sides) == 1 else f"{k}_{s}"
                cols[name] = mean_sd(gp[s][k])
        sections[title] = dict(index_name='offset_mm', index=centers, cols=cols)

    # 4. 비대칭 지수 (both)
    if mode == 'both':
        R, L = gp['R']['intensity'], gp['L']['intensity']
        with np.errstate(invalid='ignore', divide='ignore'):
            asym = (R - L) / (R + L)
        sections['Asymmetry (R-L)/(R+L)'] = dict(index_name='offset_mm', index=centers,
                                                 cols={'asymmetry': mean_sd(asym)})

    # 5. 피팅 (그룹별로 피팅 후 평균 ± sd)
    x0, x1 = fit_range
    fr = (centers >= x0) & (centers <= x1)
    fit_cols, seg_cols = {}, {}
    seg_edges = np.arange(x0, x1 + 1e-9, seg_width)
    seg_lbl = [f"{a:.2f}-{b:.2f}" for a, b in zip(seg_edges[:-1], seg_edges[1:])]
    for s in sides:
        suf = '' if len(sides) == 1 else f"_{s}"
        D, I = gp[s]['mean_depth'], gp[s]['intensity']
        rows = []
        segs = []
        for g in range(G):
            a, b, r2 = linfit(centers[fr], D[g, fr])
            with np.errstate(divide='ignore', invalid='ignore'):
                lnI = np.log(np.where(I[g, fr] > 0, I[g, fr], np.nan))
            ai, bi, r2i = linfit(centers[fr], lnI)
            rows.append([a, np.degrees(np.arctan(a)), r2, 1 / a if a else np.nan,
                         -ai, r2i])
            sl = []
            for e0, e1 in zip(seg_edges[:-1], seg_edges[1:]):
                m = (centers >= e0) & (centers <= e1)
                sl.append(linfit(centers[m], D[g, m])[0])
            segs.append(sl)
        rows = np.array(rows)
        names = ['depth_offset_slope', 'angle_deg', 'depth_fit_R2', 'shift_rate_mm_per_mm',
                 'mu_eff_per_mm', 'decay_fit_R2']
        fit_cols['value' + suf] = mean_sd(rows)
        seg_cols['local_slope' + suf] = mean_sd(np.array(segs).reshape(G, -1))
    sections['Fit summary'] = dict(index_name='quantity', index=names, cols=fit_cols)
    sections['Local slopes (depth vs offset)'] = dict(index_name='segment_mm', index=seg_lbl,
                                                      cols=seg_cols)

    # 6. 깊이별 FWHM
    offs = [float(t) for t in range(1, int(np.floor(max_offset)) + 1)]
    zc, fw = fwhm_by_depth(result, mode, offs, G)
    sections['Lateral FWHM by depth'] = dict(
        index_name='depth_mm', index=zc,
        cols={f"fwhm_at_offset_{t:g}mm": mean_sd(fw[t]) for t in offs})
    return centers, gp, sections


def write_sections_csv(sections, path_prefix):
    """section 별 CSV: <prefix>_<section>.csv. 열 = index, 각 지표의 mean, sd"""
    import csv
    import re
    paths = []
    for title, sec in sections.items():
        slug = re.sub(r'[^A-Za-z0-9]+', '_', title).strip('_').lower()
        p = f"{path_prefix}_{slug}.csv"
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            hdr = [sec['index_name']]
            for name in sec['cols']:
                hdr += [name, name + '_sd']
            w.writerow(hdr)
            for i, idx in enumerate(sec['index']):
                row = [idx if isinstance(idx, str) else f"{idx:.6g}"]
                for m, sd in sec['cols'].values():
                    row += [f"{m[i]:.6g}", f"{sd[i]:.6g}"]
                w.writerow(row)
        paths.append(p)
    return paths

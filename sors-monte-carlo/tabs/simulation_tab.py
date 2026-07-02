"""
Tab 2: Simulation — 실행 + 시각화
시각화:
  1. Banana Path (+ 깊이 확률분포 violin overlay, 정수 offset)   ← 수정
  2. Propagation Angle
  3. Offset Profile
  4. Depth & Direct/Side Analysis                              ← 신규
     - offset별 probed-depth 분포(violin) + 평균선
     - 깊이별 직진(central) vs 측면(side) 비율(stacked bar)

※ core(monte_carlo) 물리는 변경하지 않음. 분석은 result.paths / result.detected_x 에서 추출.
※ result.paths 는 시각화용 샘플(config.n_paths_to_save)일 수 있음.
   - 표면 복귀 광자가 부족하면 자동 fallback(모든 saved path 사용, 근사)으로 전환하고 로그에 표시.
   - 정밀 통계는 n_paths_to_save 를 키우거나 core 에서 검출광자별 (exit_x, z_max) 저장 권장.
"""

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QProgressBar, QTabWidget, QTextEdit, QMessageBox, QSizePolicy
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavTool
from matplotlib.colors import LogNorm

from core.monte_carlo import simulate, compute_offset_profile


class SimulationThread(QThread):
    progress = pyqtSignal(int, int)
    finished_with_result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            def cb(done, total):
                self.progress.emit(done, total)
            result = simulate(self.config, progress_callback=cb)
            self.finished_with_result.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ──────────────────────────────────────────────────────────────────────────
#  공용 분석 헬퍼
# ──────────────────────────────────────────────────────────────────────────
def _collect_path_points(paths):
    all_x, all_z = [], []
    for p in paths:
        all_x.extend(p[:, 0])
        all_z.extend(p[:, 2])
    return np.array(all_x), np.array(all_z)


def _photon_exit_depth(paths, total_thickness=None):
    """
    각 (표면 복귀=backscatter) 광자의 exit 위치와 최대 도달 깊이 추출.
      exit_x : 상부 표면으로 되돌아온 lateral 위치
      z_max  : 도달한 최대 깊이 (probed depth)
    paths 종점이 정확히 z=0 이 아닐 수 있어 임계값을 넉넉히 둠.
    표면 복귀 광자가 30개 미만이면 fallback(모든 saved path) 사용.
    반환: (exit_x, z_max, used_fallback)
    """
    top_thresh = 0.5 if total_thickness is None else max(0.5, 0.15 * total_thickness)

    ex, zm = [], []
    for p in paths:
        if p.shape[0] < 2:
            continue
        if p[-1, 2] <= top_thresh:
            ex.append(p[-1, 0]); zm.append(p[:, 2].max())
    if len(zm) >= 30:
        return np.asarray(ex), np.asarray(zm), False

    # fallback
    ex, zm = [], []
    for p in paths:
        if p.shape[0] < 2:
            continue
        ex.append(p[-1, 0]); zm.append(p[:, 2].max())
    return np.asarray(ex), np.asarray(zm), True


def _compute_propagation_angle(all_x, all_z, total_thickness, n_bins=25):
    z_bins = np.linspace(0.2, total_thickness * 0.95, n_bins)
    mean_abs_x = np.full(n_bins - 1, np.nan)
    for i in range(n_bins - 1):
        mask = (all_z >= z_bins[i]) & (all_z < z_bins[i+1])
        if mask.sum() > 10:
            mean_abs_x[i] = np.mean(np.abs(all_x[mask]))
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    valid = ~np.isnan(mean_abs_x)
    if valid.sum() < 3:
        return z_centers, mean_abs_x, valid, 0.0, 0.0
    slope = (np.sum(z_centers[valid] * mean_abs_x[valid])
              / np.sum(z_centers[valid]**2))
    angle = np.degrees(np.arctan(slope))
    return z_centers, mean_abs_x, valid, slope, angle


def _integer_offset_targets(max_offset):
    """0.1, 2, 4, 6, 8 ... 형태의 정수 offset 타깃 생성"""
    t = [0.1] + list(range(2, int(np.floor(max_offset)) + 1, 2))
    return [x for x in t if x <= max_offset]


class MplBananaConcept(FigureCanvas):
    """1. banana path (정수 offset) + 깊이 확률분포 violin"""

    def __init__(self):
        self.fig = Figure(figsize=(3.5, 5), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(700)

    def plot(self, result, paths, total_thickness, max_offset=8.0,
             show_depth_violin=True, offset_halfwidth=0.5):
        self.ax.clear()
        if not paths or len(result.detected_x) == 0:
            self.ax.text(0.5, 0.5, "(no data)\nRun simulation first",
                          ha='center', va='center', transform=self.ax.transAxes,
                          fontsize=11)
            self.draw()
            return

        centers, intensity, mean_depth = compute_offset_profile(
            result, n_bins=40, max_offset=max_offset
        )
        valid = (mean_depth > 0) & (intensity > 0)
        vc, vd = centers[valid], mean_depth[valid]

        # ── 정수 offset 타깃에서 depth 를 보간(interp) → 위치가 정수로 떨어짐 ──
        banana_data = []
        if vc.size >= 2:
            order = np.argsort(vc)
            vc_s, vd_s = vc[order], vd[order]
            for t in _integer_offset_targets(max_offset):
                d = float(np.interp(t, vc_s, vd_s))
                banana_data.append({'offset': float(t), 'depth': d})

        # 배경 photon 분포
        all_x, all_z = [], []
        for p in paths[:2000]:
            all_x.extend(p[:, 0]); all_z.extend(p[:, 2])
        if all_x:
            all_x = np.array(all_x); all_z = np.array(all_z)
            max_depth_plot = max(total_thickness,
                                 max(b['depth'] for b in banana_data) * 1.2)
            self.ax.hexbin(all_x, all_z, gridsize=50, cmap='Blues',
                            mincnt=1, alpha=0.2,
                            extent=[-max_offset, max_offset, 0, max_depth_plot])

        # 깊이 확률분포 violin (각 정수 offset 위치)
        if show_depth_violin:
            exit_x, z_max, _ = _photon_exit_depth(paths, total_thickness)
            if z_max.size > 20:
                for b in banana_data:
                    off = b['offset']
                    sel = np.abs(exit_x - off) <= offset_halfwidth
                    d_here = z_max[sel]
                    if d_here.size >= 8:
                        vp = self.ax.violinplot(d_here, positions=[off],
                                                widths=max_offset * 0.10,
                                                showmeans=True, showextrema=False)
                        for body in vp['bodies']:
                            body.set_facecolor('#60a5fa')
                            body.set_alpha(0.30)
                            body.set_edgecolor('#3b82f6')
                            body.set_linewidth(0.8)
                        if 'cmeans' in vp:
                            vp['cmeans'].set_color('#1d4ed8')

        import matplotlib.cm as cm
        colors = cm.plasma(np.linspace(0.1, 0.85, len(banana_data)))
        for i, b in enumerate(banana_data):
            offset, depth = b['offset'], b['depth']
            theta = np.linspace(np.pi, 0, 100)
            x_curve = offset/2 + (offset/2) * np.cos(theta)
            z_curve = depth * np.sin(theta)
            self.ax.plot(x_curve, z_curve, '-', color=colors[i], linewidth=3,
                          label=f"Offset {offset:.1f} mm → depth {depth:.2f} mm")

        self.ax.scatter([0], [0], color='lime', s=300, marker='*',
                         edgecolor='darkgreen', linewidth=2, zorder=15,
                         label='Source')
        for i, b in enumerate(banana_data):
            self.ax.scatter([b['offset']], [0], color=colors[i], s=120,
                             marker='v', edgecolor='black', linewidth=1.2, zorder=12)

        self.ax.axhline(0, color='gray', linewidth=2, alpha=0.5)
        max_depth_y = max(b['depth'] for b in banana_data) * 1.2
        max_offset_x = max(b['offset'] for b in banana_data) * 1.15
        self.ax.set_xlim(-0.5, max_offset_x)
        self.ax.set_ylim(max_depth_y, -0.4)
        self.ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('auto')
        self.draw()


class MplAngle(FigureCanvas):
    """2. 전파 각도 시각화"""

    def __init__(self):
        self.fig = Figure(figsize=(6, 5), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(400)
        self._last_angle = None

    def plot(self, paths, total_thickness, max_offset=8.0):
        self.ax.clear()
        self._last_angle = None
        if not paths:
            self.ax.text(0.5, 0.5, "(no paths)",
                          ha='center', va='center', transform=self.ax.transAxes)
            self.draw()
            return

        all_x, all_z = _collect_path_points(paths)
        z_centers, mean_abs_x, valid, slope, angle = _compute_propagation_angle(
            all_x, all_z, total_thickness)
        self._last_angle = angle

        self.ax.hexbin(all_x, all_z, gridsize=40, cmap='Blues',
                        mincnt=1, alpha=0.15,
                        extent=[-max_offset, max_offset, 0, total_thickness])

        zmax = total_thickness
        for ref_angle in [15, 30, 45, 60, 75]:
            rad = np.radians(ref_angle)
            xmax_ref = zmax * np.tan(rad)
            if xmax_ref < max_offset:
                x_end, z_end = xmax_ref, zmax
            else:
                x_end = max_offset; z_end = max_offset / np.tan(rad)
            self.ax.plot([0, x_end], [0, z_end], ':', color='gray',
                          alpha=0.45, linewidth=1, zorder=2)
            self.ax.plot([0, -x_end], [0, z_end], ':', color='gray',
                          alpha=0.45, linewidth=1, zorder=2)
            self.ax.text(x_end * 0.92, z_end * 0.92, f'{ref_angle}°',
                          fontsize=8, color='gray', alpha=0.7, zorder=3,
                          ha='left', va='top')

        if valid.sum() >= 3:
            rad_mean = np.radians(angle)
            xmax_mean = zmax * np.tan(rad_mean)
            if xmax_mean < max_offset:
                x_end_m, z_end_m = xmax_mean, zmax
            else:
                x_end_m = max_offset; z_end_m = max_offset / np.tan(rad_mean)
            self.ax.plot([0, x_end_m], [0, z_end_m], '-', color='#dc2626',
                          linewidth=3, label=f'Mean propagation: {angle:.1f}°', zorder=10)
            self.ax.plot([0, -x_end_m], [0, z_end_m], '-', color='#dc2626',
                          linewidth=3, zorder=10)

        self.ax.axhline(0, color='gray', linewidth=1.5, alpha=0.5, zorder=1)
        self.ax.scatter([0], [0], color='lime', s=300, marker='*',
                         edgecolor='darkgreen', linewidth=2, label='Source', zorder=20)
        self.ax.set_xlabel("Lateral X (mm)", fontsize=14)
        self.ax.set_ylabel("Depth Z (mm)", fontsize=14)
        self.ax.invert_yaxis()
        self.ax.set_xlim(-max_offset, max_offset)
        self.ax.set_ylim(total_thickness, -0.5)
        self.ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
        self.draw()


class MplOffsetProfile(FigureCanvas):
    """3. Offset profile"""

    def __init__(self):
        self.fig = Figure(figsize=(7, 5), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot(self, result, max_offset=8.0):
        self.ax.clear()
        centers, intensity, _ = compute_offset_profile(
            result, n_bins=40, max_offset=max_offset)
        self.ax.plot(centers, intensity, 'o-', color='#7c3aed',
                     linewidth=2, markersize=6)
        self.ax.fill_between(centers, 0, intensity, alpha=0.25, color='#7c3aed')
        if intensity.max() > 0:
            peak_idx = np.argmax(intensity)
            self.ax.axvline(centers[peak_idx], color='red', linestyle=':',
                             alpha=0.6, label=f'Peak at {centers[peak_idx]:.2f} mm')
            self.ax.legend()
        self.ax.set_xlabel("Distance from source on surface (mm)", fontsize=14)
        self.ax.set_ylabel("Number of photons returning (a.u.)", fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.draw()


# ──────────────────────────────────────────────────────────────────────────
#  4. (신규) Depth & Direct/Side Analysis
# ──────────────────────────────────────────────────────────────────────────
class MplDepthAnalysis(FigureCanvas):
    """
    (좌) offset별 probed-depth 분포(violin) + 평균선
    (우) 깊이 구간별 직진(central) vs 측면(side) 비율(stacked bar)
    central_radius : |exit_x| < 값(mm) 이면 '직진'으로 분류. 빔/객체 반경 정도로.
    """

    def __init__(self):
        self.fig = Figure(figsize=(11, 5), tight_layout=True)
        super().__init__(self.fig)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.last_table_text = ""

    def plot(self, result, paths, total_thickness, max_offset=8.0,
             central_radius=1.0):
        self.ax1.clear(); self.ax2.clear()
        self.last_table_text = ""

        exit_x, z_max, used_fallback = _photon_exit_depth(paths, total_thickness)
        note = "  [fallback: all saved paths]" if used_fallback else ""

        if z_max.size < 20:
            for ax in (self.ax1, self.ax2):
                ax.text(0.5, 0.5,
                        f"(insufficient photons: {z_max.size})\n"
                        "increase n_paths_to_save\n"
                        "or check path z-column / surface return",
                        ha='center', va='center', transform=ax.transAxes)
            self.last_table_text = f"[depth analysis] insufficient photons ({z_max.size})"
            self.draw()
            return

        # ── (좌) offset별 probed-depth 분포 (정수 offset) ──
        offset_targets = [t for t in [0.1, 1, 2, 3, 4, 5, 6, 7, 8] if t <= max_offset]
        hw = 0.5
        data, positions, means, rows = [], [], [], []
        for t in offset_targets:
            d = z_max[np.abs(exit_x - t) <= hw]
            if d.size >= 8:
                data.append(d); positions.append(t); means.append(d.mean())
                rows.append((t, d.mean(), np.median(d), d.std(),
                             float(np.mean(d >= 0.5 * total_thickness)), d.size))
        if data:
            vp = self.ax1.violinplot(data, positions=positions, widths=0.6,
                                     showmeans=True, showextrema=True)
            for body in vp['bodies']:
                body.set_facecolor('#60a5fa'); body.set_alpha(0.5)
            self.ax1.plot(positions, means, 'k--o', lw=1.2, ms=4,
                          label='mean probed depth')
            self.ax1.legend(fontsize=9)
        self.ax1.set_xlabel("Offset (mm)", fontsize=12)
        self.ax1.set_ylabel("Probed depth z_max (mm)", fontsize=12)
        self.ax1.set_title("Probed-depth per offset" + note, fontsize=11)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.invert_yaxis()

        # ── (우) 깊이별 직진 vs 측면 비율 ──
        depth_edges = np.arange(0, total_thickness + 1e-6, 1.0)
        centrals, sides, dlabels, ratio_rows = [], [], [], []
        for i in range(len(depth_edges) - 1):
            d0, d1 = depth_edges[i], depth_edges[i+1]
            m = (z_max >= d0) & (z_max < d1)
            n = int(m.sum())
            if n >= 8:
                c = float(np.mean(np.abs(exit_x[m]) < central_radius))
                centrals.append(c * 100); sides.append((1 - c) * 100)
                dlabels.append(f"{d0:.0f}-{d1:.0f}")
                ratio_rows.append((d0, d1, c * 100, (1 - c) * 100, n))
        if centrals:
            xpos = np.arange(len(dlabels))
            self.ax2.bar(xpos, centrals, color='#ef4444', alpha=0.85,
                         label=f'Direct/central')
            self.ax2.bar(xpos, sides, bottom=centrals, color='#3b82f6', alpha=0.85,
                         label='Side/lateral')
            self.ax2.set_xticks(xpos); self.ax2.set_xticklabels(dlabels)
            self.ax2.legend(fontsize=9, loc='lower left')
        self.ax2.set_xlabel("Probed depth bin (mm)", fontsize=12)
        self.ax2.set_ylabel("Fraction of returning photons (%)", fontsize=12)
        self.ax2.set_title("Direct vs Side by depth", fontsize=11)
        self.ax2.set_ylim(0, 100)
        self.ax2.grid(True, axis='y', alpha=0.3)

        lines = ["[Probed-depth per offset]" + note,
                 "offset  mean  median  std  frac(z>=0.5T)  N"]
        for r in rows:
            lines.append(f"  {r[0]:5.1f}  {r[1]:5.2f}  {r[2]:5.2f}  {r[3]:4.2f}"
                         f"   {r[4]:5.2f}   {r[5]}")
        lines += ["", "[Direct vs Side by depth]", "depth(mm)  direct%  side%   N"]
        for r in ratio_rows:
            lines.append(f"  {r[0]:.0f}-{r[1]:.0f}   {r[2]:6.1f}  {r[3]:6.1f}   {r[4]}")
        self.last_table_text = "\n".join(lines)
        self.draw()


class SimulationTab(QWidget):
    result_ready = pyqtSignal(object)

    def __init__(self, state: dict, get_config_callback, parent=None):
        super().__init__(parent)
        self.state = state
        self.get_config = get_config_callback
        self.thread = None
        self.result = None
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        control = QGroupBox("Simulation Control")
        ctrl_layout = QHBoxLayout(control)
        self.run_btn = QPushButton("▶ Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        ctrl_layout.addWidget(self.run_btn)
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_simulation)
        ctrl_layout.addWidget(self.stop_btn)
        self.progress = QProgressBar(); self.progress.setRange(0, 100)
        ctrl_layout.addWidget(self.progress, stretch=1)
        layout.addWidget(control)

        self.stats_label = QLabel("Ready. Configure Input tab and click Run.")
        self.stats_label.setStyleSheet(
            "padding: 8px 12px; background: #f3f4f6; border-radius: 4px; "
            "font-family: monospace; font-size: 11px;")
        layout.addWidget(self.stats_label)

        self.viz_tabs = QTabWidget()

        banana_tab = QWidget(); banana_layout = QVBoxLayout(banana_tab)
        self.canvas_banana = MplBananaConcept()
        banana_layout.addWidget(NavTool(self.canvas_banana, self))
        banana_layout.addWidget(self.canvas_banana)
        self.viz_tabs.addTab(banana_tab, "🍌 Banana Path")

        angle_tab = QWidget(); angle_layout = QVBoxLayout(angle_tab)
        self.canvas_angle = MplAngle()
        angle_layout.addWidget(NavTool(self.canvas_angle, self))
        angle_layout.addWidget(self.canvas_angle)
        self.viz_tabs.addTab(angle_tab, "📐 Propagation Angle")

        offset_tab = QWidget(); offset_layout = QVBoxLayout(offset_tab)
        self.canvas_offset = MplOffsetProfile()
        offset_layout.addWidget(NavTool(self.canvas_offset, self))
        offset_layout.addWidget(self.canvas_offset)
        self.viz_tabs.addTab(offset_tab, "📊 Offset Profile")

        depth_tab = QWidget(); depth_layout = QVBoxLayout(depth_tab)
        self.canvas_depth = MplDepthAnalysis()
        depth_layout.addWidget(NavTool(self.canvas_depth, self))
        depth_layout.addWidget(self.canvas_depth)
        self.viz_tabs.addTab(depth_tab, "🔬 Depth / Direct-Side")

        layout.addWidget(self.viz_tabs, stretch=1)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        self.log.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.log)

    def log_msg(self, msg):
        self.log.append(msg)

    def run_simulation(self):
        config = self.get_config()
        if not config.layers:
            QMessageBox.warning(self, "No layers",
                                "Input 탭에서 최소 1개 layer를 추가하세요.")
            return
        total_thickness = sum(l.thickness for l in config.layers)
        self.log_msg(f"\n=== Simulation start ===")
        self.log_msg(f"  Photons: {config.n_photons:,}")
        self.log_msg(f"  Layers: {len(config.layers)} (total {total_thickness:.2f} mm)")
        for i, l in enumerate(config.layers):
            self.log_msg(f"    {i+1}. {l.name}: μa={l.mua}, μs={l.mus}, "
                          f"g={l.g}, μs'={l.mus_prime:.2f} cm⁻¹")
        self.log_msg(f"  Save {config.n_paths_to_save} paths for visualization")

        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.progress.setValue(0); self.stats_label.setText("Running...")

        self.thread = SimulationThread(config)
        self.thread.progress.connect(self._on_progress)
        self.thread.finished_with_result.connect(self._on_finished)
        self.thread.error.connect(self._on_error)
        self.thread.start()

    def stop_simulation(self):
        if self.thread and self.thread.isRunning():
            self.thread.terminate(); self.thread.wait(1000)
            self.log_msg("[Stopped by user]")
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)

    def _on_progress(self, done, total):
        pct = int(100 * done / total) if total else 0
        self.progress.setValue(pct)

    def _on_error(self, msg):
        self.log_msg(f"[ERROR] {msg}")
        QMessageBox.critical(self, "Simulation Error", msg[:1000])
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)

    def _on_finished(self, result):
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.progress.setValue(100)
        self.result = result; self.state['last_result'] = result

        det_rate = 100 * result.n_detected / result.n_simulated if result.n_simulated else 0
        stats = (f"Simulated: {result.n_simulated:,} | "
                 f"Detected: {result.n_detected:,} ({det_rate:.1f}%) | "
                 f"Transmitted: {result.n_transmitted:,} | "
                 f"Absorbed: {result.n_absorbed:,} | Paths: {len(result.paths)}")
        self.stats_label.setText(stats)
        self.log_msg(f"[Done] {stats}")

        config = self.get_config()
        max_offset = config.max_offset
        total = result.total_thickness

        self.canvas_banana.plot(result, result.paths, total, max_offset)
        self.canvas_angle.plot(result.paths, total, max_offset)
        self.canvas_offset.plot(result, max_offset)
        # central_radius 는 실험 빔/객체 반경에 맞춰 조정
        self.canvas_depth.plot(result, result.paths, total, max_offset,
                               central_radius=1.0)

        # 진단: 표면 복귀 광자 수 로그
        ex, zm, fb = _photon_exit_depth(result.paths, total)
        self.log_msg(f"[Diag] saved paths={len(result.paths)}, "
                     f"usable photons={zm.size}, fallback={fb}")

        if self.canvas_angle._last_angle is not None:
            self.log_msg(f"[Analysis] Light propagation angle = "
                         f"{self.canvas_angle._last_angle:.1f}°")
        if self.canvas_depth.last_table_text:
            self.log_msg("[Depth/Direct-Side]\n" + self.canvas_depth.last_table_text)

        self.result_ready.emit(result)
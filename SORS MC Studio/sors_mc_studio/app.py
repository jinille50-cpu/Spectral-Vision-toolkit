"""메인 창: 왼쪽 = 무엇을 시뮬레이션할지 (재실행 필요), 그래프 위 = 어떻게 볼지 (즉시 반영)."""
from __future__ import annotations

import csv
import time
import traceback

import numpy as np
from PySide6 import QtCore, QtWidgets

from . import __version__, theme
from .core import analysis as A
from .core.monte_carlo import compute_offset_profile, get_backend_info, simulate
from .plots import AngleCanvas, BananaCanvas, DepthCanvas, OffsetCanvas
from .sidebar import Sidebar
from .views import MetricsPanel, OverlayPanel
from .widgets import PlotFrame, SegmentedControl, muted

VIEWS = ["바나나 경로", "전파 각도", "오프셋 프로파일", "깊이 분석", "실측 비교"]
MODES = [("+x 한쪽", 'single'), ("양쪽 따로", 'both'), ("|x| 합침", 'symmetric')]


class SimThread(QtCore.QThread):
    progress = QtCore.Signal(int, int)
    done = QtCore.Signal(object, float)
    failed = QtCore.Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            t0 = time.time()
            r = simulate(self.config, progress_callback=lambda d, t: self.progress.emit(int(d), int(t)))
            self.done.emit(r, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class Stat(QtWidgets.QWidget):
    """요약 줄의 값 하나 (작은 이름 + 값)"""

    def __init__(self, label, accent=False):
        super().__init__()
        self.setObjectName("Transparent")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.name = QtWidgets.QLabel(label)
        self.name.setObjectName("StatLabel")
        self.value = QtWidgets.QLabel("–")
        self.value.setObjectName("StatValueAccent" if accent else "StatValue")
        lay.addWidget(self.name)
        lay.addWidget(self.value)

    def set(self, text, tooltip=""):
        self.value.setText(text)
        self.setToolTip(tooltip)


class McWorkspace(QtWidgets.QWidget):
    log = QtCore.Signal(str)
    status = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.run_cfg = None
        self.thread = None
        self._build()

    # ---------------------------------------------------------------- UI
    def _build(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(split)

        self.sidebar = Sidebar()
        self.sidebar.run_clicked.connect(self.run)
        self.sidebar.stop_clicked.connect(self.stop)
        self.sidebar.changed.connect(self._on_settings_changed)
        split.addWidget(self.sidebar)

        main = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(main)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(8)

        # ---- 보기 헤더 (즉시 반영)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(10)
        self.view_seg = SegmentedControl(VIEWS, 0)
        head.addWidget(self.view_seg)
        head.addStretch(1)
        head.addWidget(muted("검출 방향"))
        self.mode_seg = SegmentedControl([m[0] for m in MODES], 0)
        self.mode_seg.setToolTip("다시 실행하지 않고 바로 바뀝니다")
        head.addWidget(self.mode_seg)
        head.addWidget(muted("그룹"))
        self.groups = QtWidgets.QComboBox()
        self.groups.addItems([str(g) for g in A.GROUP_CHOICES])
        self.groups.setCurrentText("10")
        self.groups.setToolTip("불확도: 광자를 N개 그룹으로 나눠 평균 ± 표준편차. 다시 실행하지 않고 바로 바뀝니다")
        head.addWidget(self.groups)
        self.legend_chk = QtWidgets.QCheckBox("범례")
        self.legend_chk.setChecked(True)
        head.addWidget(self.legend_chk)
        mv.addLayout(head)

        # ---- 결과 요약 줄
        summ = QtWidgets.QFrame()
        summ.setObjectName("Summary")
        sl = QtWidgets.QHBoxLayout(summ)
        sl.setContentsMargins(16, 8, 16, 8)
        sl.setSpacing(28)
        self.s_photons = Stat("광자")
        self.s_det = Stat("표면 검출", accent=True)
        self.s_trans = Stat("바닥 투과")
        self.s_abs = Stat("흡수")
        self.s_angle = Stat("평균 전파 각도", accent=True)
        self.s_medium = Stat("매질 두께")
        self.s_strip = Stat("검출 띠")
        self.s_time = Stat("실행 시간")
        for s in (self.s_photons, self.s_det, self.s_trans, self.s_abs, self.s_angle,
                  self.s_medium, self.s_strip, self.s_time):
            sl.addWidget(s)
        sl.addStretch(1)
        mv.addWidget(summ)

        # ---- 그래프 + 아래 패널
        vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.stack = QtWidgets.QStackedWidget()
        self.banana = BananaCanvas()
        self.angle = AngleCanvas()
        self.offset = OffsetCanvas()
        self.depth = DepthCanvas()
        self.canvases = [self.banana, self.angle, self.offset, self.depth]
        for c, name in zip(self.canvases, ["banana_path.csv", "propagation_angle.csv",
                                           "offset_profile.csv", "depth_analysis.csv"]):
            c.message("왼쪽에서 매질과 광원을 정하고 '시뮬레이션 실행'을 누르세요.")
            self.stack.addWidget(PlotFrame(c, name))
        self.overlay = OverlayPanel()
        self.stack.addWidget(self.overlay)
        vsplit.addWidget(self.stack)

        bottom = QtWidgets.QTabWidget()
        self.metrics = MetricsPanel()
        self.metrics.overlay = self.overlay
        self.overlay.on_change = self.metrics.recompute
        bottom.addTab(self.metrics, "지표 표")
        self.logbox = QtWidgets.QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setStyleSheet("font-family: Consolas, 'Malgun Gothic'; font-size: 9pt;")
        bottom.addTab(self.logbox, "로그")
        vsplit.addWidget(bottom)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([640, 240])
        mv.addWidget(vsplit, 1)
        split.addWidget(main)
        split.setStretchFactor(1, 1)

        self.view_seg.changed.connect(self.stack.setCurrentIndex)
        self.mode_seg.changed.connect(lambda _: self.analyze())
        self.groups.currentIndexChanged.connect(lambda _: self.analyze())
        self.legend_chk.toggled.connect(self._set_legend)
        self.log.connect(self.logbox.appendPlainText)

    # ---------------------------------------------------------------- 실행
    def run(self):
        cfg = self.sidebar.get_config()
        T = sum(l.thickness for l in cfg.layers)
        self.log.emit(f"── 실행: 광자 {cfg.n_photons:,}, 매질 {len(cfg.layers)}층 총 {T:g} mm, "
                      f"광원 {self.sidebar.source_label()}, 띠 폭 {cfg.detector_radius:g} mm, "
                      f"최대 offset {cfg.max_offset:g} mm")
        for i, l in enumerate(cfg.layers):
            self.log.emit(f"   {i + 1}. {l.name}: {l.thickness:g} mm, μa={l.mua}, μs={l.mus}, g={l.g}, n={l.n}")
        self.run_cfg = cfg
        self.sidebar.run_btn.setEnabled(False)
        self.sidebar.stop_btn.setEnabled(True)
        self.sidebar.progress.setValue(0)
        self.sidebar.progress_label.setText("실행 중…")
        self._t0 = time.time()
        self.thread = SimThread(cfg)
        self.thread.progress.connect(self._on_progress)
        self.thread.done.connect(self._on_done)
        self.thread.failed.connect(self._on_failed)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait(2000)
            self.log.emit("사용자가 실행을 중지했습니다.")
        self.sidebar.run_btn.setEnabled(True)
        self.sidebar.stop_btn.setEnabled(False)
        self.sidebar.progress_label.setText("중지됨")

    def _on_progress(self, done, total):
        pct = int(100 * done / total) if total else 0
        self.sidebar.progress.setValue(pct)
        el = time.time() - self._t0
        eta = el / done * (total - done) if done else 0
        self.sidebar.progress_label.setText(f"{done:,} / {total:,} 광자   남은 시간 약 {eta:.0f}초")

    def _on_failed(self, msg):
        self.sidebar.run_btn.setEnabled(True)
        self.sidebar.stop_btn.setEnabled(False)
        self.sidebar.progress_label.setText("오류로 멈췄습니다. 로그를 확인하세요.")
        self.log.emit("오류: " + msg)
        QtWidgets.QMessageBox.critical(self, "시뮬레이션 오류", msg.splitlines()[0])

    def _on_done(self, result, seconds):
        self.result = result
        self.sidebar.run_btn.setEnabled(True)
        self.sidebar.stop_btn.setEnabled(False)
        self.sidebar.progress.setValue(100)
        self.sidebar.progress_label.setText(f"완료  {seconds:.1f}초")
        self.sidebar.set_stale(False)
        N = max(1, result.n_simulated)
        self.s_photons.set(f"{result.n_simulated:,}")
        self.s_det.set(f"{100 * result.n_detected / N:.1f}%", "표면으로 돌아와 최대 offset 안에서 나온 광자")
        self.s_trans.set(f"{100 * result.n_transmitted / N:.1f}%", "매질 바닥으로 빠져나간 광자")
        self.s_abs.set(f"{100 * result.n_absorbed / N:.1f}%")
        self.s_medium.set(f"{result.total_thickness:g} mm")
        self.s_strip.set(f"{result.path_hist.strip_width:g} × {result.path_hist.max_offset:g} mm",
                         "띠 폭 × 최대 offset (실행 때 값)")
        self.s_time.set(f"{seconds:.1f} s")
        self.log.emit(f"완료: 검출 {result.n_detected:,}, 투과 {result.n_transmitted:,}, "
                      f"흡수 {result.n_absorbed:,}, {seconds:.1f}초")
        self.status.emit(f"마지막 실행: 광자 {result.n_simulated:,}, 매질 {result.total_thickness:g} mm")
        self.analyze()

    # ---------------------------------------------------------------- 분석 (재실행 없음)
    def analyze(self):
        r = self.result
        if r is None:
            return
        mode = MODES[self.mode_seg.current()][1]
        ctx = dict(mode=mode, strip_width=r.path_hist.strip_width, max_offset=r.path_hist.max_offset,
                   total=r.total_thickness, layers=self.run_cfg.layers,
                   n_groups=int(self.groups.currentText()))
        self.banana.plot(r, ctx)
        self.angle.plot(r, ctx)
        self.offset.plot(r, ctx)
        self.depth.plot(r, ctx)
        centers, gp = A.grouped_profile(r, mode, ctx['strip_width'], ctx['max_offset'], ctx['n_groups'])
        self.overlay.set_sim(centers, gp['S' if mode == 'symmetric' else 'R']['mean_depth'])
        self.metrics.update_metrics(r, mode, ctx['strip_width'], ctx['max_offset'], ctx['n_groups'])
        if self.angle.last_angles:
            txt = "  ".join(f"{a:.1f}°" if s != 'L' else f"(−x) {a:.1f}°"
                            for s, a in self.angle.last_angles.items())
            self.s_angle.set(txt, "깊이 방향 기준, 원점 통과 직선")
        else:
            self.s_angle.set("–")

    def _set_legend(self, show):
        for c in self.canvases + [self.overlay.canvas]:
            c.set_legend_visible(show)

    def _on_settings_changed(self):
        if self.result is not None:
            self.sidebar.set_stale(True)

    # ---------------------------------------------------------------- 저장
    def _need_result(self):
        if self.result is None:
            QtWidgets.QMessageBox.information(self, "저장할 결과 없음", "먼저 시뮬레이션을 실행하세요.")
            return None
        return self.result

    def save_npz(self):
        r = self._need_result()
        if r is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "전체 결과 저장", "sors_mc_result.npz",
                                                        "NumPy (*.npz)")
        if not path:
            return
        h = r.path_hist
        np.savez_compressed(
            path, detected_x=r.detected_x, detected_y=r.detected_y, detected_weight=r.detected_weight,
            detected_max_depth=r.detected_max_depth, detected_pathlen=r.detected_pathlen,
            detected_nscat=r.detected_nscat, detected_group=r.detected_group,
            path_hist=h.H, path_hist_z_edges=h.z_edges, path_hist_x_edges=h.x_edges,
            strip_width=h.strip_width, max_offset=h.max_offset,
            n_simulated=r.n_simulated, n_detected=r.n_detected, n_transmitted=r.n_transmitted,
            n_absorbed=r.n_absorbed, total_thickness=r.total_thickness)
        self.log.emit(f"저장: {path}")

    def save_radial_profile(self):
        r = self._need_result()
        if r is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "반경 기준 오프셋 프로파일", "offset_profile_radial.csv",
                                                        "CSV (*.csv)")
        if not path:
            return
        c, inten, depth = compute_offset_profile(r, n_bins=40, max_offset=r.path_hist.max_offset)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['offset_mm', 'intensity', 'mean_depth_mm'])
            for a, b, d in zip(c, inten, depth):
                w.writerow([f"{a:.4f}", f"{b:.6f}", f"{d:.4f}"])
        self.log.emit(f"저장: {path}")

    def save_detections(self):
        r = self._need_result()
        if r is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "검출 광자 저장", "detections.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['x_mm', 'y_mm', 'weight', 'max_depth_mm', 'pathlen_mm', 'n_scatter'])
            for row in zip(r.detected_x, r.detected_y, r.detected_weight, r.detected_max_depth,
                           r.detected_pathlen, r.detected_nscat):
                w.writerow([f"{row[0]:.4f}", f"{row[1]:.4f}", f"{row[2]:.6g}", f"{row[3]:.4f}",
                            f"{row[4]:.4f}", int(row[5])])
        self.log.emit(f"저장: {path} ({len(r.detected_x):,} 광자)")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"SORS MC Studio {__version__}")
        self.resize(1720, 1000)

        self.mc = McWorkspace()
        central = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(central)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.addWidget(self.mc)
        self.setCentralWidget(central)

        self._build_toolbar()
        sb = self.statusBar()
        info = get_backend_info()
        self.lbl_backend = QtWidgets.QLabel(f"계산 엔진: {info['backend']}  ({info['description']})")
        self.lbl_last = QtWidgets.QLabel("")
        sb.addWidget(self.lbl_backend)
        sb.addPermanentWidget(self.lbl_last)
        self.mc.status.connect(self.lbl_last.setText)

    def _build_toolbar(self):
        tb = QtWidgets.QToolBar("main")
        tb.setMovable(False)
        self.addToolBar(tb)
        title = QtWidgets.QLabel("  SORS MC Studio  ")
        title.setStyleSheet(f"font-weight: 600; font-size: 11pt; color: {theme.TEXT};")
        tb.addWidget(title)
        tb.addSeparator()

        menu = QtWidgets.QMenu(self)
        menu.addAction("전체 결과 (.npz)", self.mc.save_npz)
        menu.addAction("검출 광자 (.csv)", self.mc.save_detections)
        menu.addAction("반경 기준 오프셋 프로파일 (.csv, 이전 앱 형식)", self.mc.save_radial_profile)
        menu.addSeparator()
        menu.addAction("지표 표 전체 (.csv)", self.mc.metrics.export_csv)
        b = QtWidgets.QToolButton()
        b.setText("MC 결과 저장 ▾")
        b.setMenu(menu)
        b.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        tb.addWidget(b)
        spacer = QtWidgets.QWidget()
        spacer.setObjectName("Transparent")
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tip = QtWidgets.QLabel("Ctrl+Enter 실행  ")
        tip.setObjectName("Muted")
        tb.addWidget(tip)

"""메인 윈도우 (간소화판): 좌측 이미지, 우측 상단 그래프, 우측 하단 옵션 두 칸. 스크롤 없이 한 화면."""
from __future__ import annotations

import csv
import os
import sys
import time
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import __version__, theme
from .clustering import ClusterResult, cluster_pieces, cluster_pixels, piece_features, relabel
from .io_utils import FILE_FILTER, SUPPORTED_EXTS, Cube, load_cube
from .processing import CompositeCache, binned_spectrum, false_color, pca_composite, rect_spectrum, snv
from .sam_engine import DEFAULT_MODEL, SamEngine, SegmentationResult
from .widgets.image_view import ImageCanvas
from .widgets.panels import SamPanel, ViewPanel
from .widgets.plot_area import PlotArea

ORG, APP = "SPILab", "HSIViewer"
MAX_RECENT = 10


# --------------------------------------------------------------------------- 백그라운드 로더
class _LoadSignals(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class _LoadTask(QtCore.QRunnable):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.signals = _LoadSignals()

    def run(self):
        try:
            self.signals.done.emit(load_cube(self.path))
        except Exception as e:  # noqa: BLE001
            self.signals.failed.emit(f"{e}\n{traceback.format_exc()}")


# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"HSI Viewer {__version__}")
        self.resize(1700, 960)
        self.setAcceptDrops(True)
        self.settings = QtCore.QSettings(ORG, APP)

        self.cube: Cube | None = None
        self.entries: list[dict] = []
        self.rgb_bands = [1, 3, 4]
        self.cache = CompositeCache()
        self._pca_cache: tuple[int, np.ndarray] | None = None
        self.seg: SegmentationResult | None = None
        self.clus: ClusterResult | None = None
        self.selected_piece = 0
        self._last_rgb: np.ndarray | None = None
        self._last_rgb_key = None
        self._pending_cluster = False
        self._load_task = None
        self.pool = QtCore.QThreadPool.globalInstance()

        self._build_ui()
        self._build_toolbar()
        self._build_statusbar()
        self._build_shortcuts()
        self._connect()

        self.sam = SamEngine(DEFAULT_MODEL, self)
        self.sam.status.connect(self._sam_status)
        self.sam.model_ready.connect(self._sam_ready)
        self.sam.finished.connect(self._sam_finished)
        self.sam.failed.connect(self._sam_failed)
        self.sam_panel.set_model_state("missing" if not self.sam.model_exists else "idle")

        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(16)
        self._render_timer.timeout.connect(self._render_base)

        self._restore_settings()
        self._update_recent_menu()
        self._refresh_entries()
        self.spectrum.set_cursors_visible(False)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.canvas = ImageCanvas()
        self.plots = PlotArea()
        self.spectrum = self.plots.spectrum
        self.cluster_view = self.plots.cluster

        self.view_panel = ViewPanel()
        self.sam_panel = SamPanel()
        options = QtWidgets.QWidget()
        ol = QtWidgets.QHBoxLayout(options)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(8)
        ol.addWidget(self.view_panel, 1)
        ol.addWidget(self.sam_panel, 1)
        for p in (self.view_panel, self.sam_panel):
            p.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.right_split.addWidget(self._framed(self.plots))
        self.right_split.addWidget(options)
        self.right_split.setStretchFactor(0, 1)
        self.right_split.setStretchFactor(1, 0)
        self.right_split.setCollapsible(1, False)

        self.main_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_split.addWidget(self._framed(self.canvas))
        self.main_split.addWidget(self.right_split)
        self.main_split.setStretchFactor(0, 5)
        self.main_split.setStretchFactor(1, 4)
        root.addWidget(self.main_split, 1)

    @staticmethod
    def _framed(w: QtWidgets.QWidget) -> QtWidgets.QFrame:
        f = QtWidgets.QFrame()
        f.setObjectName("Card")
        lay = QtWidgets.QVBoxLayout(f)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(w)
        return f

    def _build_toolbar(self):
        tb = QtWidgets.QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        act = QtGui.QAction("열기…", self)
        act.setShortcut(QtGui.QKeySequence.Open)
        act.triggered.connect(self.open_dialog)
        tb.addAction(act)

        self.recent_menu = QtWidgets.QMenu("최근 파일", self)
        b = QtWidgets.QToolButton()
        b.setText("최근 파일 ▾")
        b.setMenu(self.recent_menu)
        b.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        tb.addWidget(b)
        tb.addSeparator()

        act = QtGui.QAction("화면 맞춤", self)
        act.setShortcut("F")
        act.triggered.connect(self.canvas.fit_view)
        tb.addAction(act)

        act = QtGui.QAction("False color 저장", self)
        act.setToolTip("현재 합성 이미지를 PNG로 저장")
        act.triggered.connect(self.export_false_color)
        tb.addAction(act)
        act = QtGui.QAction("화면 캡처", self)
        act.triggered.connect(self.export_screenshot)
        tb.addAction(act)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.file_label = QtWidgets.QLabel("파일 없음")
        self.file_label.setObjectName("Muted")
        tb.addWidget(self.file_label)
        tb.addWidget(QtWidgets.QLabel("  "))

    def _build_statusbar(self):
        sb = self.statusBar()
        self.lbl_cursor = QtWidgets.QLabel("")
        self.lbl_cursor.setObjectName("Mono")
        self.lbl_cursor.setMinimumWidth(260)
        self.lbl_shape = QtWidgets.QLabel("")
        self.lbl_sam = QtWidgets.QLabel("SAM3: 대기")
        sb.addWidget(self.lbl_cursor)
        sb.addPermanentWidget(self.lbl_shape)
        sb.addPermanentWidget(self.lbl_sam)

    def _build_shortcuts(self):
        for key, fn in (
            ("1", lambda: self.view_panel.mode.setCurrentIndex(0)),
            ("2", lambda: self.view_panel.mode.setCurrentIndex(1)),
            ("3", lambda: self.view_panel.mode.setCurrentIndex(2)),
            ("O", lambda: self.sam_panel.chk_overlay.toggle()),
            ("Ctrl+E", self.export_spectra),
            ("Ctrl+Shift+C", self.clear_entries),
        ):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(fn)

    def _connect(self):
        c = self.canvas
        c.pointAdded.connect(self.add_point)
        c.pointRemoveRequested.connect(self.remove_nearest_point)
        c.pointMoved.connect(self.move_point)
        c.pointMoveFinished.connect(self._refresh_entries)
        c.hovered.connect(self._on_hover)
        c.roiChanged.connect(self._on_roi_changed)
        c.roiRemoveRequested.connect(self._on_roi_remove)
        c.pieceClicked.connect(self.select_piece)
        c.pieceContextRequested.connect(self._piece_menu)
        self.spectrum.cursorMoved.connect(self._on_cursor_moved)

        v = self.view_panel
        v.changed.connect(self._schedule_render)
        v.spectraChanged.connect(self._refresh_entries)
        v.bandChanged.connect(self._on_band_spin)
        v.addRoiRequested.connect(self.add_roi)
        v.clearRequested.connect(self.clear_entries)
        v.exportRequested.connect(self.export_spectra)

        p = self.sam_panel
        p.runRequested.connect(lambda: self.run_detection(cluster_after=True))
        p.clusterRequested.connect(self.run_clustering)
        p.clearRequested.connect(self.clear_segmentation)
        p.exportRequested.connect(self.export_sam)
        p.overlayChanged.connect(self._render_overlay)
        p.loadModelRequested.connect(self._sam_load)

    # ------------------------------------------------------------------ 설정 / 최근 파일
    def _restore_settings(self):
        g = self.settings.value("geometry")
        if g is not None:
            self.restoreGeometry(g)
        st = self.settings.value("main_split_v3")
        if st is not None:
            self.main_split.restoreState(st)
        else:
            self.main_split.setSizes([900, 760])
        opt_h = max(self.view_panel.sizeHint().height(), self.sam_panel.sizeHint().height()) + 8
        self.right_split.setSizes([max(300, self.height() - opt_h - 120), opt_h])
        prompt = self.settings.value("prompt", "")
        if prompt:
            self.sam_panel.prompt.setText(prompt)

    def closeEvent(self, ev):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("main_split_v3", self.main_split.saveState())
        self.settings.setValue("prompt", self.sam_panel.prompt.text())
        try:
            self.sam.shutdown()
        except Exception:
            pass
        super().closeEvent(ev)

    def _recent(self) -> list[str]:
        v = self.settings.value("recent", [])
        if isinstance(v, str):
            v = [v]
        return [p for p in (v or []) if isinstance(p, str)]

    def _push_recent(self, path: str):
        r = [p for p in self._recent() if p != path]
        r.insert(0, path)
        self.settings.setValue("recent", r[:MAX_RECENT])
        self.settings.setValue("last_dir", os.path.dirname(path))
        self._update_recent_menu()

    def _update_recent_menu(self):
        self.recent_menu.clear()
        rec = self._recent()
        if not rec:
            self.recent_menu.addAction("(비어 있음)").setEnabled(False)
            return
        for p in rec:
            self.recent_menu.addAction(os.path.basename(p), lambda p=p: self.open_file(p)).setToolTip(p)

    def _last_dir(self) -> str:
        return self.settings.value("last_dir", "") or ""

    # ------------------------------------------------------------------ 파일 로드
    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "초분광 이미지 열기", self._last_dir(), FILE_FILTER)
        if path:
            self.open_file(path)

    def open_file(self, path: str):
        if not os.path.isfile(path):
            self._error(f"파일을 찾을 수 없습니다:\n{path}")
            return
        self.statusBar().showMessage(f"{os.path.basename(path)} 불러오는 중…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)
        self.setEnabled(False)
        task = _LoadTask(path)
        task.signals.done.connect(self._on_loaded)
        task.signals.failed.connect(self._on_load_failed)
        self._load_task = task
        self.pool.start(task)

    def _on_load_failed(self, msg: str):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.setEnabled(True)
        self.statusBar().clearMessage()
        self._error(f"파일을 불러오지 못했습니다:\n{msg}")

    def _on_loaded(self, cube: Cube):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.setEnabled(True)
        self.cube = cube
        self._pca_cache = None
        self._last_rgb_key = None
        self.cache.reset(id(cube.data))
        n = cube.n_bands
        self.rgb_bands = [min(b, n - 1) for b in self.rgb_bands]
        if n >= 3 and len(set(self.rgb_bands)) < 3:
            self.rgb_bands = [0, n // 2, n - 1]
        self.entries.clear()
        self.canvas.clear_all()
        self.clear_segmentation(silent=True)
        self.view_panel.set_n_bands(n)
        self.view_panel.set_bands(self.rgb_bands)
        self._apply_axis()
        self.spectrum.set_cursors_visible(self.view_panel.state()["mode"] == 0)
        h, w, b = cube.shape
        self.file_label.setText(cube.name)
        self.file_label.setToolTip(cube.path)
        self.lbl_shape.setText(f"{w} × {h} px · {b} 밴드 · {cube.data.nbytes / 1e6:.0f} MB")
        self.setWindowTitle(f"{cube.name} — HSI Viewer")
        self._push_recent(cube.path)
        self._render_base(reset_view=True)
        self._refresh_entries()
        self.statusBar().showMessage(f"{cube.name} 불러옴  ({w}×{h}×{b})", 4000)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith(SUPPORTED_EXTS) for u in ev.mimeData().urls()):
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for u in ev.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(SUPPORTED_EXTS):
                self.open_file(p)
                return

    # ------------------------------------------------------------------ 축
    def _axis(self) -> tuple[np.ndarray, str]:
        if self.cube is None:
            return np.arange(1, dtype=np.float32), "밴드"
        if self.cube.has_wavelengths:
            return self.cube.wavelengths, "파장 (nm)"
        return np.arange(self.cube.n_bands, dtype=np.float32), "밴드 번호"

    def _apply_axis(self):
        x, label = self._axis()
        self.spectrum.set_axis(x, label)
        self.cluster_view.set_axis(x, label)
        self.spectrum.set_cursors(tuple(self.rgb_bands))

    # ------------------------------------------------------------------ 렌더링
    def _schedule_render(self):
        self._render_timer.start()

    def _current_rgb(self) -> np.ndarray:
        st = self.view_panel.state()
        key = (id(self.cube.data), tuple(self.rgb_bands), round(st["low"], 2), round(st["high"], 2), st["invert"])
        if self._last_rgb is None or key != self._last_rgb_key:
            self._last_rgb = false_color(self.cube.data, tuple(self.rgb_bands), st["low"], st["high"], st["invert"], self.cache)
            self._last_rgb_key = key
        return self._last_rgb

    def _render_base(self, reset_view: bool = False):
        if self.cube is None:
            return
        st = self.view_panel.state()
        mode = st["mode"]
        self.spectrum.set_cursors_visible(mode == 0)
        if mode == 0:
            self.canvas.set_image(self._current_rgb(), reset_view=reset_view)
        elif mode == 1:
            b = int(np.clip(st["band"], 0, self.cube.n_bands - 1))
            lo, hi = self.cache.levels(self.cube.data, b, st["low"], st["high"])
            if st["invert"]:
                lo, hi = hi, lo
            self.canvas.set_image(self.cube.data[:, :, b], levels=(lo, hi), lut=self._lut("viridis"), reset_view=reset_view)
        else:
            if self._pca_cache is None or self._pca_cache[0] != id(self.cube.data):
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)
                try:
                    self._pca_cache = (id(self.cube.data), pca_composite(self.cube.data))
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
            sc = self._pca_cache[1]
            out = np.empty(sc.shape, np.uint8)
            for i in range(3):
                flat = sc[:, :, i].ravel()[:: max(1, sc[:, :, i].size // 200_000)]
                lo, hi = np.percentile(flat, (st["low"], st["high"]))
                v = (sc[:, :, i] - lo) * (255.0 / (hi - lo + 1e-8))
                np.clip(v, 0, 255, out=v)
                out[:, :, i] = (255 - v if st["invert"] else v).astype(np.uint8)
            self.canvas.set_image(out, reset_view=reset_view)
        self._render_overlay()

    @staticmethod
    def _lut(name: str):
        try:
            import pyqtgraph as pg

            return pg.colormap.get(name, source="matplotlib").getLookupTable(0.0, 1.0, 256)
        except Exception:
            return None

    def _render_overlay(self):
        if self.cube is None or self.seg is None:
            self.canvas.set_overlay(None)
            self.canvas.set_piece_labels(None)
            self.canvas.set_highlight(None)
            return
        ov = self.sam_panel.overlay_state()
        if not ov["overlay"]:
            self.canvas.set_overlay(None)
            self.canvas.set_piece_labels(None, self.seg.label_map)
            self.canvas.set_highlight(None)
            return
        kind = ov["kind"] if self.clus is not None else 0
        alpha = int(round(ov["opacity"] * 255))
        obj_cols = theme.POINT_COLORS[1:] + theme.CLUSTER_COLORS
        lut = np.zeros((len(self.seg.pieces) + 1, 4), np.uint8)
        labels = []
        for q in self.seg.pieces:
            if kind == 1:
                col, text = theme.CLUSTER_COLORS[q.cluster % len(theme.CLUSTER_COLORS)], f"{q.id}:C{q.cluster}"
            else:
                col, text = obj_cols[(q.id - 1) % len(obj_cols)], f"{q.id}"
            a = alpha
            if self.selected_piece and q.id == self.selected_piece:
                a = min(255, alpha + 80)
            elif self.selected_piece:
                a = max(0, alpha - 40)
            lut[q.id] = (*theme.hex_to_rgb(col), a)
            labels.append((q.centroid[0] + 0.5, q.centroid[1] + 0.5, text, col))
        self.canvas.set_overlay(np.ascontiguousarray(lut[self.seg.label_map]))
        self.canvas.set_piece_labels(labels, self.seg.label_map)
        bbox = None
        if self.selected_piece:
            q = self.seg.pieces[self.selected_piece - 1]
            bbox = (q.bbox[0], q.bbox[1], q.bbox[2] + 1, q.bbox[3] + 1)
        self.canvas.set_highlight(bbox)

    # ------------------------------------------------------------------ 밴드 커서
    def _on_cursor_moved(self, i: int, band: int):
        if self.cube is None:
            return
        self.rgb_bands[i] = int(band)
        self.view_panel.set_bands(self.rgb_bands)
        if self.view_panel.state()["mode"] == 0:
            self._schedule_render()

    def _on_band_spin(self, i: int, band: int):
        if self.cube is None:
            return
        self.rgb_bands[i] = int(np.clip(band, 0, self.cube.n_bands - 1))
        self.spectrum.set_cursors(tuple(self.rgb_bands))
        self._schedule_render()

    # ------------------------------------------------------------------ 포인트 / ROI
    def _next_color(self) -> str:
        used = [e["color"] for e in self.entries]
        for c in theme.POINT_COLORS:
            if c not in used:
                return c
        return theme.POINT_COLORS[len(self.entries) % len(theme.POINT_COLORS)]

    def _next_label(self, prefix: str) -> str:
        n = 1
        labels = {e["label"] for e in self.entries}
        while f"{prefix}{n}" in labels:
            n += 1
        return f"{prefix}{n}"

    def add_point(self, x: int, y: int):
        if self.cube is None:
            return
        self.entries.append({"kind": "point", "x": int(x), "y": int(y), "label": self._next_label("P"), "color": self._next_color()})
        self._refresh_entries()

    def remove_nearest_point(self, x: int, y: int):
        pts = [(i, e) for i, e in enumerate(self.entries) if e["kind"] == "point"]
        if not pts:
            return
        i, _ = min(pts, key=lambda t: (t[1]["x"] - x) ** 2 + (t[1]["y"] - y) ** 2)
        self.remove_entry(i)

    def move_point(self, idx: int, x: int, y: int):
        pts = [i for i, e in enumerate(self.entries) if e["kind"] == "point"]
        if idx < len(pts):
            e = self.entries[pts[idx]]
            e["x"], e["y"] = int(x), int(y)
            self._refresh_entries()

    def remove_entry(self, idx: int):
        if not (0 <= idx < len(self.entries)):
            return
        e = self.entries.pop(idx)
        if e["kind"] == "roi":
            self.canvas.remove_roi(e["roi_idx"])
            for o in self.entries:
                if o["kind"] == "roi" and o["roi_idx"] > e["roi_idx"]:
                    o["roi_idx"] -= 1
        self._refresh_entries()

    def clear_entries(self):
        self.entries.clear()
        self.canvas.clear_rois()
        self._refresh_entries()

    def add_roi(self):
        if self.cube is None:
            return
        color = self._next_color()
        ri = self.canvas.add_roi(color)
        if ri >= 0:
            self.entries.append({"kind": "roi", "roi_idx": ri, "rect": self.canvas.roi_rect(ri), "label": self._next_label("R"), "color": color})
            self._refresh_entries()

    def _on_roi_changed(self, ri: int):
        for e in self.entries:
            if e["kind"] == "roi" and e["roi_idx"] == ri:
                e["rect"] = self.canvas.roi_rect(ri)
        self._refresh_entries()

    def _on_roi_remove(self, ri: int):
        for i, e in enumerate(self.entries):
            if e["kind"] == "roi" and e["roi_idx"] == ri:
                self.remove_entry(i)
                return

    def _entry_spectrum(self, e: dict, bin_size: int, use_snv: bool):
        data = self.cube.data
        if e["kind"] == "point":
            spec, _ = binned_spectrum(data, e["x"], e["y"], bin_size)
            sd = None
        else:
            x0, y0, x1, y1 = e["rect"]
            spec, sd = rect_spectrum(data, x0, y0, x1, y1)
        if use_snv:
            spec, sd = snv(spec), None
        return spec, sd

    def _refresh_entries(self):
        st = self.view_panel.state()
        pts = [(e["x"], e["y"], e["label"]) for e in self.entries if e["kind"] == "point"]
        cols = [e["color"] for e in self.entries if e["kind"] == "point"]
        self.canvas.set_points(pts, cols, st["bin"])
        if self.cube is None:
            self.spectrum.set_spectra([])
            return
        traces = []
        for e in self.entries:
            spec, sd = self._entry_spectrum(e, st["bin"], st["snv"])
            label = e["label"] + (f" ({e['x']},{e['y']})" if e["kind"] == "point" else " (ROI 평균)")
            traces.append({"label": label, "color": e["color"], "y": spec, "std": sd, "width": 2})
        self.spectrum.set_spectra(traces)
        self.spectrum.set_ylabel("SNV" if st["snv"] else "세기 (Intensity)")
        self._update_cluster_view()

    # ------------------------------------------------------------------ 마우스 이동
    def _on_hover(self, x: int, y: int):
        if self.cube is None or x < 0:
            self.lbl_cursor.setText("")
            self.spectrum.set_hover(None)
            return
        st = self.view_panel.state()
        data = self.cube.data
        if st["mode"] == 1:
            b = int(st["band"])
            txt = f"x {x:4d}  y {y:4d}   밴드 {b}: {data[y, x, b]:.4g}"
        else:
            r, g, b = self.rgb_bands
            txt = f"x {x:4d}  y {y:4d}   R {data[y, x, r]:.4g}  G {data[y, x, g]:.4g}  B {data[y, x, b]:.4g}"
        if self.seg is not None:
            pid = int(self.seg.label_map[y, x])
            if pid:
                q = self.seg.pieces[pid - 1]
                txt += f"   객체 {pid}" + (f" · C{q.cluster}" if q.cluster >= 0 else "") + f" · 점수 {q.score:.2f}"
        self.lbl_cursor.setText(txt)
        spec, _ = binned_spectrum(data, x, y, st["bin"])
        self.spectrum.set_hover(snv(spec) if st["snv"] else spec)

    # ------------------------------------------------------------------ SAM3
    def _sam_load(self):
        if not self.sam.model_exists:
            self._error(f"sam3.pt 파일이 없습니다:\n{DEFAULT_MODEL}")
            return
        self.sam_panel.set_model_state("loading")
        self.sam_panel.set_busy(True)
        self.sam.load_model()

    def _sam_status(self, text: str):
        self.sam_panel.set_status(text)
        self.lbl_sam.setText(text if text.startswith("SAM3") else f"SAM3: {text}")

    def _sam_ready(self, ok: bool):
        self.sam_panel.set_model_state("ready" if ok else "error")
        if not self.sam.busy:
            self.sam_panel.set_busy(False)

    def _sam_failed(self, msg: str):
        self.sam_panel.set_busy(False)
        self.sam_panel.set_model_state("error" if "로딩 실패" in msg else ("ready" if self.sam.ready else "idle"))
        self.sam_panel.set_status(msg.splitlines()[0])
        self.lbl_sam.setText("SAM3: 오류")
        self._error(msg)

    def run_detection(self, cluster_after: bool = True):
        if self.cube is None:
            self.statusBar().showMessage("먼저 이미지를 불러오세요", 3000)
            return
        if not self.sam.model_exists:
            self._error(f"sam3.pt 파일이 없습니다:\n{DEFAULT_MODEL}")
            return
        p = self.sam_panel.params()
        if not p["prompts"]:
            self.sam_panel.set_status("프롬프트를 입력하세요 (예: plastic piece).")
            self.sam_panel.prompt.setFocus()
            return
        self._pending_cluster = cluster_after
        self.sam_panel.set_busy(True)
        if not self.sam.ready:
            self.sam_panel.set_model_state("loading")
        self.sam_panel.set_status("SAM3 실행 중…")
        ok = self.sam.segment(rgb=self._current_rgb(), key=self._last_rgb_key, prompts=p["prompts"], conf=p["conf"],
                              nms_iou=p["nms_iou"], min_area=p["min_area"], overlap_iou=p["overlap_iou"])
        if not ok:
            self.sam_panel.set_busy(False)

    def _sam_finished(self, res: SegmentationResult):
        self.sam_panel.set_busy(False)
        self.sam_panel.set_model_state("ready")
        self.seg = res
        self.clus = None
        self.selected_piece = 0
        for q in res.pieces:
            q.cluster = -1
        self.sam_panel.set_has_result(bool(res.pieces))
        self.cluster_view.clear()
        self.plots.show_cluster(False)
        if not res.pieces:
            self.sam_panel.set_status("객체를 찾지 못했습니다. 신뢰도를 낮추거나 프롬프트를 바꿔 보세요.")
        else:
            self.sam_panel.set_status(f"객체 {len(res.pieces)}개 (원시 {res.raw_count}개) · {res.elapsed:.2f}s")
        self._render_overlay()
        if self._pending_cluster and res.pieces:
            self.run_clustering()
        self._pending_cluster = False

    def run_clustering(self):
        if self.cube is None or self.seg is None or not self.seg.pieces:
            return
        p = self.sam_panel.params()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)
        try:
            if p["pixel"]:
                self.clus = cluster_pixels(self.cube.data, self.seg.pieces, self.seg.label_map, p["k"], p["snv"], p["bg_norm"])
            else:
                self.clus = cluster_pieces(self.cube.data, self.seg.pieces, self.seg.label_map, p["k"], p["method"],
                                           p["snv"], p["bg_norm"], p["stat"])
        except Exception as e:  # noqa: BLE001
            self._error(f"클러스터링 실패:\n{e}\n{traceback.format_exc()}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._show_cluster_result(switch=True)

    def _show_cluster_result(self, switch: bool = False):
        c = self.clus
        if c is None:
            return
        sizes = " · ".join(f"C{i}: {int(n)}개" for i, n in enumerate(c.counts_pieces))
        sil = f" · 실루엣 {c.silhouette:.2f}" if c.silhouette is not None else ""
        self.sam_panel.set_status(f"객체 {len(self.seg.pieces)}개 → K={c.k} ({c.method}){sil}\n{sizes}\n"
                                  f"객체 우클릭: 클러스터 변경 / 삭제")
        self._update_cluster_view()
        self.plots.show_cluster(True, switch_to=switch)
        self.sam_panel.overlay_kind.setCurrent(1)
        self._render_overlay()

    def _update_cluster_view(self):
        """클러스터 그래프를 포인트 그래프와 같은 공간(세기 / SNV)으로 그린다."""
        if self.clus is None:
            return
        use_snv = self.view_panel.state()["snv"]
        self.cluster_view.set_result(self.clus, theme.CLUSTER_COLORS, "SNV" if use_snv else "세기 (Intensity)",
                                     self.selected_piece or None, space="snv" if use_snv else "raw")

    def select_piece(self, pid: int):
        if self.seg is None:
            return
        self.selected_piece = 0 if pid == self.selected_piece else int(pid)
        if self.clus is not None:
            self._update_cluster_view()
        self._render_overlay()
        if self.selected_piece:
            q = self.seg.pieces[self.selected_piece - 1]
            self.statusBar().showMessage(
                f"객체 {q.id}: 점수 {q.score:.2f}, {q.area} px, 중심 ({q.centroid[0]:.0f}, {q.centroid[1]:.0f})"
                + (f", 클러스터 C{q.cluster}" if q.cluster >= 0 else ""), 5000)

    def clear_segmentation(self, silent: bool = False):
        self.seg = None
        self.clus = None
        self.selected_piece = 0
        self.sam_panel.set_has_result(False)
        if not silent:
            self.sam_panel.set_status("")
        self.cluster_view.clear()
        self.plots.show_cluster(False)
        self._render_overlay()

    # ------------------------------------------------------------------ 객체 우클릭 메뉴
    def _piece_menu(self, pid: int, gpos):
        if self.seg is None or not (1 <= pid <= len(self.seg.pieces)):
            return
        q = self.seg.pieces[pid - 1]
        if self.selected_piece != pid:
            self.select_piece(pid)
        m = QtWidgets.QMenu(self)
        m.addAction(f"객체 {pid}  ·  {q.area} px  ·  점수 {q.score:.2f}").setEnabled(False)
        m.addSeparator()
        if self.clus is not None:
            for c in range(self.clus.k):
                a = m.addAction(f"클러스터 C{c}로 이동" + ("  ✓" if q.cluster == c else ""), lambda c=c: self.set_piece_cluster(pid, c))
                a.setEnabled(q.cluster != c)
            m.addAction(f"새 클러스터 C{self.clus.k}로", lambda: self.set_piece_cluster(pid, self.clus.k))
            m.addSeparator()
        m.addAction("객체 삭제", lambda: self.delete_piece(pid))
        m.exec(gpos if isinstance(gpos, QtCore.QPoint) else QtGui.QCursor.pos())

    def set_piece_cluster(self, pid: int, c: int):
        if self.seg is None or self.clus is None:
            return
        self.seg.pieces[pid - 1].cluster = int(c)
        self.clus = relabel(self.cube.data, self.clus, self.seg.pieces, self.seg.label_map)
        self._show_cluster_result()

    def delete_piece(self, pid: int):
        if self.seg is None:
            return
        seg = self.seg
        keep = [i for i, q in enumerate(seg.pieces) if q.id != pid]
        pieces = [seg.pieces[i] for i in keep]
        label_map = np.zeros_like(seg.label_map)
        for i, q in enumerate(pieces, 1):
            q.id = i
            label_map[q.mask] = i
        seg.pieces, seg.label_map = pieces, label_map
        self.selected_piece = 0
        if self.clus is not None and pieces:
            feats = self.clus.features
            self.clus.features = feats[keep] if feats.shape[0] == len(keep) + 1 else piece_features(
                self.cube.data, pieces, label_map, self.clus.normalised, self.clus.bg_norm)[0]
            self.clus = relabel(self.cube.data, self.clus, pieces, label_map)
            self._show_cluster_result()
        else:
            self.clus = None
            self.cluster_view.clear()
            self.plots.show_cluster(False)
            self.sam_panel.set_status(f"객체 {len(pieces)}개")
            self._render_overlay()
        self.sam_panel.set_has_result(bool(pieces))

    # ------------------------------------------------------------------ 내보내기
    def _save_dialog(self, title: str, default_name: str, filt: str) -> str:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, os.path.join(self._last_dir(), default_name), filt)
        if path:
            self.settings.setValue("last_dir", os.path.dirname(path))
        return path

    def export_spectra(self):
        if self.cube is None or not self.entries:
            self.statusBar().showMessage("내보낼 것이 없습니다. 포인트나 ROI를 먼저 추가하세요.", 3000)
            return
        st = self.view_panel.state()
        name = f"{self.cube.stem}_spectra{'_SNV' if st['snv'] else ''}.csv"
        path = self._save_dialog("스펙트럼 저장", name, "CSV (*.csv);;Text (*.txt)")
        if not path:
            return
        x, xlabel = self._axis()
        cols, header = [x], [xlabel]
        for e in self.entries:
            spec, sd = self._entry_spectrum(e, st["bin"], st["snv"])
            cols.append(spec)
            header.append(e["label"] + ("_mean" if e["kind"] == "roi" else ""))
            if sd is not None:
                cols.append(sd)
                header.append(e["label"] + "_std")
        delim = "," if path.lower().endswith(".csv") else "\t"
        np.savetxt(path, np.column_stack(cols), fmt="%.6f", delimiter=delim, header=delim.join(header), comments="", encoding="utf-8")
        self.statusBar().showMessage(f"스펙트럼 저장됨 → {path}", 4000)

    def export_false_color(self):
        if self.cube is None:
            return
        r, g, b = self.rgb_bands
        path = self._save_dialog("False color 이미지 저장", f"{self.cube.stem}_falsecolor_R{r}_G{g}_B{b}.png", "PNG (*.png);;TIFF (*.tif)")
        if path:
            from PIL import Image

            Image.fromarray(self._current_rgb()).save(path)
            self.statusBar().showMessage(f"저장됨 → {path}", 4000)

    def export_screenshot(self):
        if self.cube is None:
            return
        path = self._save_dialog("화면 캡처 저장", f"{self.cube.stem}_view.png", "PNG (*.png)")
        if path:
            self.canvas.grab_image().save(path)
            self.statusBar().showMessage(f"저장됨 → {path}", 4000)

    def export_sam(self, key: str):
        if self.cube is None or self.seg is None:
            return
        stem = self.cube.stem
        if key == "all":
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "전부 저장할 폴더 선택", self._last_dir())
            if not d:
                return
            for k in ("pieces_csv", "cluster_means_csv", "piece_spectra_csv", "label_map_npy", "cluster_map_npy", "overlay_png"):
                try:
                    self._export_one(k, os.path.join(d, self._default_name(k, stem)))
                except Exception as e:  # noqa: BLE001
                    self.statusBar().showMessage(f"{k} 실패: {e}", 4000)
            self.statusBar().showMessage(f"{d} 에 저장됨", 5000)
            return
        ext = self._default_name(key, stem).rsplit(".", 1)[1]
        filt = {"npy": "NumPy (*.npy)", "csv": "CSV (*.csv)", "png": "PNG (*.png)"}[ext]
        path = self._save_dialog("내보내기", self._default_name(key, stem), filt)
        if not path:
            return
        try:
            self._export_one(key, path)
            self.statusBar().showMessage(f"저장됨 → {path}", 4000)
        except Exception as e:  # noqa: BLE001
            self._error(f"내보내기 실패:\n{e}")

    @staticmethod
    def _default_name(key: str, stem: str) -> str:
        return {
            "pieces_csv": f"{stem}_objects.csv",
            "cluster_means_csv": f"{stem}_cluster_spectra.csv",
            "piece_spectra_csv": f"{stem}_object_spectra.csv",
            "label_map_npy": f"{stem}_labels.npy",
            "cluster_map_npy": f"{stem}_clusters.npy",
            "overlay_png": f"{stem}_overlay.png",
        }[key]

    def _export_one(self, key: str, path: str):
        seg, clus = self.seg, self.clus
        x, xlabel = self._axis()
        if key == "pieces_csv":
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "prompt", "cluster", "score", "area_px", "cx", "cy", "x0", "y0", "x1", "y1"])
                for q in seg.pieces:
                    w.writerow([q.id, q.prompt, q.cluster if q.cluster >= 0 else "", f"{q.score:.4f}", q.area,
                                f"{q.centroid[0]:.1f}", f"{q.centroid[1]:.1f}", *q.bbox])
        elif key == "cluster_means_csv":
            if clus is None:
                raise ValueError("클러스터링 결과가 아직 없습니다")
            cols, header = [x], [xlabel]
            _, m_raw, s_raw = clus.display.get("raw", (None, clus.means, clus.stds))
            for c in range(clus.k):
                cols += [m_raw[c], s_raw[c], clus.means[c]]
                header += [f"C{c}_mean", f"C{c}_std", f"C{c}_mean_norm"]
            np.savetxt(path, np.column_stack(cols), fmt="%.6f", delimiter=",", header=",".join(header), comments="", encoding="utf-8")
        elif key == "piece_spectra_csv":
            f_raw = clus.display.get("raw", (None,))[0] if clus is not None else None
            if f_raw is not None and f_raw.shape[0] == len(seg.pieces):
                feats, hdr = f_raw, [f"obj{q.id}_C{q.cluster}" for q in seg.pieces]
            else:
                feats = np.vstack([np.median(self.cube.data[q.mask], axis=0) for q in seg.pieces])
                hdr = [f"obj{q.id}" for q in seg.pieces]
            np.savetxt(path, np.column_stack([x, feats.T]), fmt="%.6f", delimiter=",", header=",".join([xlabel] + hdr), comments="", encoding="utf-8")
        elif key == "label_map_npy":
            np.save(path, seg.label_map)
        elif key == "cluster_map_npy":
            if clus is None:
                raise ValueError("클러스터링 결과가 아직 없습니다")
            np.save(path, clus.cluster_map)
        elif key == "overlay_png":
            self.canvas.grab_image().save(path)
        else:
            raise ValueError(key)

    def _error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "HSI Viewer", msg)


# =============================================================================
def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="HSI Viewer")
    parser.add_argument("file", nargs="?")
    parser.add_argument("--screenshot")
    parser.add_argument("--screenshot-delay", type=float, default=2.0)
    args = parser.parse_args(argv)

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setOrganizationName(ORG)
    theme.apply_theme(app)
    win = MainWindow()
    win.show()
    if args.file:
        QtCore.QTimer.singleShot(50, lambda: win.open_file(args.file))
    if args.screenshot:
        def _shot():
            win.grab().save(args.screenshot)
            app.quit()
        QtCore.QTimer.singleShot(int(args.screenshot_delay * 1000), _shot)
    return app.exec()

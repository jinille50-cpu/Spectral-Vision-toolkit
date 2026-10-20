# -*- coding: utf-8 -*-
"""Raman Data Processing App v2 — 코드 기반 레이아웃 (PySide6 + pyqtgraph).

실행:  python main.py
필요:  PySide6, pyqtgraph, numpy, scipy, PyWavelets

원본(../main.py, ../ui_raman.py)의 처리 흐름과 기능은 그대로 두고 화면만 다시 짰다.
"""
import datetime
import json
import os
import sys

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)

import core
import theme
from calibration_dialog import CalibrationDialog

PANEL_WIDTH = 340
FILE_FILTER = "Spectra (*.asc *.txt *.csv);;All files (*)"


def _primary(btn: QPushButton) -> QPushButton:
    btn.setObjectName("primary")
    return btn


def _muted(text="") -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("muted")
    lbl.setWordWrap(True)
    return lbl


class MainWindow(QMainWindow):
    ORG, APP = "Lab", "RamanDataProcessingV2"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raman Data Processing")
        self.resize(1400, 860)
        self.setMinimumSize(1000, 650)

        # ---- 상태 ----
        self.filepath = None
        self.filelist = []
        self.x_nm = None          # 파일 원본 x
        self.y_raw = None         # 파일 원본 y
        self.crop = None
        self.slope, self.intercept = 1.0, 0.0
        self.result = None
        self._syncing = False

        self._build_ui()
        self._connect()
        self._load_settings()
        self._update_enabled()

    # ================================================================== UI
    def _build_ui(self):
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self.tabs = tabs
        self.setCentralWidget(tabs)
        tabs.addTab(self._build_processing_tab(), "Processing")
        tabs.addTab(self._build_save_tab(), "Batch save")
        self.statusBar().showMessage("Load a spectrum to begin.")

    # ---------------------------------------------------------------- 처리 탭
    def _build_processing_tab(self) -> QWidget:
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(10)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        lay.addWidget(split)

        # 왼쪽: 컨트롤 패널 (스크롤)
        panel = QWidget()
        panel.setMinimumWidth(PANEL_WIDTH)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 6, 0)
        pl.setSpacing(8)
        pl.addWidget(self._group_data())
        pl.addWidget(self._group_axis())
        pl.addWidget(self._group_baseline())
        pl.addWidget(self._group_smoothing())
        pl.addWidget(self._group_range())
        pl.addWidget(self._group_actions())
        pl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(PANEL_WIDTH + 16)
        scroll.setMaximumWidth(PANEL_WIDTH + 60)
        split.addWidget(scroll)

        # 오른쪽: 플롯
        plots = QWidget()
        vl = QVBoxLayout(plots)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(8)
        self.ax_raw = self._make_plot("Raw spectrum")
        self.ax_out = self._make_plot("Processed spectrum")
        self.ax_out.setXLink(self.ax_raw)
        vl.addWidget(self.ax_raw, 1)
        vl.addWidget(self.ax_out, 1)
        split.addWidget(plots)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        # 곡선은 한 번 만들고 setData 로 갱신 (legend 유지)
        self.ax_raw.addLegend(offset=(-10, 10))
        self.c_raw = self.ax_raw.plot([], [], pen=pg.mkPen(theme.PLOT_RAW, width=1.5), name="Raw")
        self.c_base = self.ax_raw.plot([], [], pen=pg.mkPen(theme.PLOT_BASE, width=1.5), name="Baseline")
        self.c_out = self.ax_out.plot([], [], pen=pg.mkPen(theme.PLOT_OUT, width=1.5))

        # crop 영역 (raw 플롯 위에서 드래그)
        self.region = pg.LinearRegionItem(brush=theme.REGION, pen=pg.mkPen(theme.ACCENT, width=1))
        self.region.setZValue(-10)
        self.region.setVisible(False)
        self.ax_raw.addItem(self.region)
        return page

    def _make_plot(self, title: str) -> pg.PlotWidget:
        ax = pg.PlotWidget()
        ax.setBackground(theme.PANEL)
        ax.getPlotItem().setTitle(title, color=theme.TEXT_MUTED, size="10pt")
        ax.setLabel("left", "Intensity (A.U.)")
        ax.setLabel("bottom", "Raman shift (cm⁻¹)")
        ax.getAxis("bottom").enableAutoSIPrefix(False)
        ax.getAxis("left").enableAutoSIPrefix(False)
        ax.showGrid(x=True, y=True, alpha=0.15)
        ax.setStyleSheet(f"border: 1px solid {theme.BORDER}; border-radius: 6px;")
        return ax

    def _group_data(self) -> QGroupBox:
        g = QGroupBox("Data")
        v = QVBoxLayout(g)
        v.setSpacing(6)
        self.btn_load = _primary(QPushButton("Load spectrum…"))
        v.addWidget(self.btn_load)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setObjectName("fileName")
        self.lbl_file.setWordWrap(True)
        v.addWidget(self.lbl_file)
        self.lbl_info = _muted("")
        v.addWidget(self.lbl_info)
        return g

    def _group_axis(self) -> QGroupBox:
        g = QGroupBox("X axis")
        f = QFormLayout(g)
        f.setSpacing(6)
        self.laser = QComboBox()
        self.laser.setEditable(True)
        self.laser.addItems(core.LASER_PRESETS)
        self.laser.setCurrentText("532")
        f.addRow("Laser (nm)", self.laser)
        self.chk_shift = QCheckBox("Convert to Raman shift")
        self.chk_shift.setChecked(True)
        f.addRow(self.chk_shift)
        row = QHBoxLayout()
        self.chk_multi = QCheckBox("Multi-frame (PI)")
        self.n_frames = QSpinBox()
        self.n_frames.setRange(1, 100)
        self.n_frames.setValue(10)
        self.n_frames.setSuffix(" frames")
        row.addWidget(self.chk_multi, 1)
        row.addWidget(self.n_frames)
        f.addRow(row)
        return g

    def _group_baseline(self) -> QGroupBox:
        g = QGroupBox("Baseline (AsLS)")
        f = QFormLayout(g)
        f.setSpacing(6)
        self.p_exp = QSpinBox()
        self.p_exp.setRange(0, 10)
        self.p_exp.setValue(3)
        self.p_exp.setPrefix("10^-")
        f.addRow("Asymmetry p", self.p_exp)
        self.lam_exp = QSpinBox()
        self.lam_exp.setRange(0, 10)
        self.lam_exp.setValue(5)
        self.lam_exp.setPrefix("10^")
        f.addRow("Smoothness λ", self.lam_exp)
        return g

    def _group_smoothing(self) -> QGroupBox:
        g = QGroupBox("Smoothing")
        f = QFormLayout(g)
        f.setSpacing(6)
        self.sm_method = QComboBox()
        self.sm_method.addItems(core.SMOOTHING_METHODS)
        f.addRow("Method", self.sm_method)
        self.span = QSpinBox()
        self.span.setRange(1, 201)
        self.span.setValue(11)
        self.lbl_span = QLabel("Window")
        f.addRow(self.lbl_span, self.span)
        return g

    def _group_range(self) -> QGroupBox:
        g = QGroupBox("X range")
        grid = QGridLayout(g)
        grid.setSpacing(6)
        self.x_start = QDoubleSpinBox()
        self.x_end = QDoubleSpinBox()
        for sb, val in ((self.x_start, 200.0), (self.x_end, 2000.0)):
            sb.setRange(-1e6, 1e6)
            sb.setDecimals(1)
            sb.setValue(val)
        grid.addWidget(QLabel("Start"), 0, 0)
        grid.addWidget(self.x_start, 0, 1)
        grid.addWidget(QLabel("End"), 1, 0)
        grid.addWidget(self.x_end, 1, 1)
        self.btn_apply = QPushButton("Apply")
        self.btn_reset = QPushButton("Reset")
        grid.addWidget(self.btn_apply, 0, 2)
        grid.addWidget(self.btn_reset, 1, 2)
        grid.setColumnStretch(1, 1)
        return g

    def _group_actions(self) -> QGroupBox:
        g = QGroupBox("Calibration && output")
        v = QVBoxLayout(g)
        v.setSpacing(6)
        self.btn_cal = QPushButton("X calibration (Cyclohexane)…")
        v.addWidget(self.btn_cal)
        self.lbl_cal = _muted("No calibration applied.")
        v.addWidget(self.lbl_cal)
        self.btn_to_save = QPushButton("Batch save folder…")
        v.addWidget(self.btn_to_save)
        return g

    # ---------------------------------------------------------------- 저장 탭
    def _build_save_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(10)

        head = QHBoxLayout()
        self.lbl_folder = QLabel("Load a spectrum first. Every file with the same extension "
                                 "in its folder will be processed with the current settings.")
        self.lbl_folder.setObjectName("muted")
        self.lbl_folder.setWordWrap(True)
        head.addWidget(self.lbl_folder, 1)
        self.btn_refresh = QPushButton("Refresh list")
        head.addWidget(self.btn_refresh)
        outer.addLayout(head)

        grid = QGridLayout()
        grid.setSpacing(10)
        self.lbl_before = QLabel("Files in folder")
        self.lbl_before.setObjectName("fileName")
        self.txt_before = QPlainTextEdit()
        self.txt_before.setReadOnly(True)
        self.txt_before.setPlaceholderText("No folder loaded")
        grid.addWidget(self.lbl_before, 0, 0)
        grid.addWidget(self.txt_before, 1, 0)

        self.lbl_after = QLabel("Column names in merged output (editable)")
        self.lbl_after.setObjectName("fileName")
        self.txt_after = QPlainTextEdit()
        self.txt_after.setPlaceholderText("One name per line, same order as the file list")
        grid.addWidget(self.lbl_after, 0, 1)
        grid.addWidget(self.txt_after, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        outer.addLayout(grid, 1)

        box = QGroupBox("Name trimming")
        h = QHBoxLayout(box)
        h.addWidget(QLabel("Cut each name at"))
        self.suffix = QLineEdit()
        self.suffix.setPlaceholderText("e.g.  1s 10 avg")
        h.addWidget(self.suffix, 1)
        self.btn_trim = QPushButton("Trim names")
        h.addWidget(self.btn_trim)
        outer.addWidget(box)

        foot = QHBoxLayout()
        foot.addWidget(_muted("Outputs:  Raw Data/  ·  AsLS/  ·  Merge/Merge_Raw.txt, "
                              "Merge_AsLS.txt, metadata.json"), 1)
        self.btn_save_all = _primary(QPushButton("Process && save all"))
        self.btn_save_all.setMinimumWidth(180)
        foot.addWidget(self.btn_save_all)
        outer.addLayout(foot)
        return page

    # ================================================================== 시그널
    def _connect(self):
        self.btn_load.clicked.connect(self.on_load)
        self.btn_apply.clicked.connect(self.on_apply_range)
        self.btn_reset.clicked.connect(self.on_reset_range)
        self.btn_cal.clicked.connect(self.on_calibrate)
        self.btn_to_save.clicked.connect(self.on_goto_save)
        self.btn_refresh.clicked.connect(self.refresh_filelist)
        self.btn_trim.clicked.connect(self.on_trim_names)
        self.btn_save_all.clicked.connect(self.on_save_all)
        self.region.sigRegionChangeFinished.connect(self._region_to_spin)

        for w in (self.p_exp, self.lam_exp, self.span, self.n_frames):
            w.valueChanged.connect(self.refresh)
        self.laser.currentTextChanged.connect(self.refresh)
        self.sm_method.currentTextChanged.connect(self._on_method_changed)
        self.chk_shift.toggled.connect(self._on_shift_toggled)
        self.chk_multi.toggled.connect(self._on_multi_toggled)

    def _on_method_changed(self, method):
        self.lbl_span.setText("Level" if method == "Wavelet Denoising" else "Window")
        self._update_enabled()
        self.refresh()

    def _on_shift_toggled(self, on):
        self._update_enabled()
        self.refresh(reset_range=True)

    def _on_multi_toggled(self, on):
        self._update_enabled()
        self.refresh(reset_range=True)

    def _update_enabled(self):
        loaded = self.y_raw is not None
        self.span.setEnabled(self.sm_method.currentText() != "Raw Data")
        self.lbl_span.setEnabled(self.span.isEnabled())
        self.n_frames.setEnabled(self.chk_multi.isChecked())
        self.laser.setEnabled(self.chk_shift.isChecked())
        self.btn_cal.setEnabled(self.chk_shift.isChecked())
        for w in (self.btn_apply, self.btn_reset, self.btn_to_save, self.btn_save_all,
                  self.btn_refresh, self.btn_trim):
            w.setEnabled(loaded)

    # ================================================================== 파라미터
    def _laser_nm(self) -> float:
        try:
            v = float(self.laser.currentText())
            if v <= 0:
                raise ValueError
            return v
        except ValueError:
            raise ValueError(f"Laser wavelength '{self.laser.currentText()}' is not a positive number.")

    def _params(self) -> core.ProcessParams:
        return core.ProcessParams(
            raman_shift=self.chk_shift.isChecked(),
            laser_nm=self._laser_nm(),
            slope=self.slope, intercept=self.intercept,
            multiframe=self.chk_multi.isChecked(), n_frames=self.n_frames.value(),
            crop=self.crop,
            lam_exp=self.lam_exp.value(), p_exp=self.p_exp.value(),
            smoothing=self.sm_method.currentText(), span=self.span.value(),
        )

    # ================================================================== 동작
    def on_load(self):
        start = os.path.dirname(self.filepath) if self.filepath else self._settings().value("last_dir", "")
        fn, _ = QFileDialog.getOpenFileName(self, "Open spectrum", start, FILE_FILTER)
        if not fn:
            return
        self.load_file(fn)

    def load_file(self, fn: str):
        try:
            data = core.read_xy(fn)
            if data.size == 0:
                raise ValueError("No numeric X/Y rows found in file.")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self.filepath = fn
        self.x_nm, self.y_raw = data[:, 0], data[:, 1]
        self.crop = None
        self.lbl_file.setText(os.path.basename(fn))
        self.setWindowTitle(f"Raman Data Processing — {os.path.basename(fn)}")
        self._update_enabled()
        self.refresh_filelist()
        self.refresh(reset_range=True)

    def refresh(self, *_, reset_range=False):
        """현재 파라미터로 처리 후 플롯 갱신. 오류는 상태바에 표시."""
        if self.y_raw is None:
            return
        try:
            r = core.process_spectrum(self.x_nm, self.y_raw, self._params())
        except ValueError as e:
            self.statusBar().showMessage(f"⚠ {e}")
            return
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Processing error", repr(e))
            return
        self.result = r
        self.c_raw.setData(r.x, r.y)
        self.c_base.setData(r.x, r.baseline)
        self.c_out.setData(r.x, r.corrected)
        for ax in (self.ax_raw, self.ax_out):
            ax.setLabel("bottom", r.xlabel)
            ax.enableAutoRange()
        n_frames = self.n_frames.value() if self.chk_multi.isChecked() else 1
        self.lbl_info.setText(f"{len(r.x)} points · {n_frames} frame{'s' if n_frames > 1 else ''}"
                              + (f" · cropped {r.x.min():.0f}–{r.x.max():.0f}" if self.crop else ""))
        if reset_range or not self.region.isVisible():
            self._sync_region(float(r.x.min()), float(r.x.max()))
            self.region.setVisible(True)
        self.statusBar().showMessage(f"Processed {os.path.basename(self.filepath)}  ·  "
                                     "Tip: drag the shaded region on the raw plot, then Apply.")

    # ---- X range ----
    def _sync_region(self, lo, hi):
        self._syncing = True
        self.region.setRegion((lo, hi))
        self.x_start.setValue(lo)
        self.x_end.setValue(hi)
        self._syncing = False

    def _region_to_spin(self):
        if self._syncing:
            return
        lo, hi = self.region.getRegion()
        self.x_start.setValue(lo)
        self.x_end.setValue(hi)

    def on_apply_range(self):
        lo, hi = sorted((self.x_start.value(), self.x_end.value()))
        self.crop = (lo, hi)
        self._sync_region(lo, hi)
        self.refresh()

    def on_reset_range(self):
        self.crop = None
        self.refresh(reset_range=True)

    # ---- 캘리브레이션 ----
    def on_calibrate(self):
        if not self.chk_shift.isChecked():
            QMessageBox.information(self, "X calibration",
                                    "Enable 'Convert to Raman shift' before calibrating.")
            return
        start = os.path.dirname(self.filepath) if self.filepath else ""
        fn, _ = QFileDialog.getOpenFileName(self, "Open cyclohexane spectrum", start, FILE_FILTER)
        if not fn:
            return
        try:
            data = core.read_xy(fn)
            if data.size == 0:
                raise ValueError("No numeric X/Y rows found in file.")
            xc, yc = data[:, 0], data[:, 1]
            if self.chk_multi.isChecked():
                xc, yc = core.average_frames(xc, yc, self.n_frames.value())
            xr = core.nm_to_raman_shift(xc, self._laser_nm())
            mask = (xr >= 200) & (xr <= 2000)
            xr, yc = xr[mask], yc[mask]
            if len(xr) < 10:
                raise ValueError("Cyclohexane spectrum has no data in 200–2000 cm⁻¹.")
            ycb = yc - core.AsLS(yc, lam_exp=self.lam_exp.value(), p_exp=self.p_exp.value())
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "X calibration", str(e))
            return
        dlg = CalibrationDialog(xr, ycb, self)
        if dlg.exec() != CalibrationDialog.Accepted:
            return
        self.slope, self.intercept = dlg.slope, dlg.intercept
        self.lbl_cal.setText(f"x' = {self.slope:.5f}·x + {self.intercept:.2f}  "
                             f"({os.path.basename(fn)})")
        self.refresh(reset_range=True)

    # ---- 일괄 저장 ----
    def on_goto_save(self):
        self.refresh_filelist()
        self.tabs.setCurrentIndex(1)

    def refresh_filelist(self):
        if not self.filepath:
            return
        folder = os.path.dirname(self.filepath)
        ext = os.path.splitext(self.filepath)[1].lower()
        self.filelist = sorted(f for f in os.listdir(folder) if f.lower().endswith(ext))
        self.txt_before.setPlainText("\n".join(self.filelist))
        self.txt_after.setPlainText("\n".join(os.path.splitext(f)[0] for f in self.filelist))
        self.lbl_before.setText(f"Files in folder ({len(self.filelist)} × {ext})")
        self.lbl_folder.setText(f"Folder: {folder}")

    def on_trim_names(self):
        key = self.suffix.text().strip()
        if not key or not self.filelist:
            return
        names = [(f.split(key)[0] if key in f else os.path.splitext(f)[0]).strip() for f in self.filelist]
        self.txt_after.setPlainText("\n".join(names))

    def on_save_all(self):
        if not self.filelist:
            return
        try:
            P = self._params()
        except ValueError as e:
            QMessageBox.warning(self, "Save", str(e))
            return
        folder = os.path.dirname(self.filepath)
        for sub in ("Raw Data", "AsLS", "Merge"):
            os.makedirs(os.path.join(folder, sub), exist_ok=True)

        names = self.txt_after.toPlainText().split("\n")
        if len(names) != len(self.filelist):
            names = [os.path.splitext(f)[0] for f in self.filelist]
        names = core.unique_names(names)

        xref, raw_cols, asls_cols, used, skipped = None, [], [], [], []
        for f, name in zip(self.filelist, names):
            try:
                data = core.read_xy(os.path.join(folder, f))
                r = core.process_spectrum(data[:, 0], data[:, 1], P)
            except Exception as e:  # noqa: BLE001
                skipped.append(f"{f}: {e}")
                continue
            X, Y, Ycorr = r.x, r.y, r.corrected
            stem = os.path.splitext(f)[0]
            np.savetxt(os.path.join(folder, "Raw Data", stem + ".txt"),
                       np.column_stack([X, Y]), fmt="%.6f", delimiter="\t")
            np.savetxt(os.path.join(folder, "AsLS", stem + ".txt"),
                       np.column_stack([X, Ycorr]), fmt="%.6f", delimiter="\t")
            if xref is None:
                xref = X
            if len(X) != len(xref) or not np.allclose(X, xref):
                order = np.argsort(X)
                Y = np.interp(xref, X[order], Y[order])
                Ycorr = np.interp(xref, X[order], Ycorr[order])
            raw_cols.append(Y)
            asls_cols.append(Ycorr)
            used.append(name)

        if xref is None:
            QMessageBox.warning(self, "Save", "No file could be processed.\n" + "\n".join(skipped[:5]))
            return
        header = "\t".join(["X"] + used)
        np.savetxt(os.path.join(folder, "Merge", "Merge_Raw.txt"), np.column_stack([xref] + raw_cols),
                   fmt="%.6f", delimiter="\t", header=header, comments="")
        np.savetxt(os.path.join(folder, "Merge", "Merge_AsLS.txt"), np.column_stack([xref] + asls_cols),
                   fmt="%.6f", delimiter="\t", header=header, comments="")
        meta = {
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "source_folder": folder,
            "raman_shift_conversion": P.raman_shift,
            "laser_wavelength_nm": P.laser_nm,
            "calibration": {"slope": P.slope, "intercept": P.intercept},
            "multiframe": {"enabled": P.multiframe, "frames": P.n_frames},
            "crop": list(P.crop) if P.crop else None,
            "asls": {"lam_exp": P.lam_exp, "p_exp": P.p_exp},
            "smoothing": {"method": P.smoothing, "span": P.span},
            "skipped": skipped,
        }
        with open(os.path.join(folder, "Merge", "metadata.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        msg = f"Saved {len(used)} files to {folder}"
        if skipped:
            msg += f"\n\nSkipped {len(skipped)}:\n" + "\n".join(skipped[:8])
        QMessageBox.information(self, "Save complete", msg)
        self.statusBar().showMessage(f"Saved {len(used)} files, {len(skipped)} skipped → {folder}")

    # ================================================================== 설정
    def _settings(self) -> QSettings:
        return QSettings(self.ORG, self.APP)

    def _load_settings(self):
        s = self._settings()
        self.laser.setCurrentText(str(s.value("laser", "532")))
        self.chk_shift.setChecked(s.value("raman_shift", "true") in ("true", True))
        self.chk_multi.setChecked(s.value("multiframe", "false") in ("true", True))
        self.n_frames.setValue(int(s.value("n_frames", 10)))
        self.p_exp.setValue(int(s.value("p_exp", 3)))
        self.lam_exp.setValue(int(s.value("lam_exp", 5)))
        self.sm_method.setCurrentText(str(s.value("smoothing", "Raw Data")))
        self.span.setValue(int(s.value("span", 11)))
        self.suffix.setText(str(s.value("suffix", "")))
        self.lbl_span.setText("Level" if self.sm_method.currentText() == "Wavelet Denoising" else "Window")

    def closeEvent(self, ev):
        s = self._settings()
        s.setValue("laser", self.laser.currentText())
        s.setValue("raman_shift", self.chk_shift.isChecked())
        s.setValue("multiframe", self.chk_multi.isChecked())
        s.setValue("n_frames", self.n_frames.value())
        s.setValue("p_exp", self.p_exp.value())
        s.setValue("lam_exp", self.lam_exp.value())
        s.setValue("smoothing", self.sm_method.currentText())
        s.setValue("span", self.span.value())
        s.setValue("suffix", self.suffix.text())
        if self.filepath:
            s.setValue("last_dir", os.path.dirname(self.filepath))
        super().closeEvent(ev)


def create_app(argv=None) -> QApplication:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont(theme.FONT_FAMILY, theme.FONT_PT))
    app.setStyleSheet(theme.STYLESHEET)
    pg.setConfigOptions(antialias=True, background=theme.PANEL, foreground=theme.TEXT)
    return app


def main():
    app = create_app()
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

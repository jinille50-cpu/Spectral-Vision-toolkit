# -*- coding: utf-8 -*-
"""사이클로헥산(CyH) X축 캘리브레이션 다이얼로그.

원본의 matplotlib 슬라이더 창을 앱 안의 pyqtgraph 다이얼로그로 대체.
슬라이더로 피크 검출 민감도를 조절해 레퍼런스 4개와 같은 수의 피크를 고른 뒤 Confirm.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QSlider,
                               QVBoxLayout)

import core
import theme


class CalibrationDialog(QDialog):
    def __init__(self, x, y_corrected, parent=None):
        super().__init__(parent)
        self.setWindowTitle("X calibration (Cyclohexane)")
        self.resize(900, 600)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y_corrected, dtype=float)
        self.idx = np.array([], dtype=int)
        self.slope, self.intercept = 1.0, 0.0

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        hint = QLabel(f"Adjust sensitivity until exactly {len(core.CYH_REFERENCE)} peaks are marked "
                      f"(reference: {', '.join(f'{r:g}' for r in core.CYH_REFERENCE)} cm⁻¹).")
        hint.setObjectName("muted")
        lay.addWidget(hint)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Raman shift (cm⁻¹)")
        self.plot.setLabel("left", "Intensity (A.U.)")
        self.plot.plot(self.x, self.y, pen=pg.mkPen(theme.PLOT_RAW, width=1.5))
        self.marks = pg.ScatterPlotItem(size=11, pen=pg.mkPen(theme.PLOT_BASE, width=2),
                                        brush=pg.mkBrush(255, 255, 255, 0))
        self.plot.addItem(self.marks)
        lay.addWidget(self.plot, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Peak sensitivity"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 20)
        self.slider.setValue(6)
        row.addWidget(self.slider, 1)
        self.count = QLabel("")
        self.count.setMinimumWidth(140)
        row.addWidget(self.count)
        lay.addLayout(row)

        self.result = QLabel("")
        self.result.setObjectName("muted")
        lay.addWidget(self.result)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Confirm")
        self.buttons.button(QDialogButtonBox.Ok).setObjectName("primary")
        self.buttons.accepted.connect(self._confirm)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

        self.slider.valueChanged.connect(self._update)
        self._update(self.slider.value())

    def _update(self, step):
        self.idx = core.find_cyh_peaks(self.y, step)
        self.marks.setData(self.x[self.idx], self.y[self.idx])
        n, need = len(self.idx), len(core.CYH_REFERENCE)
        ok = n == need
        self.count.setText(f"{n} peak{'s' if n != 1 else ''} found" + ("" if ok else f" (need {need})"))
        self.count.setStyleSheet(f"color: {theme.ACCENT if ok else theme.PLOT_BASE}; font-weight: 600;")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
        if ok:
            measured = self.x[self.idx]
            s, b = core.fit_calibration(measured)
            resid = s * measured + b - core.CYH_REFERENCE
            self.result.setText(
                f"x' = {s:.5f}·x + {b:.2f}    residuals: "
                + ", ".join(f"{r:+.2f}" for r in resid) + " cm⁻¹")
        else:
            self.result.setText("")

    def _confirm(self):
        self.slope, self.intercept = core.fit_calibration(self.x[self.idx])
        self.accept()

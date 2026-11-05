"""실측 비교 패널(종이 스타일)과 지표 표 패널."""
from __future__ import annotations

import csv

import matplotlib
import numpy as np
from PySide6 import QtCore, QtWidgets


from .core import analysis as A
from .widgets import MplCanvas, PlotFrame, button, dspin, hint, muted, outside_legend

MEASURED_COLORS = ['#08306b', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef']
SIM_COLOR = '#d62728'
OVERLAY_CMAPS = ["기본 (지정 파랑 6색)", "Blues_r", "viridis", "plasma", "cividis", "magma", "Greys_r"]


class OverlayCanvas(MplCanvas):
    def __init__(self):
        super().__init__(paper=True)


class OverlayPanel(QtWidgets.QWidget):
    """
    실측 centroid 곡선 + 시뮬레이션 곡선.
    실측 CSV: 헤더 1줄, 첫 열 offset(mm), 이후 열 = 두께별 값(mm).
    시뮬레이션 곡선: 현재 검출 방향의 offset별 평균 깊이 (그룹 평균 ± SD).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.measured = None
        self.sim = None
        self.on_change = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        bar = QtWidgets.QHBoxLayout()
        self.load_btn = button("실측 CSV 불러오기")
        self.load_btn.clicked.connect(self.load_csv)
        bar.addWidget(self.load_btn)
        self.file_label = muted("불러온 실측 데이터 없음")
        bar.addWidget(self.file_label, 1)
        bar.addWidget(muted("실측 색"))
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(OVERLAY_CMAPS)
        self.cmap.currentTextChanged.connect(lambda _: self.replot())
        bar.addWidget(self.cmap)
        self.fig_btn = button("그림 저장")
        self.fig_btn.clicked.connect(self.export_figure)
        bar.addWidget(self.fig_btn)
        lay.addLayout(bar)

        self.canvas = OverlayCanvas()
        self.frame = PlotFrame(self.canvas, "overlay.csv")
        lay.addWidget(self.frame, 1)
        self.replot()

    def load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "실측 centroid CSV", "", "CSV (*.csv);;모든 파일 (*)")
        if not path:
            return
        try:
            with open(path, newline='', encoding='utf-8-sig') as f:
                rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
            names = [h.strip() for h in rows[0][1:]]
            data = np.array([[float(c) if c.strip() else np.nan for c in r[:len(names) + 1]]
                             for r in rows[1:]])
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "CSV를 읽지 못했습니다",
                                           f"{e}\n\n형식: 헤더 1줄, 첫 열 offset(mm), 이후 열은 두께별 값.")
            return
        self.measured = (data[:, 0], {n: data[:, i + 1] for i, n in enumerate(names)})
        self.file_label.setText(path)
        self.replot()
        if self.on_change:
            self.on_change()

    def set_sim(self, x, depth_groups):
        self.sim = (x, depth_groups)
        self.replot()

    def _colors(self, n):
        name = self.cmap.currentText()
        if name == OVERLAY_CMAPS[0]:
            return [MEASURED_COLORS[i % len(MEASURED_COLORS)] for i in range(n)]
        return list(matplotlib.colormaps[name](np.linspace(0.05, 0.8, max(n, 1))))

    def replot(self):
        if self.measured is None and self.sim is None:
            self.canvas.message("실측 CSV를 불러오거나 시뮬레이션을 실행하면 여기에 겹쳐 그립니다.")
            self.canvas.export_tables = {}
            return
        with self.canvas.style():
            self.canvas.reset_axes()
            ax = self.canvas.ax
            if self.measured is not None:
                x, cols = self.measured
                for (name, y), c in zip(cols.items(), self._colors(len(cols))):
                    ok = np.isfinite(y)
                    ax.plot(x[ok], y[ok], 'o-', color=c, lw=1.8, ms=4, label=f"실측 {name}")
            if self.sim is not None:
                sx, D = self.sim
                m, sd = A.mean_sd(D)
                if np.isfinite(sd).any():
                    ax.fill_between(sx, m - sd, m + sd, color=SIM_COLOR, alpha=0.18, lw=0)
                ax.plot(sx, m, '-', color=SIM_COLOR, lw=2.2, label="시뮬레이션 (± SD)")
            ax.set_xlabel("Offset (mm)")
            ax.set_ylabel("|Lateral shift| (mm)")
            outside_legend(ax, self.canvas.show_legend)
        self.canvas.export_tables = self._table()
        self.canvas.draw_idle()

    def _table(self):
        """실측 x 격자에 실측 값 + 보간한 시뮬레이션 평균, SD"""
        if self.measured is not None:
            x, cols = self.measured
        else:
            x, cols = self.sim[0], {}
        hdr, data = ['offset_mm'] + list(cols), [x] + list(cols.values())
        if self.sim is not None:
            sx, D = self.sim
            m, sd = A.mean_sd(D)
            ok = np.isfinite(m)
            if ok.any():
                inside = (x >= sx[ok].min()) & (x <= sx[ok].max())
                hdr += ['simulation', 'simulation_sd']
                data += [np.where(inside, np.interp(x, sx[ok], m[ok]), np.nan),
                         np.where(inside, np.interp(x, sx[ok], np.nan_to_num(sd[ok])), np.nan)]
        return {'overlay': (hdr, [list(r) for r in zip(*data)])}

    def agreement_section(self):
        """두께별 RMSE, MAE, 기울기 비(실측/시뮬) — 그룹별 계산 후 평균 ± SD"""
        if self.measured is None or self.sim is None:
            return None
        x, cols = self.measured
        sx, D = self.sim
        names = list(cols)
        res = np.full((D.shape[0], len(names), 3), np.nan)
        for g in range(D.shape[0]):
            ok_s = np.isfinite(D[g])
            if ok_s.sum() < 2:
                continue
            for k, n in enumerate(names):
                y = cols[n]
                ok = np.isfinite(y) & (x >= sx[ok_s].min()) & (x <= sx[ok_s].max())
                if ok.sum() < 2:
                    continue
                s = np.interp(x[ok], sx[ok_s], D[g][ok_s])
                d = y[ok] - s
                a_meas, a_sim = A.linfit(x[ok], y[ok])[0], A.linfit(x[ok], s)[0]
                res[g, k] = [np.sqrt(np.mean(d**2)), np.mean(np.abs(d)),
                             a_meas / a_sim if a_sim else np.nan]
        return dict(index_name='thickness', index=names,
                    cols={'RMSE_mm': A.mean_sd(res[:, :, 0]), 'MAE_mm': A.mean_sd(res[:, :, 1]),
                          'slope_ratio_meas_over_sim': A.mean_sd(res[:, :, 2])})

    def export_figure(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "그림 저장", "overlay.png", "PNG (*.png);;SVG (*.svg)")
        if path:
            self.canvas.fig.savefig(path, dpi=300)


def _fmt_cell(m, sd):
    if not np.isfinite(m):
        return "–"
    return f"{m:.4g}" if not np.isfinite(sd) else f"{m:.4g} ± {sd:.2g}"


class MetricsPanel(QtWidgets.QWidget):
    """논문용 정량 지표 표 + CSV 내보내기"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sections = {}
        self._args = None
        self.overlay = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(muted("표"))
        self.section = QtWidgets.QComboBox()
        self.section.setMinimumWidth(240)
        self.section.currentTextChanged.connect(self.show_section)
        bar.addWidget(self.section)
        bar.addSpacing(16)
        bar.addWidget(muted("피팅 구간"))
        self.fit_x0 = dspin(0, 30, 1.0, 2, 0.5, " mm")
        self.fit_x1 = dspin(0, 30, 6.0, 2, 0.5, " mm")
        bar.addWidget(self.fit_x0)
        bar.addWidget(muted("–"))
        bar.addWidget(self.fit_x1)
        bar.addWidget(muted("국소 기울기 구간"))
        self.seg_w = dspin(0.2, 10, 1.0, 2, 0.2, " mm")
        bar.addWidget(self.seg_w)
        self.apply_btn = button("다시 계산")
        self.apply_btn.clicked.connect(self.recompute)
        bar.addWidget(self.apply_btn)
        bar.addStretch(1)
        self.export_btn = button("표 전체 CSV 저장")
        self.export_btn.clicked.connect(self.export_csv)
        bar.addWidget(self.export_btn)
        lay.addLayout(bar)

        self.note = hint("시뮬레이션을 실행하면 지표가 계산됩니다.")
        lay.addWidget(self.note)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table, 1)

    def update_metrics(self, result, mode, strip_width, max_offset, n_groups):
        self._args = (result, mode, strip_width, max_offset, n_groups)
        self.recompute()

    def recompute(self):
        if self._args is None:
            return
        result, mode, strip_width, max_offset, n_groups = self._args
        _, _, self.sections = A.compute_metrics(
            result, mode, strip_width, max_offset, n_groups,
            fit_range=(self.fit_x0.value(), self.fit_x1.value()), seg_width=self.seg_w.value())
        agree = self.overlay.agreement_section() if self.overlay else None
        if agree is not None:
            self.sections['Agreement vs measured'] = agree
        self.note.setText(
            f"값은 그룹 {n_groups}개(그룹당 광자 {result.n_simulated // max(1, n_groups):,}개)의 평균 ± 표준편차입니다. "
            f"전체 결과의 표준오차는 표준편차 ÷ √{n_groups} 입니다. "
            f"각도 = atan(깊이/offset 기울기), shift rate = 1/기울기.")
        cur = self.section.currentText()
        self.section.blockSignals(True)
        self.section.clear()
        self.section.addItems(list(self.sections))
        self.section.blockSignals(False)
        if cur in self.sections:
            self.section.setCurrentText(cur)
        self.show_section(self.section.currentText())

    def show_section(self, title):
        sec = self.sections.get(title)
        if sec is None:
            return
        names = list(sec['cols'])
        self.table.clear()
        self.table.setColumnCount(1 + len(names))
        self.table.setRowCount(len(sec['index']))
        self.table.setHorizontalHeaderLabels([sec['index_name']] + names)
        for i, idx in enumerate(sec['index']):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(idx if isinstance(idx, str) else f"{idx:.4g}"))
            for j, (m, sd) in enumerate(sec['cols'].values()):
                it = QtWidgets.QTableWidgetItem(_fmt_cell(m[i], sd[i]))
                it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(i, j + 1, it)
        self.table.resizeColumnsToContents()

    def export_csv(self):
        if not self.sections:
            QtWidgets.QMessageBox.information(self, "저장할 지표 없음", "먼저 시뮬레이션을 실행하세요.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "지표 저장 (표마다 CSV 1개)", "metrics.csv", "CSV (*.csv)")
        if not path:
            return
        prefix = path[:-4] if path.lower().endswith('.csv') else path
        files = A.write_sections_csv(self.sections, prefix)
        QtWidgets.QMessageBox.information(self, "저장됨", "\n".join(files))

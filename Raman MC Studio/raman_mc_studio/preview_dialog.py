"""미리보기 팝업 두 개.

- GeometryPreviewDialog: 기하·CCD 채널 — x-z 단면과 x-y 윗면에 격자, 조직, PET 큐브, 레이저, CCD 채널
- ScanPreviewDialog: Case 2·3 스캔 — x 라인 스캔 점과 |x| 구간, XY 래스터 점과 r 구간·반경 원

값을 바꾸면 그림이 바로 바뀌고, '확인'을 누르면 values() 를 RamanView 가 왼쪽 폼에 넣는다.
좌표는 geometry.py 와 같다: x = CCD offset 축, y = 슬릿 방향, z = 깊이 (0 = 조직 표면, +z 아래).
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Rectangle, Wedge
from PySide6 import QtCore, QtWidgets

from .widgets import Card, MplCanvas, PlotFrame, RangeListEditor, dspin, form_row, hint


class GeometryPreviewDialog(QtWidgets.QDialog):
    def __init__(self, values: dict, depths: list, air_mm: float, scan_mm: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("기하·CCD 채널 미리보기")
        self.resize(1180, 720)
        self.air_mm, self.scan_mm = air_mm, scan_mm
        v = values

        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)

        c = Card("기하와 광원", collapsible=False)
        self.voxel = dspin(0.02, 1, v["voxel_mm"], 2, 0.01, " mm")
        self.x0 = dspin(-200, 0, v["x_extent_mm"][0], 2, 0.5, " mm")
        self.x1 = dspin(0, 200, v["x_extent_mm"][1], 2, 0.5, " mm")
        self.y0 = dspin(-200, 0, v["y_extent_mm"][0], 2, 0.5, " mm")
        self.y1 = dspin(0, 200, v["y_extent_mm"][1], 2, 0.5, " mm")
        self.tissue_mm = dspin(2, 50, v["tissue_mm"], 1, 1, " mm")
        self.cube_mm = dspin(0.2, 10, v["cube_mm"], 1, 0.1, " mm")
        self.src = QtWidgets.QComboBox(); self.src.addItems(["cone", "pencil"])
        self.src.setCurrentText(v["source"])
        self.na = dspin(0, 1, v["na"], 2, 0.01)
        for lab, w in (("복셀", self.voxel), ("x 시작", self.x0), ("x 끝", self.x1), ("y 시작", self.y0),
                       ("y 끝", self.y1), ("조직 두께", self.tissue_mm), ("PET 큐브", self.cube_mm),
                       ("광원", self.src), ("NA", self.na)):
            c.add(form_row(lab, w))
        left.addWidget(c)

        c = Card("CCD 채널", collapsible=False)
        self.nch = QtWidgets.QSpinBox(); self.nch.setRange(4, 400); self.nch.setValue(v["n_channels"])
        self.ch_mm = dspin(0.01, 2, v["channel_mm"], 3, 0.01, " mm")
        self.laser_ch = QtWidgets.QSpinBox(); self.laser_ch.setRange(1, 400); self.laser_ch.setValue(v["laser_channel"])
        for lab, w in (("채널 수", self.nch), ("채널 간격", self.ch_mm), ("레이저 채널 (MC)", self.laser_ch)):
            c.add(form_row(lab, w))
        left.addWidget(c)

        c = Card("보기", collapsible=False)
        self.depth = QtWidgets.QComboBox()
        self.depth.addItems([str(d) for d in depths] or ["2"])
        c.add(form_row("PET 윗면 깊이", self.depth))
        c.add(hint("그림에서 강조할 깊이만 고릅니다 (설정에는 저장하지 않음)."))
        left.addWidget(c)

        self.info = hint("")
        left.addWidget(self.info)
        left.addStretch(1)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        box.button(QtWidgets.QDialogButtonBox.Ok).setText("확인 (폼에 반영)")
        box.button(QtWidgets.QDialogButtonBox.Cancel).setText("취소")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        left.addWidget(box)

        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(330)
        root.addWidget(lw)
        self.canvas = MplCanvas(paper=True, nrows=2)
        frame = PlotFrame(self.canvas, "geometry.csv")
        frame.csv_btn.hide()
        root.addWidget(frame, 1)

        # 어느 값이 바뀌어도 다시 그림
        for w in (self.voxel, self.x0, self.x1, self.y0, self.y1, self.tissue_mm, self.cube_mm, self.na, self.ch_mm):
            w.valueChanged.connect(self.redraw)
        for w in (self.nch, self.laser_ch):
            w.valueChanged.connect(self.redraw)
        self.src.currentIndexChanged.connect(self.redraw)
        self.depth.currentIndexChanged.connect(self.redraw)
        self.redraw()

    def values(self) -> dict:
        return dict(voxel_mm=self.voxel.value(), x_extent_mm=[self.x0.value(), self.x1.value()],
                    y_extent_mm=[self.y0.value(), self.y1.value()], tissue_mm=self.tissue_mm.value(),
                    cube_mm=self.cube_mm.value(), source=self.src.currentText(), na=self.na.value(),
                    n_channels=self.nch.value(), channel_mm=self.ch_mm.value(), laser_channel=self.laser_ch.value())

    def redraw(self):
        v = self.values()
        x0, x1 = v["x_extent_mm"]; y0, y1 = v["y_extent_mm"]
        air, tis, cube = self.air_mm, v["tissue_mm"], v["cube_mm"]
        d = float(self.depth.currentText())
        depths = [float(self.depth.itemText(i)) for i in range(self.depth.count())]
        # pipeline.channel_offsets 와 같은 식: 레이저 채널이 offset 0
        off = (np.arange(1, v["n_channels"] + 1) - v["laser_channel"]) * v["channel_mm"]
        h = cube / 2

        with self.canvas.style():
            self.canvas.reset_axes()
            ax, ay = self.canvas.axes

            # ---- x-z 단면 (y = 0)
            ax.add_patch(Rectangle((x0, -air), x1 - x0, air, fc="#eef3fb", ec="none", label="Air"))
            ax.add_patch(Rectangle((x0, 0), x1 - x0, tis, fc="#f3d9cf", ec="none", label="Tissue"))
            for dd in depths:                                   # 다른 깊이는 점선 윤곽만
                if dd != d:
                    ax.add_patch(Rectangle((-h, dd), cube, cube, fc="none", ec="#7a8aa0", ls=":", lw=0.8))
            ax.add_patch(Rectangle((-h, d), cube, cube, fc="#3b7dd8", ec="#1d4f99", label=f"PET cube (top {d:g} mm)"))
            ax.add_patch(Rectangle((x0, -air), x1 - x0, air + tis, fc="none", ec="#333333", lw=1.2))
            if v["source"] == "cone":                           # 표면 한 점으로 모이는 원뿔, 반각 = asin(NA)
                t = np.tan(np.arcsin(min(v["na"], 0.999)))
                ax.fill([-air * t, 0, air * t], [-air, 0, -air], color="#d62728", alpha=0.35, lw=0)
            else:
                ax.plot([0, 0], [-air, 0], color="#d62728", lw=1.5)
            ax.plot(0, 0, marker="v", color="#d62728", ms=8, ls="none", label="Laser (Case 1)")
            ax.plot(off, np.zeros_like(off), "|", color="#2ca02c", ms=10, mew=1.2, label="CCD channels")
            xl = (min(x0, off[0]) - 1, max(x1, off[-1]) + 1)   # 격자 밖으로 나간 채널도 보이게
            ax.set_xlim(*xl)
            ax.set_ylim(max(air + tis, max(depths) + cube) + 0.5, -air - 0.5)   # 깊이는 아래로 증가
            ax.set_aspect("equal")
            ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
            ax.set_title("x-z section (y = 0)")
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

            # ---- x-y 윗면 (표면)
            ay.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#f3d9cf", ec="#333333", lw=1.2))
            ay.add_patch(Rectangle((-h, -h), cube, cube, fc="#3b7dd8", ec="#1d4f99"))
            ay.plot(off, np.zeros_like(off), "|", color="#2ca02c", ms=8, mew=1.2)
            ay.plot(0, 0, marker="o", color="#d62728", ms=6)
            ay.set_xlim(*xl); ay.set_ylim(y0 - 1, y1 + 1)
            ay.set_aspect("equal")
            ay.set_xlabel("x (mm)"); ay.set_ylabel("y (mm)")
            ay.set_title("x-y top view (surface)")
        self.canvas.draw_idle()

        # ---- 숫자 정보와 경고
        nx = int(round((x1 - x0) / v["voxel_mm"])); ny = int(round((y1 - y0) / v["voxel_mm"]))
        nz = int(round((air + tis) / v["voxel_mm"]))
        lines = [f"격자 {nx} × {ny} × {nz} 복셀 (약 {nx * ny * nz * 4 / 1e6:.0f} MB)",
                 f"채널 offset {off[0]:g} ~ {off[-1]:g} mm"]
        half = self.scan_mm / 2
        if max(depths) + cube > tis:
            lines.append(f"⚠ 깊이 {max(depths):g} mm 의 큐브 바닥이 조직 두께 {tis:g} mm 를 넘습니다.")
        if off[0] < x0 or off[-1] > x1:
            lines.append("⚠ CCD 채널이 x 범위 밖으로 나갑니다.")
        if -half - h < x0 or half + h > x1 or -half - h < y0 or half + h > y1:
            lines.append(f"⚠ Case 2·3 스캔 범위 ±{half:g} mm (+ 큐브 반폭) 가 격자 밖으로 나갑니다.")
        self.info.setText("\n".join(lines))


# pipeline.plot_case2_contribution / plot_case3_contribution 과 같은 구간 색
BIN_COLORS = ["#000000", "#1565c0", "#2e7d32", "#ef6c00", "#c62828", "#6a1b9a"]


class ScanPreviewDialog(QtWidgets.QDialog):
    """Case 2 x 라인 스캔과 Case 3 XY 래스터의 레이저 위치, 기여율 구간을 그린다.
    레이저 위치는 PET 중심 기준 (pipeline.scan_positions)."""

    def __init__(self, values: dict, cube_mm: float, x_extent, y_extent, r_max: float, bin_labels, scan_positions,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Case 2·3 스캔 미리보기")
        self.resize(1180, 760)
        self.cube_mm, self.x_extent, self.y_extent, self.r_max = cube_mm, x_extent, y_extent, r_max
        self.bin_labels = bin_labels            # pipeline.contribution_bin_labels (그림 범례와 같은 글자)
        self.scan_positions = scan_positions    # pipeline.scan_positions (계산과 같은 레이저 위치)
        v = values

        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)

        c = Card("Case 2 · 3 스캔", collapsible=False)
        self.sides = QtWidgets.QLineEdit(", ".join(f"{a:g}" for a in v["laser_beside_mm"]))
        self.scan_mm = dspin(2, 60, v["scan_mm"], 1, 1, " mm")
        self.step_mm = dspin(0.05, 2, v["step_mm"], 2, 0.05, " mm")
        for lab, w in (("레이저 옆 위치", self.sides), ("스캔 폭", self.scan_mm), ("스캔 간격", self.step_mm)):
            c.add(form_row(lab, w))
        self.bins = RangeListEditor()
        self.bins.set_ranges(v["bins"])
        row = form_row("기여율 구간", self.bins)
        row.itemAt(0).setAlignment(QtCore.Qt.AlignTop)
        c.add(row)
        c.add(hint(f"Case 3 은 같은 스캔 폭·간격을 x, y 에 쓰고 반경 {r_max:g} mm 원 안만 합산합니다."))
        left.addWidget(c)

        self.info = hint("")
        left.addWidget(self.info)
        left.addStretch(1)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        box.button(QtWidgets.QDialogButtonBox.Ok).setText("확인 (폼에 반영)")
        box.button(QtWidgets.QDialogButtonBox.Cancel).setText("취소")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        left.addWidget(box)

        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(410)
        root.addWidget(lw)
        self.canvas = MplCanvas(paper=True, nrows=2)
        frame = PlotFrame(self.canvas, "scan.csv")
        frame.csv_btn.hide()
        root.addWidget(frame, 1)

        self.sides.textChanged.connect(self.redraw)
        self.scan_mm.valueChanged.connect(self.redraw)
        self.step_mm.valueChanged.connect(self.redraw)
        self.bins.changed.connect(self.redraw)
        self.redraw()

    def _sides(self):
        """레이저 옆 위치 글자 → 숫자 목록. 입력 중인 숫자가 아닌 조각은 건너뜀"""
        out = []
        for t in self.sides.text().replace(";", ",").split(","):
            try:
                out.append(float(t))
            except ValueError:
                pass
        return out

    def values(self) -> dict:
        return dict(laser_beside_mm=self._sides(), scan_mm=self.scan_mm.value(), step_mm=self.step_mm.value(),
                    bins=self.bins.ranges())

    def redraw(self):
        v = self.values()
        half, bins, h, rm = v["scan_mm"] / 2, v["bins"], self.cube_mm / 2, self.r_max
        pos = self.scan_positions(v["scan_mm"], v["step_mm"])
        cols = [BIN_COLORS[i % len(BIN_COLORS)] for i in range(len(bins))]   # %: 색이 모자라면 처음부터 반복

        def bin_index(r):
            """r (배열) 이 들어가는 구간 번호, 없으면 -1 (pipeline 과 같이 lo ≤ r < hi)"""
            idx = np.full(np.shape(r), -1)
            for i, (lo, hi) in enumerate(bins):
                idx[(idx < 0) & (r >= lo) & (r < hi)] = i
            return idx

        with self.canvas.style():
            self.canvas.reset_axes()
            ax, ay = self.canvas.axes

            # ---- Case 2: x 라인 스캔
            for (lo, hi), col, lab in zip(bins, cols, self.bin_labels(bins, "|x|")):   # zip: 목록들을 한 줄씩 묶어 돎
                ax.axvspan(lo, hi, color=col, alpha=0.18, lw=0, label=lab)
                ax.axvspan(-hi, -lo, color=col, alpha=0.18, lw=0)
            in2 = bin_index(np.abs(pos)) >= 0
            ax.plot(pos[in2], np.zeros(in2.sum()), "|", color="#333333", ms=12, label="Laser scan points")
            ax.plot(pos[~in2], np.zeros((~in2).sum()), "|", color="#bbbbbb", ms=12, label="Not in any bin")
            ax.add_patch(Rectangle((-h, -h), self.cube_mm, self.cube_mm, fc="#3b7dd8", ec="#1d4f99", alpha=0.6,
                                   label="PET cube"))
            for sd in v["laser_beside_mm"]:
                j = int(np.argmin(np.abs(pos + sd)))           # case2 와 같이 x = -s 에 가장 가까운 점
                ax.plot(pos[j], 1.8, "v", color="#d62728", ms=8)
                ax.annotate(f"{sd:g}", (pos[j], 1.8), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=8, color="#d62728")
            ax.plot([], [], "v", color="#d62728", label="Laser beside (mm)")
            lim = max(half, max([b[1] for b in bins], default=0)) + 1
            ax.set_xlim(-lim, lim); ax.set_ylim(-2.5, 3.2)
            ax.set_yticks([])
            ax.set_xlabel("Laser x relative to PET center (mm)")
            ax.set_title(f"Case 2: x line scan ({len(pos)} points)")
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

            # ---- Case 3: XY 래스터
            for (lo, hi), col, lab in zip(bins, cols, self.bin_labels(bins, "r")):
                if lo < rm:                                     # 원 밖 구간은 합산하지 않으므로 그리지 않음
                    hi_ = min(hi, rm)
                    ay.add_patch(Wedge((0, 0), hi_, 0, 360, width=hi_ - lo, color=col, alpha=0.18, lw=0, label=lab))
            X, Y = np.meshgrid(pos, pos)
            R = np.hypot(X, Y)                                  # hypot = sqrt(X**2 + Y**2)
            in3 = (R <= rm + 1e-9) & (bin_index(R) >= 0)       # case3 와 같이 원 안이고 구간 안인 점만 합산
            ay.scatter(X[in3], Y[in3], s=1.5, color="#333333", lw=0, label="Summed points")
            ay.scatter(X[~in3], Y[~in3], s=1.5, color="#c8c8c8", lw=0, label="Excluded")
            ay.add_patch(Circle((0, 0), rm, fc="none", ec="#d62728", ls="--", lw=1.2, label=f"r = {rm:g} mm"))
            ay.add_patch(Rectangle((-h, -h), self.cube_mm, self.cube_mm, fc="#3b7dd8", ec="#1d4f99", alpha=0.6))
            lim3 = max(half, rm) + 1
            ay.set_xlim(-lim3, lim3); ay.set_ylim(-lim3, lim3)
            ay.set_aspect("equal")
            ay.set_xlabel("Laser x (mm)"); ay.set_ylabel("Laser y (mm)")
            ay.set_title(f"Case 3: XY raster ({len(pos)} x {len(pos)} = {len(pos) ** 2} points)")
            ay.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, markerscale=4)
        self.canvas.draw_idle()

        # ---- 숫자 정보와 경고
        lines = [f"Case 2 스캔 점 {len(pos)}개 (x {pos[0]:g} ~ {pos[-1]:g} mm)",
                 f"Case 3 래스터 {len(pos)} × {len(pos)} = {len(pos) ** 2}점, 그중 합산 {int(in3.sum())}점"]
        for sd in v["laser_beside_mm"]:
            j = int(np.argmin(np.abs(pos + sd)))
            if abs(pos[j] + sd) > 1e-6:
                lines.append(f"⚠ 레이저 옆 {sd:g} mm 는 스캔 점에 없어 x = {pos[j]:.2f} mm 로 계산됩니다.")
        if not bins:
            lines.append("⚠ 기여율 구간이 없습니다 (저장 때 기존 구간 유지).")
        for i, (lo, hi) in enumerate(bins):
            if lo > pos[-1]:
                lines.append(f"⚠ 구간 {lo:g}~{hi:g} mm 는 스캔 범위 밖이라 Case 2 기여율이 0 입니다.")
            if lo >= rm:
                lines.append(f"⚠ 구간 {lo:g}~{hi:g} mm 는 반경 {rm:g} mm 밖이라 Case 3 기여율이 0 입니다.")
            if any(lo < b2 and a2 < hi for a2, b2 in bins[i + 1:]):
                lines.append(f"⚠ 구간 {lo:g}~{hi:g} mm 가 다른 구간과 겹칩니다.")
        (x0, x1), (y0, y1) = self.x_extent, self.y_extent
        if -half - h < x0 or half + h > x1 or -half - h < y0 or half + h > y1:
            lines.append(f"⚠ 스캔 범위 ±{half:g} mm (+ 큐브 반폭) 가 격자 밖으로 나갑니다.")
        self.info.setText("\n".join(lines))

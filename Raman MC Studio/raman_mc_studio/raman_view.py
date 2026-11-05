"""라만 신호 패턴 작업 공간 — 감도 커널 MC (Case 1 1D 깊이별 직진 패턴, Case 2 2D x 라인 스캔, Case 3 3D XY 래스터, 조직 배경).

설정은 raman_pattern/config.json 하나. 이 화면은 그 파일을 읽고 고치고 저장하며,
계산은 raman_pattern/pipeline.py 와 point_data.py 를 그대로 호출한다 (명령줄과 같은 결과).
"""
from __future__ import annotations

import json
import os
import sys
import traceback

from PySide6 import QtCore, QtWidgets

from .preview_dialog import GeometryPreviewDialog, ScanPreviewDialog
from .widgets import Card, MplCanvas, PlotFrame, RangeListEditor, SegmentedControl, block_wheel, button, dspin, form_row, hint, muted

RP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raman_pattern")
if RP_DIR not in sys.path:
    sys.path.insert(0, RP_DIR)
import pipeline as pl  # noqa: E402


def _list(txt, cast=float):
    return [cast(x) for x in txt.replace(";", ",").split(",") if x.strip()]


class Worker(QtCore.QThread):
    log = QtCore.Signal(str)
    done = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(self, cfg, steps, state):
        super().__init__()
        self.cfg, self.steps, self.state = cfg, steps, state

    def run(self):
        try:
            res = dict(self.state)
            if "fluence" in self.steps or res.get("F") is None:
                res["F"], res["geo"] = pl.get_fluence(self.cfg, log=self.log.emit,
                                                      force="fluence_force" in self.steps)
            if "measured" in self.steps:
                import point_data
                folder = pl._abs(self.cfg, self.cfg["point_data"]["folder"])
                if os.path.isdir(folder):
                    res["measured"] = point_data.measured_profiles(self.cfg, log=self.log.emit)
                else:
                    self.log.emit(f"점 측정 폴더가 없어 비교를 건너뜁니다: {folder}")
            if "case1" in self.steps:
                res["r1"] = pl.case1(self.cfg, res["F"], res["geo"], log=self.log.emit,
                                     measured=res.get("measured"))
            if "case2" in self.steps:
                res["r2"] = pl.case2(self.cfg, res["F"], res["geo"], log=self.log.emit)
            if "case3" in self.steps:
                res["r3"] = pl.case3(self.cfg, res["F"], res["geo"], log=self.log.emit)
            if "bg" in self.steps:
                res["bg"] = pl.tissue_bg(self.cfg, res["F"], res["geo"])
                self.log.emit("조직 배경 B(d) 계산 완료")
            self.done.emit(res)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"{e}\n{traceback.format_exc()}")


CASES = ["Case 1 (1D)", "Case 2 (2D)", "Case 3 (3D)", "조직 배경"]
CASE_PLOTS = [["깊이별 직진 패턴 (전체)", "깊이 하나 (측정 ± SD)"],          # Case 별 그림 목록 (CASES 와 같은 순서)
              ["레이저 위치별 패턴", "레이저 위치별 기여율 (|x| 구간)"],
              ["레이저 위치별 기여율 (r 구간)"],
              ["조직 배경 B(d)"]]


class RamanView(QtWidgets.QWidget):
    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        self.state = state
        self.cfg = pl.load_config()
        self.res = {}
        self.worker = None
        self._build()
        self._cfg_to_form()

    # ---------------------------------------------------------------- UI
    def _build(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(split)

        # ---- 왼쪽: 설정
        left = QtWidgets.QWidget()
        left.setMinimumWidth(380)
        left.setMaximumWidth(470)
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        host = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(host)
        fv.setContentsMargins(0, 0, 4, 0)
        fv.setSpacing(8)

        fv.addWidget(muted("공통 설정"))
        c = Card("설정 파일", "MC 플루언스를 한 번 계산해 Case 1, 2, 3 과 조직 배경에 함께 씁니다. "
                 "계수나 격자를 바꾸면 플루언스를 다시 계산합니다 (GPU, 2천만 광자 약 1분).")
        self.cfg_path = QtWidgets.QLineEdit(self.cfg["_path"])
        c.add(self.cfg_path)
        row = QtWidgets.QHBoxLayout()
        b1 = button("불러오기"); b1.clicked.connect(self.load_cfg)
        b2 = button("저장"); b2.clicked.connect(self.save_cfg)
        row.addWidget(b1); row.addWidget(b2); row.addStretch(1)
        c.add(row)
        fv.addWidget(c)

        c = Card("매질")
        self.mua = dspin(0, 10, 0.08, 4, 0.001, " /mm")
        self.mus = dspin(0, 100, 13.4, 2, 0.1, " /mm")
        self.g_ = dspin(0, 1, 0.915, 3, 0.001)
        self.n_ = dspin(1, 2, 1.42, 2, 0.01)
        for lab, w in (("μa", self.mua), ("μs", self.mus), ("g", self.g_), ("n", self.n_)):
            c.add(form_row(lab, w))
        fv.addWidget(c)

        c = Card("기하와 광원")
        self.voxel = dspin(0.02, 1, 0.1, 2, 0.01, " mm")
        self.xext = QtWidgets.QLineEdit()
        self.yext = QtWidgets.QLineEdit()
        self.tissue_mm = dspin(2, 50, 20, 1, 1, " mm")
        self.cube_mm = dspin(0.2, 10, 2, 1, 0.1, " mm")
        self.src = QtWidgets.QComboBox(); self.src.addItems(["cone", "pencil"])
        self.na = dspin(0, 1, 0.23, 2, 0.01)
        self.nphoton = QtWidgets.QSpinBox()
        self.nphoton.setRange(100000, 500000000)
        self.nphoton.setSingleStep(1000000)
        self.nphoton.setGroupSeparatorShown(True)
        for lab, w in (("복셀", self.voxel), ("x 범위 (a, b)", self.xext), ("y 범위 (a, b)", self.yext),
                       ("조직 두께", self.tissue_mm), ("PET 큐브", self.cube_mm), ("광원", self.src),
                       ("NA", self.na), ("광자 수", self.nphoton)):
            c.add(form_row(lab, w))
        bp = button("미리보기", "Ghost", "x-z 단면과 윗면을 그림으로 보며 조정"); bp.clicked.connect(self._open_preview)
        c.header_extra.addWidget(bp)
        fv.addWidget(c)

        c = Card("CCD 채널")
        self.nch = QtWidgets.QSpinBox(); self.nch.setRange(4, 400)
        self.ch_mm = dspin(0.01, 2, 0.2, 3, 0.01, " mm")
        self.laser_ch = QtWidgets.QSpinBox(); self.laser_ch.setRange(1, 400)
        for lab, w in (("채널 수", self.nch), ("채널 간격", self.ch_mm), ("레이저 채널 (MC)", self.laser_ch)):
            c.add(form_row(lab, w))
        bp = button("미리보기", "Ghost", "x-z 단면과 윗면을 그림으로 보며 조정"); bp.clicked.connect(self._open_preview)
        c.header_extra.addWidget(bp)
        fv.addWidget(c)

        c = Card("출력")
        self.out_dir = QtWidgets.QLineEdit()
        c.add(form_row("결과 폴더", self.out_dir))
        c.add(hint("상대 경로는 raman_pattern 폴더 기준입니다."))
        fv.addWidget(c)

        fv.addWidget(muted("Case 별 설정"))
        c = Card("Case 1 (1D) · 깊이 방향", "레이저가 PET 바로 위에 고정, offset(채널)별 PET 신호")
        self.depths = QtWidgets.QLineEdit()
        c.add(form_row("PET 윗면 깊이", self.depths))
        c.add(hint("mm 단위, 쉼표로 구분. Case 2, 3 도 같은 깊이로 계산합니다."))
        fv.addWidget(c)

        c = Card("Case 1 · 점 측정 비교", "선택 사항")
        self.use_meas = QtWidgets.QCheckBox("Case 1 그림에 점 측정 겹쳐 그리기")
        self.use_meas.setChecked(True)
        c.add(self.use_meas)
        row = QtWidgets.QHBoxLayout()
        self.meas_dir = QtWidgets.QLineEdit()
        bb = button("폴더"); bb.clicked.connect(self._pick_dir)
        row.addWidget(self.meas_dir, 1); row.addWidget(bb)
        c.add(form_row("CSV 폴더", row))
        c.add(hint("상대 경로는 raman_pattern 폴더 기준입니다 (기본: 앱 폴더의 SORS Manual)."))
        self.meas_laser = QtWidgets.QSpinBox(); self.meas_laser.setRange(1, 400)
        self.frames = QtWidgets.QSpinBox(); self.frames.setRange(1, 100)
        c.add(form_row("레이저 채널 (측정)", self.meas_laser))
        c.add(form_row("batch 당 프레임", self.frames))
        fv.addWidget(c)

        c = Card("Case 2 (2D) · x 라인 스캔", "CCD 축과 평행한 x 방향으로 PET 중심을 지나는 한 줄 스캔")
        self.sides = QtWidgets.QLineEdit()
        self.scan_mm = dspin(2, 60, 18, 1, 1, " mm")
        self.step_mm = dspin(0.05, 2, 0.3, 2, 0.05, " mm")
        self.bins = RangeListEditor()
        for lab, w in (("레이저 옆 위치", self.sides), ("스캔 폭", self.scan_mm), ("스캔 간격", self.step_mm)):
            c.add(form_row(lab, w))
        row = form_row("기여율 구간", self.bins)
        row.itemAt(0).setAlignment(QtCore.Qt.AlignTop)   # 줄이 여러 개라 이름표를 위에 붙임
        c.add(row)
        c.add(hint("위치는 mm 단위, 쉼표로 구분. 기여율 구간은 |x| 시작 ~ 끝 (끝이 시작보다 커야 함)."))
        bp = button("미리보기", "Ghost", "스캔 점과 기여율 구간을 그림으로 보며 조정"); bp.clicked.connect(self._open_scan_preview)
        c.header_extra.addWidget(bp)
        fv.addWidget(c)

        c = Card("Case 3 (3D) · XY 래스터", "x 에 직교하는 y 방향을 더한 XY 전체 스캔")
        c.add(hint(f"Case 2 의 스캔 폭, 스캔 간격, 기여율 구간을 x, y 양방향에 그대로 씁니다. "
                   f"레이저-PET 거리 r = √(x² + y²) 로 묶고, 반경 {pl.CASE3_R_MAX:g} mm 원 안만 합산합니다."))
        bp = button("미리보기", "Ghost", "래스터 점과 r 구간을 그림으로 보며 조정"); bp.clicked.connect(self._open_scan_preview)
        c.header_extra.addWidget(bp)
        fv.addWidget(c)
        fv.addStretch(1)
        block_wheel(host)                     # 휠로 내리다가 값이 바뀌지 않게
        scroll.setWidget(host)
        lv.addWidget(scroll, 1)

        runbox = QtWidgets.QFrame(); runbox.setObjectName("Card")
        rb = QtWidgets.QVBoxLayout(runbox)
        rb.setContentsMargins(12, 10, 12, 12)
        self.btn_run = button("전체 실행", "Primary", "플루언스 → Case 1, 2, 3, 조직 배경")
        self.btn_run.clicked.connect(lambda: self.run(["measured", "case1", "case2", "case3", "bg"]))
        rb.addWidget(self.btn_run)
        row = QtWidgets.QHBoxLayout()
        self.btn_fl = button("플루언스 다시 계산")
        self.btn_fl.clicked.connect(lambda: self.run(["fluence", "fluence_force"]))
        self.btn_save = button("결과 저장 (CSV, PNG)")
        self.btn_save.clicked.connect(self.save_results)
        row.addWidget(self.btn_fl, 1); row.addWidget(self.btn_save, 1)
        rb.addLayout(row)
        lv.addWidget(runbox)
        split.addWidget(left)

        # ---- 오른쪽: 그림 + 로그
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)
        self.case_seg = SegmentedControl(CASES, 0)
        self.case_seg.changed.connect(self._case_changed)
        rv.addWidget(self.case_seg)
        sel = QtWidgets.QHBoxLayout()
        sel.addWidget(muted("그림"))
        self.plot_kind = QtWidgets.QComboBox(); self.plot_kind.addItems(CASE_PLOTS[0])
        self.plot_depth = QtWidgets.QComboBox()
        self.plot_kind.currentIndexChanged.connect(self.redraw)
        self.plot_depth.currentIndexChanged.connect(self.redraw)
        sel.addWidget(self.plot_kind, 2)
        sel.addWidget(muted("깊이 (mm)"))
        sel.addWidget(self.plot_depth, 1)
        rv.addLayout(sel)

        vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.canvas = MplCanvas(paper=True)
        self.canvas.message("설정을 확인하고 '전체 실행'을 누르세요.")
        frame = PlotFrame(self.canvas, "raman_pattern.csv")
        frame.csv_btn.hide()                     # 결과 CSV 는 왼쪽 '결과 저장' 이 담당
        vsplit.addWidget(frame)
        self.logbox = QtWidgets.QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setStyleSheet("font-family: Consolas, 'Malgun Gothic'; font-size: 9pt;")
        vsplit.addWidget(self.logbox)
        vsplit.setStretchFactor(0, 4)
        vsplit.setStretchFactor(1, 1)
        rv.addWidget(vsplit, 1)
        split.addWidget(right)
        split.setStretchFactor(1, 1)

    def _log(self, msg):
        self.logbox.appendPlainText(msg)

    def _case_changed(self, i):
        self.plot_kind.blockSignals(True)          # 목록을 바꾸는 동안 redraw 가 여러 번 불리지 않게
        self.plot_kind.clear()
        self.plot_kind.addItems(CASE_PLOTS[i])
        self.plot_kind.blockSignals(False)
        self.redraw()

    def _pick_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "점 측정 CSV 폴더", pl._abs(self.cfg, self.meas_dir.text()))
        if d:
            self.meas_dir.setText(d)

    def _open_preview(self):
        """기하·CCD 미리보기 팝업. 폼의 현재 값으로 열고, '확인'이면 바뀐 값을 폼에 넣는다"""
        try:
            xext, yext, depths = _list(self.xext.text()), _list(self.yext.text()), _list(self.depths.text())
        except ValueError:
            xext = []
        if len(xext) != 2 or len(yext) != 2 or not depths:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "x·y 범위는 숫자 2개 (a, b), 깊이는 숫자 1개 이상이어야 합니다.")
            return
        vals = dict(voxel_mm=self.voxel.value(), x_extent_mm=xext, y_extent_mm=yext,
                    tissue_mm=self.tissue_mm.value(), cube_mm=self.cube_mm.value(), source=self.src.currentText(),
                    na=self.na.value(), n_channels=self.nch.value(), channel_mm=self.ch_mm.value(),
                    laser_channel=self.laser_ch.value())
        dlg = GeometryPreviewDialog(vals, [int(d) if d.is_integer() else d for d in depths],
                                    self.cfg["geometry"]["air_mm"], self.scan_mm.value(), self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        v = dlg.values()
        self.voxel.setValue(v["voxel_mm"])
        self.xext.setText(", ".join(f"{a:g}" for a in v["x_extent_mm"]))  # :g → 23.35 처럼 불필요한 0 없이
        self.yext.setText(", ".join(f"{a:g}" for a in v["y_extent_mm"]))
        self.tissue_mm.setValue(v["tissue_mm"]); self.cube_mm.setValue(v["cube_mm"])
        self.src.setCurrentText(v["source"]); self.na.setValue(v["na"])
        self.nch.setValue(v["n_channels"]); self.ch_mm.setValue(v["channel_mm"])
        self.laser_ch.setValue(v["laser_channel"])
        self._log("미리보기에서 기하·CCD 설정을 폼에 반영했습니다 (저장하려면 '저장').")

    def _open_scan_preview(self):
        """Case 2·3 스캔 미리보기 팝업. 폼의 현재 값으로 열고, '확인'이면 바뀐 값을 폼에 넣는다"""
        try:
            sides, xext, yext = _list(self.sides.text()), _list(self.xext.text()), _list(self.yext.text())
        except ValueError:
            xext = []
        if len(xext) != 2 or len(yext) != 2:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "레이저 옆 위치는 숫자 목록, x·y 범위는 숫자 2개 (a, b)여야 합니다.")
            return
        vals = dict(laser_beside_mm=sides, scan_mm=self.scan_mm.value(), step_mm=self.step_mm.value(),
                    bins=self.bins.ranges())
        dlg = ScanPreviewDialog(vals, self.cube_mm.value(), xext, yext, pl.CASE3_R_MAX,
                                pl.contribution_bin_labels, pl.scan_positions, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        v = dlg.values()
        self.sides.setText(", ".join(f"{a:g}" for a in v["laser_beside_mm"]))
        self.scan_mm.setValue(v["scan_mm"]); self.step_mm.setValue(v["step_mm"])
        if v["bins"]:                                   # 구간을 다 지웠으면 기존 구간 유지 (_form_to_cfg 와 같은 규칙)
            self.bins.set_ranges(v["bins"])
        self._log("미리보기에서 Case 2·3 스캔 설정을 폼에 반영했습니다 (저장하려면 '저장').")

    # ---------------------------------------------------------------- 설정 ↔ 폼
    def _cfg_to_form(self):
        c = self.cfg
        self.mua.setValue(c["medium"]["mua"]); self.mus.setValue(c["medium"]["mus"])
        self.g_.setValue(c["medium"]["g"]); self.n_.setValue(c["medium"]["n"])
        self.voxel.setValue(c["geometry"]["voxel_mm"])
        self.xext.setText(", ".join(str(v) for v in c["geometry"]["x_extent_mm"]))
        self.yext.setText(", ".join(str(v) for v in c["geometry"]["y_extent_mm"]))
        self.tissue_mm.setValue(c["geometry"]["tissue_mm"]); self.cube_mm.setValue(c["geometry"]["cube_mm"])
        self.src.setCurrentText(c["source"]["type"]); self.na.setValue(c["source"]["na"])
        self.nphoton.setValue(int(c["source"]["nphoton"]))
        self.nch.setValue(c["ccd"]["n_channels"]); self.ch_mm.setValue(c["ccd"]["channel_mm"])
        self.laser_ch.setValue(c["ccd"]["laser_channel"])
        self.depths.setText(", ".join(str(v) for v in c["case1"]["depths_mm"]))
        self.sides.setText(", ".join(str(v) for v in c["case2"]["laser_beside_mm"]))
        self.scan_mm.setValue(c["case2"]["scan_mm"]); self.step_mm.setValue(c["case2"]["step_mm"])
        self.bins.set_ranges(c["case2"]["contribution_bins_mm"])
        self.meas_dir.setText(c["point_data"]["folder"])
        self.meas_laser.setValue(c["point_data"]["laser_channel"])
        self.frames.setValue(c["point_data"]["frames_per_batch"])
        self.out_dir.setText(c.get("output_dir", "results/app"))

    def _form_to_cfg(self):
        c = self.cfg
        c["medium"].update(mua=self.mua.value(), mus=self.mus.value(), g=self.g_.value(), n=self.n_.value())
        c["geometry"].update(voxel_mm=self.voxel.value(), x_extent_mm=_list(self.xext.text()),
                             y_extent_mm=_list(self.yext.text()), tissue_mm=self.tissue_mm.value(),
                             cube_mm=self.cube_mm.value())
        c["source"].update(type=self.src.currentText(), na=self.na.value(), nphoton=int(self.nphoton.value()))
        c["ccd"].update(n_channels=self.nch.value(), channel_mm=self.ch_mm.value(),
                        laser_channel=self.laser_ch.value())
        c["case1"]["depths_mm"] = [int(v) if float(v).is_integer() else v for v in _list(self.depths.text())]
        c["case2"]["laser_beside_mm"] = [int(v) if float(v).is_integer() else v for v in _list(self.sides.text())]
        c["case2"].update(scan_mm=self.scan_mm.value(), step_mm=self.step_mm.value())
        bins = self.bins.ranges()
        if bins:
            c["case2"]["contribution_bins_mm"] = bins
        c["point_data"].update(folder=self.meas_dir.text(), laser_channel=self.meas_laser.value(),
                               frames_per_batch=self.frames.value())
        c["output_dir"] = self.out_dir.text()
        return c

    def load_cfg(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "설정 파일", os.path.dirname(self.cfg_path.text()),
                                                     "JSON (*.json)")
        if p:
            self.cfg = pl.load_config(p)
            self.cfg_path.setText(p)
            self._cfg_to_form()
            self._log(f"설정 불러옴: {p}")

    def save_cfg(self):
        self._form_to_cfg()
        self._log(f"설정 저장: {pl.save_config(self.cfg, self.cfg_path.text())}")

    # ---------------------------------------------------------------- 실행
    def run(self, steps):
        if self.worker and self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "계산 중", "이미 계산 중입니다. 끝난 뒤 다시 실행하세요.")
            return
        self._form_to_cfg()
        pl.save_config(self.cfg, self.cfg_path.text())
        if not self.use_meas.isChecked() and "measured" in steps:
            steps = [s for s in steps if s != "measured"]
            self.res.pop("measured", None)
        tag = pl.fluence_tag(self.cfg)
        if self.res.get("tag") != tag:          # 계수, 격자, 광원이 바뀌면 플루언스 캐시를 버림
            self.res.pop("F", None)
            self.res["tag"] = tag
        self.btn_run.setEnabled(False)
        self.btn_fl.setEnabled(False)
        cfg = json.loads(json.dumps({k: v for k, v in self.cfg.items() if not k.startswith("_")}))
        self.worker = Worker(cfg, steps, self.res)
        self.worker.log.connect(self._log)
        self.worker.done.connect(self._done)
        self.worker.error.connect(self._error)
        self.worker.start()
        self._log(f"실행: {', '.join(steps)}")

    def _done(self, res):
        self.res.update(res)
        self.state["raman_pattern"] = self.res
        self.btn_run.setEnabled(True)
        self.btn_fl.setEnabled(True)
        depths = (self.res.get("r1", {}).get("depths") or self.res.get("r2", {}).get("depths")
                  or self.res.get("r3", {}).get("depths") or [])
        cur = self.plot_depth.currentText()
        self.plot_depth.blockSignals(True)
        self.plot_depth.clear()
        self.plot_depth.addItems([str(d) for d in depths])
        if cur in [str(d) for d in depths]:
            self.plot_depth.setCurrentText(cur)
        self.plot_depth.blockSignals(False)
        self._log("완료")
        self.redraw()

    def _error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_fl.setEnabled(True)
        self._log("오류: " + msg)
        QtWidgets.QMessageBox.critical(self, "계산 오류", msg.splitlines()[0])

    # ---------------------------------------------------------------- 그림
    def redraw(self):
        case, kind = self.case_seg.current(), self.plot_kind.currentIndex()
        r1, r2, r3, bg = self.res.get("r1"), self.res.get("r2"), self.res.get("r3"), self.res.get("bg")
        d = self._depth()
        fig = self.canvas.fig
        try:
            with self.canvas.style():
                if (case, kind) == (0, 0) and r1:
                    pl.plot_case1(fig, r1, self.cfg)
                elif (case, kind) == (0, 1) and r1 and d is not None:
                    pl.plot_case1(fig, r1, self.cfg, depth=d)
                elif (case, kind) == (1, 0) and r2 and d is not None:
                    pl.plot_case2_patterns(fig, r2, d)
                elif (case, kind) == (1, 1) and r2 and d is not None:
                    pl.plot_case2_contribution(fig, r2, d)
                elif (case, kind) == (2, 0) and r3 and d is not None:
                    pl.plot_case3_contribution(fig, r3, d)
                elif case == 3 and bg:
                    m = self.res.get("measured")
                    pl.plot_background(fig, bg, m["tissue_norm"] if m else None)
                else:
                    return
            self.canvas.draw_idle()
        except Exception as e:  # noqa: BLE001
            self._log(f"그림 오류: {e}")

    def _depth(self):
        t = self.plot_depth.currentText()
        if not t:
            return None
        v = float(t)
        return int(v) if v.is_integer() else v

    # ---------------------------------------------------------------- 저장
    def save_results(self):
        if not self.res.get("r1") and not self.res.get("r2") and not self.res.get("r3"):
            QtWidgets.QMessageBox.information(self, "저장할 결과 없음", "먼저 '전체 실행'을 하세요.")
            return
        self._form_to_cfg()
        out = pl._abs(self.cfg, self.cfg["output_dir"])
        pl.save_results(self.cfg, self.res.get("r1"), self.res.get("r2"), self.res.get("bg"), out_dir=out, log=self._log,
                        r3=self.res.get("r3"))
        pl.save_figures(self.cfg, self.res.get("r1"), self.res.get("r2"), self.res.get("bg"), out_dir=out, log=self._log,
                        r3=self.res.get("r3"))
        pl.save_config(self.cfg, os.path.join(out, "config_used.json"))
        self._log(f"설정 사본: {os.path.join(out, 'config_used.json')}")

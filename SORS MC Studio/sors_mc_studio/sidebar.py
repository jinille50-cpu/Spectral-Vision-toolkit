"""왼쪽 사이드바: 무엇을 시뮬레이션할지 (바꾸면 다시 실행해야 반영되는 설정)."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from .core.presets import get_preset, get_preset_names
from .widgets import Card, SegmentedControl, StackPreview, button, dspin, form_row, hint, muted

SOURCE_KEYS = ['pencil', 'gaussian', 'flattop', 'line']
SOURCE_LABELS = ['펜슬', '가우시안', '원형 균일', '라인']


class LayerRow(QtWidgets.QFrame):
    removed = QtCore.Signal(object)
    changed = QtCore.Signal()

    def __init__(self, idx: int, parent=None):
        super().__init__(parent)
        self.setObjectName("LayerRow")
        g = QtWidgets.QGridLayout(self)
        g.setContentsMargins(10, 8, 10, 8)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)

        self.chip = QtWidgets.QLabel()
        self.chip.setFixedSize(10, 28)
        g.addWidget(self.chip, 0, 0)
        self.name = QtWidgets.QLineEdit("Tissue")
        g.addWidget(self.name, 0, 1, 1, 3)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(get_preset_names())
        self.preset.setToolTip("문헌값 불러오기")
        g.addWidget(self.preset, 0, 4, 1, 2)
        self.rm = QtWidgets.QToolButton()
        self.rm.setText("✕")
        self.rm.setToolTip("이 층 삭제")
        self.rm.setAutoRaise(True)
        self.rm.clicked.connect(lambda: self.removed.emit(self))
        g.addWidget(self.rm, 0, 6)

        self.thick = dspin(0.1, 50, 5.0, 2, 0.5)
        self.mua = dspin(0, 10, 0.03, 4, 0.001)
        self.mus = dspin(0, 100, 10.0, 2, 0.1)
        self.gg = dspin(-1, 1, 0.9, 3, 0.01)
        self.n = dspin(1, 2, 1.40, 3, 0.01)
        # 값 칸: 두께 / μa / μs 한 줄, g / n 다음 줄 (사이드바 폭 안에 들어가게)
        fields = [("두께 mm", self.thick), ("μa /mm", self.mua), ("μs /mm", self.mus),
                  ("g", self.gg), ("n", self.n)]
        for k, (lab, w) in enumerate(fields):
            r, c = (1, k) if k < 3 else (3, k - 3)
            l = muted(lab)
            l.setStyleSheet("font-size: 8.5pt;")
            g.addWidget(l, r, 1 + 2 * c, 1, 2)
            w.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            g.addWidget(w, r + 1, 1 + 2 * c, 1, 2)
        self.musp = hint("")
        g.addWidget(self.musp, 4, 5, 1, 2, alignment=QtCore.Qt.AlignVCenter)

        self.preset.currentTextChanged.connect(self._on_preset)
        for w in (self.thick, self.mua, self.mus, self.gg, self.n):
            w.valueChanged.connect(self._changed)
        self.name.textChanged.connect(self._changed)
        self.set_index(idx)
        self._changed()

    def set_index(self, idx):
        self.idx = idx
        self.chip.setStyleSheet(f"background:{theme.layer_color(idx)}; border-radius:3px;")

    def _on_preset(self, name):
        p = get_preset(name)
        for w, k in ((self.mua, 'mua'), (self.mus, 'mus'), (self.gg, 'g'), (self.n, 'n')):
            w.blockSignals(True)
            w.setValue(p[k])
            w.blockSignals(False)
        if name != "Custom":
            self.name.setText(name.split(' (')[0])
        self._changed()

    def _changed(self):
        self.musp.setText(f"μs′ {self.mus_prime():.3f} /mm")
        self.musp.setToolTip(f"μs′ = μs(1−g) = {self.mus_prime() * 10:.2f} /cm")
        self.changed.emit()

    def mus_prime(self):
        return self.mus.value() * (1 - self.gg.value())

    def data(self):
        return dict(name=self.name.text() or f"Layer{self.idx + 1}", thickness=self.thick.value(),
                    mua=self.mua.value(), mus=self.mus.value(), g=self.gg.value(), n=self.n.value())


class Sidebar(QtWidgets.QWidget):
    changed = QtCore.Signal()
    run_clicked = QtCore.Signal()
    stop_clicked = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        self.setMinimumWidth(430)
        self.setMaximumWidth(500)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        host = QtWidgets.QWidget()
        host.setObjectName("Transparent")
        col = QtWidgets.QVBoxLayout(host)
        col.setContentsMargins(0, 0, 4, 0)
        col.setSpacing(8)

        # ---- 매질
        self.medium = Card("매질", "표면(z = 0)에서 아래로 층이 쌓입니다.")
        self.preview = StackPreview()
        self.medium.add(self.preview)
        self.rows_box = QtWidgets.QVBoxLayout()
        self.rows_box.setSpacing(6)
        self.medium.add(self.rows_box)
        add = button("층 추가")
        add.clicked.connect(self.add_layer)
        self.medium.add(add)
        col.addWidget(self.medium)

        # ---- 광원
        self.source = Card("광원", "모든 광원은 표면에 수직으로 입사합니다.")
        self.src_seg = SegmentedControl(SOURCE_LABELS, 0)
        self.source.add(self.src_seg)
        p0 = QtWidgets.QWidget(); l0 = QtWidgets.QVBoxLayout(p0); l0.setContentsMargins(0, 0, 0, 0)
        l0.addWidget(hint("한 점(원점)에 입사하는 기본 광원입니다."))
        p1 = QtWidgets.QWidget(); l1 = QtWidgets.QVBoxLayout(p1); l1.setContentsMargins(0, 0, 0, 0)
        self.gauss_d = dspin(0, 10, 0.5, 3, 0.1, " mm")
        l1.addLayout(form_row("1/e² 직경", self.gauss_d))
        p2 = QtWidgets.QWidget(); l2 = QtWidgets.QVBoxLayout(p2); l2.setContentsMargins(0, 0, 0, 0)
        self.flat_d = dspin(0, 10, 0.5, 3, 0.1, " mm")
        l2.addLayout(form_row("직경", self.flat_d))
        p3 = QtWidgets.QWidget(); l3 = QtWidgets.QVBoxLayout(p3); l3.setContentsMargins(0, 0, 0, 0)
        self.line_len = dspin(0, 20, 4.0, 2, 0.5, " mm")
        self.line_w = dspin(0, 5, 0.1, 3, 0.05, " mm")
        self.line_orient = SegmentedControl(["offset 축과 평행", "수직"], 0)
        self.line_prof = SegmentedControl(["폭 균일", "폭 가우시안"], 0)
        l3.addLayout(form_row("길이", self.line_len))
        l3.addLayout(form_row("폭", self.line_w))
        l3.addLayout(form_row("방향", self.line_orient))
        l3.addLayout(form_row("폭 방향 분포", self.line_prof))
        l3.addWidget(hint("폭 가우시안이면 폭은 1/e² 전폭입니다."))
        self.src_pages = [p0, p1, p2, p3]
        for k, p in enumerate(self.src_pages):
            p.setObjectName("Transparent")
            p.setVisible(k == 0)
            self.source.add(p)
        self.src_seg.changed.connect(self._on_source)
        col.addWidget(self.source)

        # ---- 검출기
        self.det = Card("검출기", "라인스캔 CCD 한 줄: 광원을 지나는 x축 위의 띠, 채널 간격 0.2 mm.")
        self.max_offset = dspin(1, 30, 8.0, 1, 0.5, " mm")
        self.strip = dspin(0.05, 5, 0.3, 2, 0.05, " mm")
        self.det.add(form_row("최대 offset", self.max_offset))
        self.det.add(form_row("띠 폭 (y 방향)", self.strip))
        col.addWidget(self.det)

        # ---- 실행 설정
        self.runcfg = Card("실행 설정")
        self.nph = QtWidgets.QSpinBox()
        self.nph.setRange(1000, 100_000_000)
        self.nph.setSingleStep(100_000)
        self.nph.setValue(1_000_000)
        self.nph.setGroupSeparatorShown(True)
        self.nph.setAlignment(QtCore.Qt.AlignRight)
        self.npaths = QtWidgets.QSpinBox()
        self.npaths.setRange(0, 5000)
        self.npaths.setValue(100)
        self.npaths.setAlignment(QtCore.Qt.AlignRight)
        self.runcfg.add(form_row("광자 수", self.nph))
        self.runcfg.add(form_row("표본 궤적 수", self.npaths))
        self.runcfg.add(hint("GPU(PMCX) 기준 100만 광자 약 35초. 먼 offset 통계에는 수백만 광자를 권합니다."))
        col.addWidget(self.runcfg)
        col.addStretch(1)
        scroll.setWidget(host)
        outer.addWidget(scroll, 1)

        # ---- 실행 버튼 (스크롤 밖, 항상 보임)
        runbox = QtWidgets.QFrame()
        runbox.setObjectName("Card")
        rb = QtWidgets.QVBoxLayout(runbox)
        rb.setContentsMargins(12, 10, 12, 12)
        rb.setSpacing(6)
        self.stale = hint("", "Warn")
        self.stale.setVisible(False)
        rb.addWidget(self.stale)
        btns = QtWidgets.QHBoxLayout()
        self.run_btn = button("시뮬레이션 실행", "Primary")
        self.run_btn.setShortcut(QtGui.QKeySequence("Ctrl+Return"))
        self.run_btn.setToolTip("Ctrl+Enter")
        self.run_btn.clicked.connect(self.run_clicked)
        self.stop_btn = button("중지", "Danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_clicked)
        btns.addWidget(self.run_btn, 1)
        btns.addWidget(self.stop_btn)
        rb.addLayout(btns)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        rb.addWidget(self.progress)
        self.progress_label = hint("준비됨")
        rb.addWidget(self.progress_label)
        outer.addWidget(runbox)

        for w in (self.max_offset, self.strip, self.nph, self.gauss_d, self.flat_d,
                  self.line_len, self.line_w):
            w.valueChanged.connect(self._emit)
        for s in (self.line_orient, self.line_prof):
            s.changed.connect(self._emit)
        self.add_layer()

    # ------------------------------------------------------------------
    def _emit(self, *_):
        self._update_preview()
        self.changed.emit()

    def _on_source(self, i):
        for k, p in enumerate(self.src_pages):
            p.setVisible(k == i)
        self._emit()

    def add_layer(self):
        row = LayerRow(len(self.rows))
        row.removed.connect(self.remove_layer)
        row.changed.connect(self._emit)
        self.rows.append(row)
        self.rows_box.addWidget(row)
        self._emit()

    def remove_layer(self, row):
        if len(self.rows) <= 1:
            QtWidgets.QMessageBox.information(self, "층을 지울 수 없음", "매질에는 층이 하나 이상 있어야 합니다.")
            return
        self.rows.remove(row)
        self.rows_box.removeWidget(row)
        row.deleteLater()
        for i, r in enumerate(self.rows):
            r.set_index(i)
        self._emit()

    def source_label(self):
        k = self.src_seg.current()
        if k == 1:
            return f"가우시안 1/e² {self.gauss_d.value():g} mm"
        if k == 2:
            return f"원형 균일 {self.flat_d.value():g} mm"
        if k == 3:
            return f"라인 {self.line_len.value():g}×{self.line_w.value():g} mm"
        return "펜슬"

    def _update_preview(self):
        layers = [(r.data()['name'], r.data()['thickness'], r.mus_prime(), theme.layer_color(i))
                  for i, r in enumerate(self.rows)]
        self.preview.set_stack(layers, self.max_offset.value(), self.source_label())

    def total_thickness(self):
        return sum(r.thick.value() for r in self.rows)

    def set_stale(self, stale: bool):
        self.stale.setText("설정이 바뀌었습니다. 다시 실행해야 그래프에 반영됩니다.")
        self.stale.setVisible(stale)

    def get_config(self):
        from .core.monte_carlo import Layer, SimulationConfig
        layers = [Layer(**r.data()) for r in self.rows]
        n_paths = self.npaths.value()
        k = self.src_seg.current()
        return SimulationConfig(
            layers=layers,
            n_photons=self.nph.value(),
            detector_radius=self.strip.value(),
            max_offset=self.max_offset.value(),
            save_paths=n_paths > 0,
            n_paths_to_save=n_paths,
            source_type=SOURCE_KEYS[k],
            source_diameter=self.gauss_d.value() if k == 1 else self.flat_d.value(),
            line_length=self.line_len.value(),
            line_width=self.line_w.value(),
            line_orient=['parallel', 'perpendicular'][self.line_orient.current()],
            line_profile=['uniform', 'gaussian'][self.line_prof.current()],
        )

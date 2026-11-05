"""우측 상단 그래프 영역: 헤더의 전환 버튼으로 포인트 스펙트럼 / 클러스터 스펙트럼을 번갈아 표시."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .common import SegmentedControl
from .spectrum_view import ClusterSpectrumView, SpectrumView


class PlotArea(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.spectrum = SpectrumView()
        self.cluster = ClusterSpectrumView()
        # 두 뷰의 자체 헤더 대신 공용 헤더 사용
        for v in (self.spectrum, self.cluster):
            v.layout().itemAt(0).layout().setContentsMargins(12, 2, 12, 0)
            v.title.hide()

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(12, 8, 12, 0)
        self.switch = SegmentedControl(["포인트 스펙트럼", "클러스터 스펙트럼"], 0)
        self.switch.setFixedWidth(300)
        self.switch.setEnabledIndex(1, False)
        header.addWidget(self.switch)
        header.addStretch(1)
        lay.addLayout(header)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.spectrum)
        self.stack.addWidget(self.cluster)
        lay.addWidget(self.stack, 1)
        self.switch.changed.connect(self.stack.setCurrentIndex)

    def show_cluster(self, available: bool, switch_to: bool = False):
        self.switch.setEnabledIndex(1, available)
        if available and switch_to:
            self.switch.setCurrent(1)
            self.stack.setCurrentIndex(1)
        elif not available:
            self.switch.setCurrent(0)
            self.stack.setCurrentIndex(0)

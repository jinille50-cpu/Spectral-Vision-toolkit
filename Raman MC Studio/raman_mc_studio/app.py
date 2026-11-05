"""메인 창: 라만 패턴 작업 공간(RamanView) 하나만 띄운다."""
from __future__ import annotations

from PySide6 import QtWidgets

from . import __version__, theme
from .raman_view import RamanView


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Raman MC Studio {__version__}")
        self.resize(1720, 1000)

        self.raman = RamanView({})
        central = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(central)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.addWidget(self.raman)
        self.setCentralWidget(central)

        self._build_toolbar()
        sb = self.statusBar()
        sb.addWidget(QtWidgets.QLabel(f"설정: {self.raman.cfg['_path']}"))

    def _build_toolbar(self):
        tb = QtWidgets.QToolBar("main")
        tb.setMovable(False)
        self.addToolBar(tb)
        title = QtWidgets.QLabel("  Raman MC Studio  ")
        title.setStyleSheet(f"font-weight: 600; font-size: 11pt; color: {theme.TEXT};")
        tb.addWidget(title)

"""
SORS Monte Carlo Simulator
===========================
다층 매질에서의 광 전파를 Monte Carlo로 시뮬레이션하여
SORS (Spatially Offset Raman Spectroscopy) 측정의 광학적 거동을 예측.

기능:
  - 다층 매질 설정 (preset: 고기, PET, 지방 등)
  - 광자 추적 시뮬레이션
  - 3D photon path 시각화
  - Offset profile + depth distribution
  - 결과 저장 (npz, csv)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt

from tabs.input_tab import InputTab
from tabs.simulation_tab import SimulationTab
from tabs.export_tab import ExportTab


STYLE = """
QMainWindow, QWidget { background-color: #f9fafb; }
QTabWidget::pane {
    border: 1px solid #d1d5db; border-radius: 6px;
    background: white; top: -1px;
}
QTabBar::tab {
    background: #e5e7eb; padding: 10px 22px; margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    font-size: 13px; font-weight: 500; color: #374151;
}
QTabBar::tab:selected {
    background: white; border-bottom: 3px solid #7c3aed; color: #7c3aed;
}
QGroupBox {
    font-weight: bold; border: 1px solid #d1d5db; border-radius: 6px;
    margin-top: 10px; padding-top: 10px; background: white;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 8px;
    color: #1f2937;
}
QPushButton {
    background-color: #7c3aed; color: white; border: none;
    padding: 8px 16px; border-radius: 4px; font-weight: 500;
}
QPushButton:hover { background-color: #6d28d9; }
QPushButton:disabled { background-color: #9ca3af; }
QPushButton#danger {
    background-color: #ef4444;
}
QPushButton#danger:hover { background-color: #dc2626; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    padding: 6px 10px; border: 1px solid #d1d5db; border-radius: 4px;
    background: white; min-height: 18px;
}
QLineEdit:focus, QComboBox:focus { border: 2px solid #7c3aed; }
QLabel#hint {
    color: #6b7280; font-size: 11px; font-weight: normal;
}
QProgressBar {
    border: 1px solid #d1d5db; border-radius: 4px; text-align: center;
    background: #f3f4f6;
}
QProgressBar::chunk { background-color: #7c3aed; border-radius: 3px; }
QTextEdit {
    background: white; border: 1px solid #d1d5db; border-radius: 4px;
    padding: 4px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SORS Monte Carlo Simulator")
        self.resize(1400, 900)

        # 전역 state (탭 간 공유)
        self.state = {
            'last_result': None,    # 최근 시뮬레이션 결과
        }

        # 탭
        self.tabs = QTabWidget()
        self.input_tab = InputTab(self.state)
        self.sim_tab = SimulationTab(self.state, get_config_callback=self.input_tab.get_simulation_config)
        self.export_tab = ExportTab(self.state)

        self.tabs.addTab(self.input_tab, "① Input (Medium + Source)")
        self.tabs.addTab(self.sim_tab, "② Simulation & Visualization")
        self.tabs.addTab(self.export_tab, "③ Export")

        # 중앙 위젯
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        # Status bar: backend 정보
        from core.monte_carlo import get_backend_info
        info = get_backend_info()
        status_msg = (f"MC Backend: {info['backend']} | "
                      f"{info['description']} | "
                      f"{info['reference']}")
        self.statusBar().showMessage(status_msg)


def _setup_matplotlib_font():
    try:
        import matplotlib
        for font in ["Malgun Gothic", "AppleGothic", "NanumGothic",
                     "DejaVu Sans", "sans-serif"]:
            try:
                matplotlib.rcParams['font.family'] = font
                matplotlib.rcParams['axes.unicode_minus'] = False
                break
            except Exception:
                continue
    except ImportError:
        pass


def main():
    _setup_matplotlib_font()
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)

    from PyQt5.QtGui import QFont
    f = QFont("Malgun Gothic", 9)
    app.setFont(f)

    def excepthook(exctype, value, traceback):
        import traceback as tb
        msg = ''.join(tb.format_exception(exctype, value, traceback))
        print(f"[ERROR]\n{msg}", file=sys.stderr)
        sys.__excepthook__(exctype, value, traceback)
    sys.excepthook = excepthook

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

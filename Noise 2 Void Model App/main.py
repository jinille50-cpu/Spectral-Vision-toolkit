"""
RamanN2V - PyQt5 application for Noise2Void denoising of Raman hyperspectral data.
"""
import sys
import traceback
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QMessageBox, QWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt

from tabs.data_tab import DataTab
from tabs.train_tab import TrainTab
from tabs.predict_tab import PredictTab


STYLE_SHEET = """
QWidget {
    background-color: #ffffff;
    color: #1f2937;
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
    font-size: 10pt;
}
QTabWidget::pane {
    border: 1px solid #e5e7eb;
    background: #ffffff;
}
QTabBar::tab {
    background: #f3f4f6;
    color: #4b5563;
    padding: 8px 18px;
    border: 1px solid #e5e7eb;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
}
QPushButton {
    background-color: #2563eb;
    color: white;
    border: none;
    padding: 6px 14px;
    border-radius: 4px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #1d4ed8;
}
QPushButton:disabled {
    background-color: #9ca3af;
}
QPushButton#secondary {
    background-color: #ffffff;
    color: #2563eb;
    border: 1px solid #2563eb;
}
QPushButton#secondary:hover {
    background-color: #eff6ff;
}
QGroupBox {
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
    color: #374151;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 4px 8px;
    background: #ffffff;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #2563eb;
}
QTextEdit, QListWidget {
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    background: #ffffff;
}
QProgressBar {
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    text-align: center;
    background: #f9fafb;
}
QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 3px;
}
QLabel#section_title {
    font-size: 11pt;
    font-weight: 600;
    color: #111827;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RamanN2V - Noise2Void for Raman Hyperspectral Data")
        self.resize(1200, 800)

        # Shared state across tabs
        self.state = {
            "patch_dir": None,        # 패치 저장 폴더
            "model_path": None,       # 학습된 모델 경로
            "n_bands": None,          # 스펙트럼 밴드 수
            "mode": "2D",             # 2D or 3D
            "patch_size": 128,
        }

        tabs = QTabWidget()
        self.data_tab = DataTab(self.state)
        self.train_tab = TrainTab(self.state)
        self.predict_tab = PredictTab(self.state)

        # 탭 간 참조 연결 (학습 후 모델 경로 전달 등)
        self.data_tab.set_train_tab(self.train_tab)
        self.train_tab.set_predict_tab(self.predict_tab)

        tabs.addTab(self.data_tab, "1. Data")
        tabs.addTab(self.train_tab, "2. Train")
        tabs.addTab(self.predict_tab, "3. Predict")
        self.setCentralWidget(tabs)


def _setup_exception_hook():
    """예외 발생 시 앱이 죽지 않게 메시지로 표시"""
    def hook(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(msg, file=sys.stderr)
        try:
            QMessageBox.critical(None, "Error", str(exc_value))
        except Exception:
            pass
    sys.excepthook = hook


def _setup_matplotlib_font():
    try:
        import matplotlib
        for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
            try:
                matplotlib.rcParams['font.family'] = font
                matplotlib.rcParams['axes.unicode_minus'] = False
                break
            except Exception:
                continue
    except Exception:
        pass


def main():
    _setup_exception_hook()
    _setup_matplotlib_font()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

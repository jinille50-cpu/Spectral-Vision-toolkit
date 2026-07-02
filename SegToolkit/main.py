import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor

from tabs.data_tab import DataTab
from tabs.model_tab import ModelTab
from tabs.train_tab import TrainTab
from tabs.eval_tab import EvalTab
from tabs.infer_tab import InferTab


def _setup_matplotlib_font():
    """matplotlib 한글/영문 폰트 깨짐 방지"""
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
    except Exception:
        pass


def _setup_exception_hook():
    """Worker 스레드 예외가 앱 전체를 죽이지 않도록 전역 핸들러 등록"""
    import sys, traceback
    def handle_exception(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        print("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    sys.excepthook = handle_exception


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SegToolkit — Segmentation Training GUI")
        self.setMinimumSize(1100, 750)
        self.resize(1600, 900) 
        self.shared_state = {}  # 탭 간 공유 데이터
        self._apply_style()
        self._build_ui()

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #ffffff;
                color: #1a1a1a;
                font-family: 'Segoe UI', 'Apple SD Gothic Neo', sans-serif;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #f5f5f5;
                color: #555555;
                padding: 10px 24px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 500;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #1a1a1a;
                border-bottom: 2px solid #2563eb;
                font-weight: 700;
            }
            QTabBar::tab:hover:!selected {
                background: #ebebeb;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 7px 18px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
            QPushButton:disabled {
                background-color: #d1d5db;
                color: #9ca3af;
            }
            QPushButton#secondary {
                background-color: #f3f4f6;
                color: #374151;
                border: 1px solid #d1d5db;
            }
            QPushButton#secondary:hover {
                background-color: #e5e7eb;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 5px 8px;
                background: #ffffff;
                color: #1a1a1a;
                font-size: 13px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #2563eb;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QGroupBox {
                font-weight: 700;
                font-size: 13px;
                color: #374151;
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                background: #ffffff;
            }
            QListWidget {
                border: 1px solid #e5e7eb;
                border-radius: 4px;
                background: #fafafa;
                font-size: 12px;
            }
            QListWidget::item:selected {
                background: #dbeafe;
                color: #1e40af;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #e5e7eb;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #2563eb;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #2563eb;
                border-radius: 2px;
            }
            QCheckBox {
                spacing: 6px;
                font-size: 13px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #d1d5db;
                border-radius: 3px;
                background: white;
            }
            QCheckBox::indicator:checked {
                background: #2563eb;
                border-color: #2563eb;
            }
            QLabel#section_title {
                font-size: 14px;
                font-weight: 700;
                color: #111827;
            }
            QLabel#hint {
                font-size: 11px;
                color: #9ca3af;
            }
            QScrollArea {
                border: none;
                background: #ffffff;
            }
            QTextEdit {
                border: 1px solid #e5e7eb;
                border-radius: 4px;
                background: #fafafa;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
            QProgressBar {
                border: 1px solid #e5e7eb;
                border-radius: 4px;
                text-align: center;
                font-size: 12px;
                background: #f9fafb;
            }
            QProgressBar::chunk {
                background: #2563eb;
                border-radius: 3px;
            }
            QSplitter::handle {
                background: #e5e7eb;
                width: 1px;
            }
            QHeaderView::section {
                background: #f3f4f6;
                border: none;
                border-right: 1px solid #e5e7eb;
                padding: 6px 10px;
                font-weight: 600;
                font-size: 12px;
            }
            QTableWidget {
                border: 1px solid #e5e7eb;
                gridline-color: #f3f4f6;
                font-size: 12px;
            }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tab_data  = DataTab(self.shared_state)
        self.tab_model = ModelTab(self.shared_state)
        self.tab_train = TrainTab(self.shared_state)
        self.tab_eval  = EvalTab(self.shared_state)
        self.tab_infer = InferTab(self.shared_state)

        # 평가탭 → 추론탭 모델/데이터 공유 연결
        self.tab_eval.set_infer_tab(self.tab_infer)
        # 학습탭 · 모델탭 → 데이터탭 직접 참조 (저장 버튼 없이도 경로 읽기)
        self.tab_train.set_data_tab(self.tab_data)
        self.tab_model.set_data_tab(self.tab_data)

        self.tabs.addTab(self.tab_data,  "① 데이터")
        self.tabs.addTab(self.tab_model, "② 모델")
        self.tabs.addTab(self.tab_train, "③ 학습")
        self.tabs.addTab(self.tab_eval,  "④ 평가")
        self.tabs.addTab(self.tab_infer, "⑤ 추론")

        layout.addWidget(self.tabs)


def main():
    _setup_exception_hook()
    _setup_matplotlib_font()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

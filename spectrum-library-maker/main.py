"""
PolymerLib — Polymer Spectrum Classification Library
======================================================
스펙트럼 + 라벨로 폴리머 분류 라이브러리 구축

기능:
- 스펙트럼 txt 파일 불러오기
- 라벨 매핑 (세부 → 대분류)
- 데이터베이스 관리 (추가/삭제/내보내기/불러오기)
- 모델 학습 (SVM/KNN/RF/MLP) + K-Fold CV
- Confusion Matrix + 성능 지표
"""

import os
# DLL 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
# scikit-learn / joblib 멀티프로세싱이 새 콘솔창 띄우는 거 방지
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import sys
import multiprocessing as mp

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt

from tabs.data_tab import DataTab
from tabs.library_tab import LibraryTab
from tabs.train_tab import TrainTab
from tabs.predict_tab import PredictTab
from tabs.test_eval_tab import TestEvalTab


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
QPushButton#secondary {
    background-color: #f3f4f6; color: #374151; border: 1px solid #d1d5db;
}
QPushButton#secondary:hover { background-color: #e5e7eb; }
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
QTableWidget {
    background: white; border: 1px solid #d1d5db; border-radius: 4px;
    gridline-color: #e5e7eb;
}
QHeaderView::section {
    background-color: #f3f4f6; padding: 6px; border: none;
    border-right: 1px solid #d1d5db; font-weight: 600; color: #374151;
}
QTextEdit, QListWidget {
    background: white; border: 1px solid #d1d5db; border-radius: 4px;
    padding: 4px;
}
QProgressBar {
    border: 1px solid #d1d5db; border-radius: 4px; text-align: center;
    background: #f3f4f6;
}
QProgressBar::chunk { background-color: #7c3aed; border-radius: 3px; }
QCheckBox::indicator {
    width: 18px; height: 18px; border: 2px solid #d1d5db;
    border-radius: 3px; background: white;
}
QCheckBox::indicator:checked {
    background-color: #7c3aed; border-color: #7c3aed;
    image: none;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PolymerLib — Spectrum Classification Library")
        self.resize(1400, 900)

        # 전역 state (탭 간 공유)
        # state["database"]: list of dict — {'spectrum':[...], 'label':'PET', 'raw_label':'PET-001', 'source':'file.txt'}
        # state["wavelengths"]: 공통 wavelength axis (있으면)
        # state["label_mapping"]: {'PET-001': 'PET', 'PE-A': 'PE', ...}
        self.state = {
            "database": [],
            "wavelengths": None,
            "label_mapping": {},
            "trained_model": None,
            "model_meta": None,  # 학습 정보 (scaler, classes, model name 등)
        }

        # 탭 생성
        self.tabs = QTabWidget()
        self.data_tab     = DataTab(self.state)
        self.library_tab  = LibraryTab(self.state)
        self.train_tab    = TrainTab(self.state)
        self.predict_tab  = PredictTab(self.state)
        self.testeval_tab = TestEvalTab(self.state)

        self.tabs.addTab(self.data_tab,     "① 데이터 추가")
        self.tabs.addTab(self.library_tab,  "② 라이브러리")
        self.tabs.addTab(self.train_tab,    "③ 학습")
        self.tabs.addTab(self.testeval_tab, "④ 테스트 평가")
        self.tabs.addTab(self.predict_tab,  "⑤ 예측")

        # 탭 간 신호 연결
        self.data_tab.database_changed.connect(self.library_tab.refresh)
        self.data_tab.database_changed.connect(self.train_tab.refresh_db_info)
        self.library_tab.database_changed.connect(self.train_tab.refresh_db_info)

        self.train_tab.model_trained.connect(self.predict_tab.refresh_model_info)
        self.train_tab.model_trained.connect(self.testeval_tab.refresh_model_info)

        # 메인 위젯
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)


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

    # 한국어 폰트
    from PyQt5.QtGui import QFont
    f = QFont("Malgun Gothic", 9)
    app.setFont(f)

    # 예외 후킹 (PyQt에서 죽지 않게)
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
    # Windows에서 멀티프로세싱이 새 콘솔 띄우는 거 방지
    mp.freeze_support()
    main()

"""Graphite + teal 테마 (HSI Viewer 와 같은 색 토큰) — Qt 스타일시트와 matplotlib 스타일."""
from __future__ import annotations

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from PySide6 import QtGui, QtWidgets

# ---- 색 토큰 (HSI Viewer 와 동일) -------------------------------------------
BG0 = "#121418"      # 창 배경
BG1 = "#1a1d23"      # 패널
BG2 = "#22262e"      # 입력칸, 카드 안쪽
BG3 = "#2b303a"      # hover
BORDER = "#323845"
TEXT = "#e8eaf0"
MUTED = "#8d93a1"
ACCENT = "#4fd1c5"   # teal
ACCENT_DIM = "#2f7f78"
DANGER = "#f26d6d"
WARN = "#f5b84f"

# ---- 데이터 색 -----------------------------------------------------------------
# 층(조직) 색: 어두운 바탕 위에서 구분되되 데이터 선보다 튀지 않는 차분한 톤
LAYER_COLORS = ["#d98c8c", "#e2c27a", "#7fb2d9", "#a993d9", "#86cfa0", "#d48fc0"]
# offset 순서용 순차 램프 (가까운 offset = teal → 먼 offset = amber)
_OFFSET_CMAP = LinearSegmentedColormap.from_list(
    "offset_ramp", ["#4fd1c5", "#6aa3ff", "#b48cff", "#ffb74d"])
SIDE_COLORS = {"R": ACCENT, "S": ACCENT, "L": "#ffb74d"}
SOURCE_COLOR = "#ffffff"

FONT_FAMILY = "Segoe UI"


def offset_colors(n: int):
    """offset n 개에 대한 순차 색"""
    if n <= 1:
        return [_OFFSET_CMAP(0.0)]
    return [_OFFSET_CMAP(i / (n - 1)) for i in range(n)]


def layer_color(i: int) -> str:
    return LAYER_COLORS[i % len(LAYER_COLORS)]


# ---- matplotlib ----------------------------------------------------------------
_FONTS = ["Malgun Gothic", "DejaVu Sans"]

MPL_DARK = {
    "figure.facecolor": BG1, "savefig.facecolor": BG1,
    "axes.facecolor": BG1, "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": BORDER, "grid.linewidth": 0.8, "grid.alpha": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": TEXT, "ytick.labelcolor": TEXT,
    "text.color": TEXT, "font.family": _FONTS, "font.size": 10,
    "legend.facecolor": BG2, "legend.edgecolor": BORDER, "legend.labelcolor": TEXT,
    "legend.framealpha": 0.95, "legend.fontsize": 8.5,
    "lines.linewidth": 2.0, "axes.unicode_minus": False,
}

# 실측 비교 / 라만 패턴: 지정 색과 파이프라인 그림이 밝은 배경 기준이라 '종이' 스타일
MPL_PAPER = {
    "figure.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff", "axes.edgecolor": "#3a3f4a",
    "axes.labelcolor": "#1b1e24", "axes.titlecolor": "#1b1e24",
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#d9dce3", "grid.linewidth": 0.8, "grid.alpha": 1.0,
    "xtick.color": "#3a3f4a", "ytick.color": "#3a3f4a",
    "xtick.labelcolor": "#1b1e24", "ytick.labelcolor": "#1b1e24",
    "text.color": "#1b1e24", "font.family": _FONTS, "font.size": 10,
    "legend.facecolor": "#ffffff", "legend.edgecolor": "#d9dce3", "legend.labelcolor": "#1b1e24",
    "legend.framealpha": 0.95, "legend.fontsize": 8.5,
    "lines.linewidth": 2.0, "axes.unicode_minus": False,
}


# ---- Qt 스타일시트 ---------------------------------------------------------------
QSS = f"""
* {{
    font-family: "{FONT_FAMILY}", "Malgun Gothic";
    font-size: 10pt;
    color: {TEXT};
}}
QMainWindow, QWidget {{ background: {BG0}; }}
QLabel, QCheckBox {{ background: transparent; }}
QWidget#Transparent {{ background: transparent; }}
QToolTip {{ background: {BG2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px; }}

QToolBar {{
    background: {BG1}; border: none; border-bottom: 1px solid {BORDER};
    padding: 4px 8px; spacing: 6px;
}}
QToolBar QToolButton {{
    background: transparent; border: 1px solid transparent; border-radius: 6px;
    padding: 5px 10px; color: {TEXT};
}}
QToolBar QToolButton:hover {{ background: {BG3}; border-color: {BORDER}; }}
QToolBar QToolButton:pressed {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}
QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 4px 6px; }}
QToolButton::menu-indicator {{ image: none; width: 0; height: 0; }}

QStatusBar {{ background: {BG1}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QStatusBar QLabel {{ color: {MUTED}; padding: 0 8px; }}
QStatusBar::item {{ border: none; }}

QFrame#Card {{ background: {BG1}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#Paper {{ background: #ffffff; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#LayerRow {{ background: {BG2}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#Summary {{ background: {BG1}; border: 1px solid {BORDER}; border-radius: 10px; }}
QLabel#CardTitle {{ font-size: 10.5pt; font-weight: 600; }}
QLabel#CardSubtitle, QLabel#Hint {{ color: {MUTED}; font-size: 9pt; }}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#SectionLabel {{ color: {MUTED}; font-size: 9pt; font-weight: 600; }}
QLabel#StatLabel {{ color: {MUTED}; font-size: 8.5pt; }}
QLabel#StatValue {{ font-size: 13pt; font-weight: 600; }}
QLabel#StatValueAccent {{ font-size: 13pt; font-weight: 600; color: {ACCENT}; }}
QLabel#Warn {{ color: {WARN}; font-size: 9pt; }}
QToolButton#CardToggle {{
    background: transparent; border: none; color: {TEXT}; font-size: 10.5pt;
    font-weight: 600; text-align: left; padding: 4px 2px;
}}
QToolButton#CardToggle:hover {{ color: {ACCENT}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 4px 8px; selection-background-color: {ACCENT_DIM};
}}
QFrame#LayerRow QLineEdit, QFrame#LayerRow QDoubleSpinBox, QFrame#LayerRow QComboBox {{
    background: {BG1};
}}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {MUTED}; background: {BG1};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 14px; border: none; background: transparent;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {BG3}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {BG2}; border: 1px solid {BORDER}; selection-background-color: {ACCENT_DIM}; outline: none;
}}

QPushButton {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 12px;
}}
QPushButton:hover {{ background: {BG3}; border-color: #3f4654; }}
QPushButton:pressed {{ background: {BG1}; }}
QPushButton:disabled {{ color: {MUTED}; background: {BG1}; border-color: {BORDER}; }}
QPushButton#Primary {{
    background: {ACCENT}; color: #0b1f1d; border: none; font-weight: 600; padding: 9px 14px;
    font-size: 10.5pt;
}}
QPushButton#Primary:hover {{ background: #6ee0d5; }}
QPushButton#Primary:pressed {{ background: #3bb8ac; }}
QPushButton#Primary:disabled {{ background: {ACCENT_DIM}; color: #0b1f1d; }}
QPushButton#Danger {{ color: {DANGER}; }}
QPushButton#Danger:hover {{ border-color: {DANGER}; }}
QPushButton#Ghost {{ background: transparent; border: 1px solid transparent; color: {MUTED}; }}
QPushButton#Ghost:hover {{ background: {BG3}; border-color: {BORDER}; color: {TEXT}; }}
QPushButton::menu-indicator {{ width: 0; }}

QPushButton#Seg, QPushButton#SegL, QPushButton#SegR {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 0; padding: 5px 12px; color: {MUTED};
}}
QPushButton#Seg:hover, QPushButton#SegL:hover, QPushButton#SegR:hover {{ color: {TEXT}; }}
QPushButton#Seg:checked, QPushButton#SegL:checked, QPushButton#SegR:checked {{
    background: {ACCENT_DIM}; border-color: {ACCENT}; color: #ffffff;
}}
QPushButton#SegL {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
QPushButton#SegR {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px; }}

QCheckBox, QRadioButton {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border: 1px solid {BORDER}; border-radius: 4px; background: {BG2};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QTableView, QTableWidget {{
    background: {BG2}; alternate-background-color: {BG1}; border: 1px solid {BORDER};
    border-radius: 6px; gridline-color: {BORDER}; selection-background-color: {ACCENT_DIM};
    selection-color: #ffffff; outline: none;
}}
QHeaderView::section {{
    background: {BG1}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 4px 6px; font-weight: 600; font-size: 9pt;
}}
QTableCornerButton::section {{ background: {BG1}; border: none; }}

QSplitter::handle {{ background: {BG0}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QSplitter::handle:hover {{ background: {ACCENT_DIM}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BG3}; min-height: 24px; border-radius: 5px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BG3}; min-width: 24px; border-radius: 5px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

QProgressBar {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 4px; max-height: 8px;
    text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QMenu {{ background: {BG2}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_DIM}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}

QTabWidget::pane {{ border: none; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; padding: 6px 12px; border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QMessageBox {{ background: {BG1}; }}
"""


def apply_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, col in [
        (QtGui.QPalette.Window, BG0), (QtGui.QPalette.WindowText, TEXT),
        (QtGui.QPalette.Base, BG2), (QtGui.QPalette.AlternateBase, BG1),
        (QtGui.QPalette.Text, TEXT), (QtGui.QPalette.Button, BG2),
        (QtGui.QPalette.ButtonText, TEXT), (QtGui.QPalette.Highlight, ACCENT_DIM),
        (QtGui.QPalette.HighlightedText, "#ffffff"), (QtGui.QPalette.ToolTipBase, BG2),
        (QtGui.QPalette.ToolTipText, TEXT), (QtGui.QPalette.PlaceholderText, MUTED),
    ]:
        pal.setColor(role, QtGui.QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    app.setFont(QtGui.QFont(FONT_FAMILY, 10))
    matplotlib.rcParams.update(MPL_DARK)

"""Dark graphite theme: colour tokens, Qt stylesheet and pyqtgraph defaults."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

# ---- colour tokens ---------------------------------------------------------
BG0 = "#121418"      # window background
BG1 = "#1a1d23"      # panels / docks
BG2 = "#22262e"      # inputs, cards
BG3 = "#2b303a"      # hover
BORDER = "#323845"
TEXT = "#e8eaf0"
MUTED = "#8d93a1"
ACCENT = "#4fd1c5"   # teal
ACCENT_DIM = "#2f7f78"
DANGER = "#f26d6d"
WARN = "#f5b84f"

CURSOR_R = "#ff6161"
CURSOR_G = "#63f08b"
CURSOR_B = "#6aa3ff"

# bright, distinguishable on dark backgrounds (points / spectra)
POINT_COLORS = [
    "#ffffff", "#4fc3f7", "#ffb74d", "#81c784", "#ef5350", "#ba68c8",
    "#ffd54f", "#4dd0e1", "#f06292", "#aed581", "#ff8a65", "#9fa8da",
]

# cluster colours (qualitative, dark-friendly)
CLUSTER_COLORS = [
    "#4fd1c5", "#ff7a59", "#8e7dff", "#ffd166", "#06d6a0", "#ef476f",
    "#59c3ff", "#c3f73a", "#ff9ff3", "#f9a03f", "#80ffdb", "#d4a5ff",
]

FONT_FAMILY = "Segoe UI"
MONO_FAMILY = "Consolas"


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


QSS = f"""
* {{
    font-family: "{FONT_FAMILY}", "Malgun Gothic";
    font-size: 10pt;
    color: {TEXT};
}}
QMainWindow, QWidget {{ background: {BG0}; }}
QToolTip {{
    background: {BG2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px;
}}

/* ---------- toolbar ---------- */
QToolBar {{
    background: {BG1}; border: none; border-bottom: 1px solid {BORDER};
    padding: 4px 8px; spacing: 6px;
}}
QToolBar QToolButton {{
    background: transparent; border: 1px solid transparent; border-radius: 6px;
    padding: 5px 10px; color: {TEXT};
}}
QToolBar QToolButton:hover {{ background: {BG3}; border-color: {BORDER}; }}
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {{
    background: {ACCENT_DIM}; border-color: {ACCENT}; color: #ffffff;
}}
QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 4px 6px; }}
QToolButton::menu-indicator {{ image: none; width: 0; height: 0; }}

/* ---------- status bar ---------- */
QStatusBar {{ background: {BG1}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QStatusBar QLabel {{ color: {MUTED}; padding: 0 8px; }}
QStatusBar::item {{ border: none; }}

/* ---------- panels / cards ---------- */
QFrame#Card {{
    background: {BG1}; border: 1px solid {BORDER}; border-radius: 10px;
}}
QLabel#CardTitle {{
    color: {TEXT}; font-size: 10.5pt; font-weight: 600; letter-spacing: 0.3px;
}}
QLabel#CardSubtitle {{ color: {MUTED}; font-size: 9pt; }}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#Mono {{ font-family: "{MONO_FAMILY}"; }}
QLabel#Hint {{ color: {MUTED}; font-size: 9pt; }}
QLabel#SectionLabel {{ color: {MUTED}; font-size: 8.5pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#Accent {{ color: {ACCENT}; }}
QToolButton#CardToggle {{
    background: transparent; border: none; color: {TEXT}; font-size: 10.5pt;
    font-weight: 600; text-align: left; padding: 6px 4px;
}}
QToolButton#CardToggle:hover {{ color: {ACCENT}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* ---------- inputs ---------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 4px 8px; selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {MUTED}; background: {BG1};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 16px; border: none; background: transparent; subcontrol-origin: border;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {BG3};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {BG2}; border: 1px solid {BORDER}; selection-background-color: {ACCENT_DIM};
    outline: none;
}}

/* ---------- buttons ---------- */
QPushButton {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 6px 12px; color: {TEXT};
}}
QPushButton:hover {{ background: {BG3}; border-color: #3f4654; }}
QPushButton:pressed {{ background: {BG1}; }}
QPushButton:disabled {{ color: {MUTED}; background: {BG1}; border-color: {BORDER}; }}
QPushButton#Primary {{
    background: {ACCENT}; color: #0b1f1d; border: none; font-weight: 600;
}}
QPushButton#Primary:hover {{ background: #6ee0d5; }}
QPushButton#Primary:pressed {{ background: #3bb8ac; }}
QPushButton#Primary:disabled {{ background: {ACCENT_DIM}; color: #0b1f1d; }}
QPushButton#Danger {{ color: {DANGER}; }}
QPushButton#Danger:hover {{ border-color: {DANGER}; }}
QPushButton#Ghost {{ background: transparent; border: 1px solid transparent; }}
QPushButton#Ghost:hover {{ background: {BG3}; border-color: {BORDER}; }}
QPushButton::menu-indicator {{ width: 0; }}

/* ---------- segmented buttons ---------- */
QPushButton#Seg, QPushButton#SegL, QPushButton#SegR {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 0; padding: 5px 12px;
}}
QPushButton#Seg:checked, QPushButton#SegL:checked, QPushButton#SegR:checked {{
    background: {ACCENT_DIM}; border-color: {ACCENT}; color: #ffffff;
}}
QPushButton#SegL {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
QPushButton#SegR {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px; }}

/* ---------- check / radio ---------- */
QCheckBox, QRadioButton {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {BORDER}; border-radius: 4px; background: {BG2};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
}}
QCheckBox::indicator:disabled {{ background: {BG1}; }}

/* ---------- sliders ---------- */
QSlider {{ background: transparent; }}
QSlider::groove:horizontal {{
    height: 4px; background: {BG3}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
    background: {TEXT}; border: 2px solid {BG1};
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}
QSlider:disabled::sub-page:horizontal {{ background: {BORDER}; }}

/* ---------- tables ---------- */
QTableView, QTableWidget {{
    background: {BG2}; alternate-background-color: {BG1}; border: 1px solid {BORDER};
    border-radius: 6px; gridline-color: {BORDER}; selection-background-color: {ACCENT_DIM};
    selection-color: #ffffff; outline: none;
}}
QHeaderView::section {{
    background: {BG1}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 4px 6px; font-weight: 600; font-size: 9pt;
}}
QTableView::item {{ padding: 2px 6px; }}
QTableCornerButton::section {{ background: {BG1}; border: none; }}

/* ---------- splitter / scrollbar ---------- */
QSplitter::handle {{ background: {BG0}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QSplitter::handle:hover {{ background: {ACCENT_DIM}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BG3}; min-height: 24px; border-radius: 5px; }}
QScrollBar::handle:vertical:hover {{ background: #3b4250; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BG3}; min-width: 24px; border-radius: 5px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---------- progress ---------- */
QProgressBar {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 5px; max-height: 8px; text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}

/* ---------- menus ---------- */
QMenu {{ background: {BG2}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_DIM}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}

/* ---------- tabs ---------- */
QTabWidget::pane {{ border: none; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; padding: 6px 12px; border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QGroupBox {{ border: 1px solid {BORDER}; border-radius: 8px; margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}

QMessageBox {{ background: {BG1}; }}
"""


def apply_theme(app: QtWidgets.QApplication) -> None:
    """Apply the dark palette, stylesheet and pyqtgraph defaults to the app."""
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(BG0))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(BG2))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(BG1))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(BG2))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(ACCENT_DIM))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(BG2))
    pal.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(MUTED))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    app.setFont(QtGui.QFont(FONT_FAMILY, 10))

    pg.setConfigOptions(
        imageAxisOrder="row-major",
        background=BG1,
        foreground=TEXT,
        antialias=True,
        useOpenGL=False,
    )

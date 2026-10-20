# -*- coding: utf-8 -*-
"""앱 전체 테마. 색상 토큰 하나로 통일하고 Qt 스타일시트로 적용."""

ACCENT = "#2563EB"        # 주 강조색 (버튼, 포커스)
ACCENT_HOVER = "#1D4ED8"
ACCENT_PRESSED = "#1E40AF"
BG = "#FFFFFF"            # 창 배경
PANEL = "#FFFFFF"         # 패널/그룹 배경
BORDER = "#D1D5DB"
TEXT = "#1F2937"
TEXT_MUTED = "#6B7280"
PLOT_RAW = "#1F2937"      # 원시 스펙트럼
PLOT_BASE = "#DC2626"     # 베이스라인
PLOT_OUT = ACCENT         # 처리 결과
REGION = (37, 99, 235, 18)  # crop 영역 (RGBA)

FONT_FAMILY = "Segoe UI"
FONT_PT = 10

STYLESHEET = f"""
QMainWindow, QDialog {{ background: {BG}; }}
QWidget {{ color: {TEXT}; font-family: "{FONT_FAMILY}"; font-size: {FONT_PT}pt; }}

QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding: 6px 8px 4px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_MUTED};
    font-weight: 600;
}}

QLabel#muted {{ color: {TEXT_MUTED}; }}
QLabel#fileName {{ font-weight: 600; }}

QPushButton {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 12px;
    min-height: 22px;
}}
QPushButton:hover {{ background: #F9FAFB; border-color: #9CA3AF; }}
QPushButton:pressed {{ background: #E5E7EB; }}
QPushButton:disabled {{ color: #9CA3AF; background: #F3F4F6; }}
QPushButton#primary {{
    background: {ACCENT}; color: white; border: 1px solid {ACCENT}; font-weight: 600;
}}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton#primary:disabled {{ background: #93C5FD; border-color: #93C5FD; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 22px;
    selection-background-color: {ACCENT};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus,
QPlainTextEdit:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {ACCENT};
    selection-color: white;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; }}
QPlainTextEdit {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px;
    font-family: Consolas, "Courier New", monospace;
}}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}

QTabWidget::pane {{ border: none; background: {BG}; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_MUTED};
    padding: 8px 18px; border-bottom: 2px solid transparent; font-weight: 600;
}}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {TEXT_MUTED}; }}
QSplitter::handle {{ background: {BG}; width: 6px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {ACCENT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
"""

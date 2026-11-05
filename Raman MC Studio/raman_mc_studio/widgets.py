"""공용 위젯: 카드, 세그먼트 버튼, 매질 단면 미리보기, 그래프 틀, CSV 내보내기."""
from __future__ import annotations

import csv

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme


# ------------------------------------------------------------------ 기본 조각
class Card(QtWidgets.QFrame):
    """제목 + 접을 수 있는 본문 패널"""

    def __init__(self, title: str, subtitle: str = "", collapsible: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 14)
        outer.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setObjectName("CardToggle")
        self.toggle.setText(title)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        if collapsible:
            self.toggle.setCheckable(True)
            self.toggle.setChecked(True)
            self.toggle.setArrowType(QtCore.Qt.DownArrow)
            self.toggle.setCursor(QtCore.Qt.PointingHandCursor)
            self.toggle.toggled.connect(self._on_toggle)
        header.addWidget(self.toggle, 1)
        self.header_extra = QtWidgets.QHBoxLayout()
        header.addLayout(self.header_extra)
        outer.addLayout(header)

        self.subtitle = QtWidgets.QLabel(subtitle)
        self.subtitle.setObjectName("CardSubtitle")
        self.subtitle.setWordWrap(True)
        self.subtitle.setVisible(bool(subtitle))
        outer.addWidget(self.subtitle)

        self.body = QtWidgets.QWidget()
        self.body.setObjectName("Transparent")
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 2, 0, 0)
        self.body_layout.setSpacing(8)
        outer.addWidget(self.body)

    def _on_toggle(self, checked: bool):
        self.body.setVisible(checked)
        self.subtitle.setVisible(checked and bool(self.subtitle.text()))
        self.toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)

    def add(self, item):
        if isinstance(item, QtWidgets.QLayout):
            self.body_layout.addLayout(item)
        else:
            self.body_layout.addWidget(item)
        return item


class SegmentedControl(QtWidgets.QWidget):
    """서로 배타적인 버튼 묶음"""

    changed = QtCore.Signal(int)

    def __init__(self, labels: list[str], current: int = 0, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = []
        for i, text in enumerate(labels):
            b = QtWidgets.QPushButton(text)
            b.setCheckable(True)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setObjectName("SegL" if i == 0 else "SegR" if i == len(labels) - 1 else "Seg")
            self.group.addButton(b, i)
            lay.addWidget(b)
            self.buttons.append(b)
        self.buttons[current].setChecked(True)
        self.group.idClicked.connect(self.changed)

    def current(self) -> int:
        return self.group.checkedId()

    def setCurrent(self, i: int):
        self.buttons[i].setChecked(True)


def hint(text: str, name: str = "Hint") -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setObjectName(name)
    lab.setWordWrap(True)
    return lab


def muted(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("Muted")
    return lab


def form_row(label: str, widget, label_width: int = 104) -> QtWidgets.QHBoxLayout:
    row = QtWidgets.QHBoxLayout()
    row.setSpacing(8)
    lab = muted(label)
    lab.setFixedWidth(label_width)
    row.addWidget(lab)
    if isinstance(widget, QtWidgets.QLayout):
        row.addLayout(widget, 1)
    else:
        row.addWidget(widget, 1)
    return row


def dspin(lo, hi, value, decimals=2, step=None, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(value)
    s.setSingleStep(step if step is not None else 10 ** -decimals * 10)
    if suffix:
        s.setSuffix(suffix)
    s.setAlignment(QtCore.Qt.AlignRight)
    return s


def button(text, kind="", tooltip=""):
    b = QtWidgets.QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if tooltip:
        b.setToolTip(tooltip)
    b.setCursor(QtCore.Qt.PointingHandCursor)
    return b


class _NoWheel(QtCore.QObject):
    """휠 이벤트를 숫자칸·선택칸 대신 감싸는 스크롤 영역으로 넘긴다"""

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Wheel:
            p = obj.parentWidget()
            while p is not None and not isinstance(p, QtWidgets.QAbstractScrollArea):
                p = p.parentWidget()
            if p is not None:
                QtCore.QCoreApplication.sendEvent(p.verticalScrollBar(), ev)   # 스크롤만 함
            return True          # 이 칸 자신은 이벤트를 받지 않음 (값 안 바뀜)
        return False


def block_wheel(root: QtWidgets.QWidget):
    """root 안의 모든 숫자칸(SpinBox)·선택칸(ComboBox)이 휠로 값이 바뀌지 않게 한다"""
    f = _NoWheel(root)           # root 의 자식으로 두어 root 와 같이 살아 있게
    for w in root.findChildren(QtWidgets.QAbstractSpinBox) + root.findChildren(QtWidgets.QComboBox):
        w.installEventFilter(f)


class RangeListEditor(QtWidgets.QWidget):
    """구간 목록 [[a, b], ...] 편집: 한 줄에 시작 ~ 끝 숫자칸 두 개, ✕ 로 삭제, '+ 구간 추가' 로 추가"""

    changed = QtCore.Signal()                  # 값이 바뀌거나 줄이 추가/삭제되면 알림

    def __init__(self, suffix=" mm", parent=None):
        super().__init__(parent)
        self.suffix = suffix
        self.rows = []                          # [(줄 위젯, 시작칸, 끝칸), ...]
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.row_box = QtWidgets.QVBoxLayout()
        self.row_box.setSpacing(4)
        lay.addLayout(self.row_box)
        add = button("+ 구간 추가", "Ghost")
        add.clicked.connect(self._add_next)
        lay.addWidget(add, 0, QtCore.Qt.AlignLeft)

    def _add_row(self, a, b):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        sa = dspin(0, 100, a, 2, 0.5, self.suffix)
        sb = dspin(0, 100, b, 2, 0.5, self.suffix)
        for s in (sa, sb):
            s.setMinimumWidth(60)               # 왼쪽 패널 폭 안에 두 칸이 들어가게
        x = button("✕", "Ghost", "이 구간 삭제")
        x.setFixedWidth(26)
        x.setStyleSheet("padding: 0;")
        h.addWidget(sa, 1); h.addWidget(QtWidgets.QLabel("~")); h.addWidget(sb, 1); h.addWidget(x)
        block_wheel(w)                          # 새로 생긴 칸도 휠로 값이 바뀌지 않게
        sa.valueChanged.connect(self.changed)
        sb.valueChanged.connect(self.changed)
        x.clicked.connect(lambda: self._remove(w))   # lambda: 클릭 때 이 줄(w)을 넘겨 지움
        self.row_box.addWidget(w)
        self.rows.append((w, sa, sb))

    def _add_next(self):
        last = self.rows[-1][2].value() if self.rows else 0.0   # 새 구간은 마지막 끝에서 2 mm
        self._add_row(last, last + 2)
        self.changed.emit()

    def _remove(self, w):
        self.rows = [r for r in self.rows if r[0] is not w]
        self.row_box.removeWidget(w)
        w.hide()                                # deleteLater 는 나중에 지우므로 화면에서는 바로 숨김
        w.deleteLater()
        self.changed.emit()

    def set_ranges(self, ranges):
        for w, _, _ in list(self.rows):         # list(): 지우는 동안 목록이 바뀌어도 안전하게 복사본으로 돎
            self._remove(w)
        for a, b in ranges:
            self._add_row(a, b)

    def ranges(self):
        """끝이 시작보다 큰 구간만 [[a, b], ...] 로 돌려줌"""
        return [[sa.value(), sb.value()] for _, sa, sb in self.rows if sb.value() > sa.value()]


# ------------------------------------------------------------------ 매질 단면 미리보기
class StackPreview(QtWidgets.QWidget):
    """
    입력한 층을 두께 비율대로 그린 단면. 광원 위치와 검출 띠(0 ~ max offset)도 표시.
    지금 몇 mm 매질을 계산하는지 한눈에 보이게 하는 것이 목적.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers = []           # (name, thickness, mus_prime_per_mm, color)
        self.max_offset = 8.0
        self.source_label = "pencil"
        self.setMinimumHeight(190)
        self.setObjectName("Transparent")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_stack(self, layers, max_offset, source_label):
        self.layers = layers
        self.max_offset = max_offset
        self.source_label = source_label
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        top, bottom, left, right = 34, 26, 4, 4
        area = QtCore.QRectF(left, top, W - left - right, H - top - bottom)
        total = sum(t for _, t, _, _ in self.layers) or 1.0

        # x 축: -1 mm ~ max_offset + 1 mm
        x0, x1 = -1.0, self.max_offset + 1.0

        def px(xmm):
            return area.left() + (xmm - x0) / (x1 - x0) * area.width()

        # 층 띠
        y = area.top()
        f_name = QtGui.QFont(theme.FONT_FAMILY, 9)
        f_name.setWeight(QtGui.QFont.DemiBold)
        f_small = QtGui.QFont(theme.FONT_FAMILY, 9)
        for i, (name, t, musp, col) in enumerate(self.layers):
            h = area.height() * t / total
            rect = QtCore.QRectF(area.left(), y, area.width(), h)
            c = QtGui.QColor(col)
            c.setAlpha(70)
            p.fillRect(rect, c)
            edge = QtGui.QColor(col)
            edge.setAlpha(160)
            p.setPen(QtGui.QPen(edge, 1))
            p.drawLine(rect.bottomLeft(), rect.bottomRight())
            if h >= 16:
                p.setPen(QtGui.QColor(theme.TEXT))
                p.setFont(f_name)
                txt_rect = rect.adjusted(8, 0, -8, 0)
                p.drawText(txt_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, name)
                p.setFont(f_small)
                p.setPen(QtGui.QColor(theme.TEXT))
                p.drawText(txt_rect, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                           f"{t:g} mm    μs′ {musp:.2f}/mm")
            y += h

        # 표면선
        p.setPen(QtGui.QPen(QtGui.QColor(theme.MUTED), 1))
        p.drawLine(QtCore.QPointF(area.left(), area.top()), QtCore.QPointF(area.right(), area.top()))

        # 검출 띠 (0 ~ max offset)
        acc = QtGui.QColor(theme.ACCENT)
        p.setPen(QtGui.QPen(acc, 3, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
        yd = area.top() - 6
        p.drawLine(QtCore.QPointF(px(0), yd), QtCore.QPointF(px(self.max_offset), yd))
        p.setPen(QtGui.QPen(acc, 1))
        for k in range(0, int(self.max_offset) + 1):
            p.drawLine(QtCore.QPointF(px(k), yd - 3), QtCore.QPointF(px(k), yd + 3))
        p.setFont(f_small)
        p.drawText(QtCore.QRectF(px(0.6), 2, area.width(), 16),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   f"검출 띠 0–{self.max_offset:g} mm")

        # 광원 (아래 방향 삼각형)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(theme.SOURCE_COLOR))
        sx = px(0)
        tri = QtGui.QPolygonF([QtCore.QPointF(sx - 6, area.top() - 18),
                               QtCore.QPointF(sx + 6, area.top() - 18),
                               QtCore.QPointF(sx, area.top() - 8)])
        p.drawPolygon(tri)

        # 총 두께
        p.setPen(QtGui.QColor(theme.MUTED))
        p.setFont(f_small)
        p.drawText(QtCore.QRectF(area.left(), area.bottom() + 4, area.width(), 18),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, f"광원: {self.source_label}")
        p.setPen(QtGui.QColor(theme.TEXT))
        p.drawText(QtCore.QRectF(area.left(), area.bottom() + 4, area.width(), 18),
                   QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"총 두께 {sum(t for _, t, _, _ in self.layers):g} mm")
        p.end()


# ------------------------------------------------------------------ 그래프
def outside_legend(ax, show=True):
    """범례를 축 오른쪽 바깥에 배치. show=False 면 숨김"""
    if not ax.get_legend_handles_labels()[0]:
        return
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    leg.set_visible(show)


def export_tables(parent, tables, default_name):
    """tables = {이름: (header, rows)}. 표가 1개면 지정 파일, 여러 개면 <파일명>_<이름>.csv"""
    if not tables:
        QtWidgets.QMessageBox.information(parent, "저장할 데이터 없음", "먼저 시뮬레이션을 실행하세요.")
        return
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, "그래프 데이터 저장", default_name, "CSV (*.csv)")
    if not path:
        return
    prefix = path[:-4] if path.lower().endswith(".csv") else path
    files = []
    for name, (header, rows) in tables.items():
        p = f"{prefix}.csv" if len(tables) == 1 else f"{prefix}_{name}.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow([v if isinstance(v, str) else f"{v:.6g}" for v in r])
        files.append(p)
    QtWidgets.QMessageBox.information(parent, "저장됨", "\n".join(files))


class MplCanvas(FigureCanvasQTAgg):
    """matplotlib 캔버스. paper=True 면 흰 바탕 스타일로 그림"""

    def __init__(self, paper: bool = False, nrows: int = 1, ncols: int = 1):
        self.paper = paper
        with matplotlib.rc_context(theme.MPL_PAPER if paper else theme.MPL_DARK):
            self.fig = Figure(figsize=(7, 5), layout="constrained")
            super().__init__(self.fig)
            self.axes = [self.fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]
        self.ax = self.axes[0]
        self.export_tables = {}
        self.show_legend = True
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def style(self):
        return matplotlib.rc_context(theme.MPL_PAPER if self.paper else theme.MPL_DARK)

    def reset_axes(self):
        """축을 현재 스타일로 다시 만든다 (ax.clear 는 rc 색을 다시 읽지 않으므로)"""
        n = len(self.axes)
        geom = [ax.get_subplotspec().get_geometry()[:2] for ax in self.axes][0]
        self.fig.clear()
        self.axes = [self.fig.add_subplot(geom[0], geom[1], i + 1) for i in range(n)]
        self.ax = self.axes[0]

    def message(self, text):
        with self.style():
            self.reset_axes()
            for ax in self.axes:
                ax.set_axis_off()
            self.axes[0].text(0.5, 0.5, text, ha="center", va="center",
                              transform=self.axes[0].transAxes, color=theme.MUTED if not self.paper else "#5b6070",
                              fontsize=11)
        self.draw_idle()

    def set_legend_visible(self, show):
        self.show_legend = show
        for ax in self.fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.set_visible(show)
        self.draw_idle()


class _PaperNav(NavigationToolbar2QT):
    """흰 바탕용 툴바: matplotlib 은 palette 밝기로 아이콘 색을 정하므로 밝은 palette 를 돌려준다"""

    def palette(self):
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#ffffff"))
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#1b1e24"))
        return pal


class PlotFrame(QtWidgets.QFrame):
    """캔버스 + 확대/이동 툴바 + 'CSV 저장' 버튼을 묶은 카드"""

    def __init__(self, canvas: MplCanvas, csv_name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Paper" if canvas.paper else "Card")
        self.canvas = canvas
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(2)
        bar = QtWidgets.QHBoxLayout()
        self.nav = (_PaperNav if canvas.paper else NavigationToolbar2QT)(canvas, self)
        self.nav.setIconSize(QtCore.QSize(18, 18))
        if canvas.paper:
            self.nav.setStyleSheet("QToolBar { background: #ffffff; border: none; }"
                                   "QToolButton { background: transparent; color: #1b1e24; }"
                                   "QLabel { color: #5b6070; background: transparent; }")
        else:
            self.nav.setStyleSheet(f"QToolBar {{ background: {theme.BG1}; border: none; }}")
        bar.addWidget(self.nav, 1)
        self.csv_btn = button("그래프 데이터 CSV", "Ghost", "이 그래프에 그린 값을 CSV 로 저장")
        if canvas.paper:
            self.csv_btn.setStyleSheet("QPushButton { color: #3a3f4a; } QPushButton:hover { background: #eef0f4; }")
        self.csv_btn.clicked.connect(lambda: export_tables(self, canvas.export_tables, csv_name))
        bar.addWidget(self.csv_btn)
        lay.addLayout(bar)
        lay.addWidget(canvas, 1)

"""
Tab 1: Input — 매질 layer + 광원 설정
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QScrollArea, QFrame, QGridLayout
)

from core.presets import get_preset_names, get_preset


class LayerWidget(QGroupBox):
    """단일 layer 입력 위젯"""

    removed = pyqtSignal(object)  # self를 emit

    def __init__(self, idx: int, parent=None):
        super().__init__(f"Layer {idx + 1}", parent)
        self.idx = idx
        self._build()

    def _build(self):
        layout = QGridLayout(self)

        # Preset 선택
        layout.addWidget(QLabel("Preset:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(get_preset_names())
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        layout.addWidget(self.preset_combo, 0, 1, 1, 3)

        # Name
        layout.addWidget(QLabel("Name:"), 1, 0)
        self.name_edit = QLineEdit("Tissue")
        layout.addWidget(self.name_edit, 1, 1, 1, 3)

        # Thickness
        layout.addWidget(QLabel("Thickness (mm):"), 2, 0)
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.1, 50.0)
        self.thickness_spin.setValue(5.0)
        self.thickness_spin.setDecimals(2)
        self.thickness_spin.setSingleStep(0.5)
        layout.addWidget(self.thickness_spin, 2, 1)

        # μa
        layout.addWidget(QLabel("μa (mm⁻¹):"), 3, 0)
        self.mua_spin = QDoubleSpinBox()
        self.mua_spin.setRange(0.0, 10.0)
        self.mua_spin.setValue(0.03)
        self.mua_spin.setDecimals(4)
        self.mua_spin.setSingleStep(0.001)
        layout.addWidget(self.mua_spin, 3, 1)

        # μs
        layout.addWidget(QLabel("μs (mm⁻¹):"), 3, 2)
        self.mus_spin = QDoubleSpinBox()
        self.mus_spin.setRange(0.0, 100.0)
        self.mus_spin.setValue(10.0)
        self.mus_spin.setDecimals(3)
        self.mus_spin.setSingleStep(0.1)
        layout.addWidget(self.mus_spin, 3, 3)

        # g
        layout.addWidget(QLabel("g:"), 4, 0)
        self.g_spin = QDoubleSpinBox()
        self.g_spin.setRange(-1.0, 1.0)
        self.g_spin.setValue(0.9)
        self.g_spin.setDecimals(3)
        self.g_spin.setSingleStep(0.01)
        layout.addWidget(self.g_spin, 4, 1)

        # n
        layout.addWidget(QLabel("n:"), 4, 2)
        self.n_spin = QDoubleSpinBox()
        self.n_spin.setRange(1.0, 2.0)
        self.n_spin.setValue(1.40)
        self.n_spin.setDecimals(3)
        self.n_spin.setSingleStep(0.01)
        layout.addWidget(self.n_spin, 4, 3)

        # μs' (계산값 표시)
        self.musp_label = QLabel("μs' = ? cm⁻¹")
        self.musp_label.setObjectName("hint")
        layout.addWidget(self.musp_label, 5, 0, 1, 3)
        self._update_musp()
        self.mus_spin.valueChanged.connect(self._update_musp)
        self.g_spin.valueChanged.connect(self._update_musp)

        # Remove button
        self.remove_btn = QPushButton("Remove Layer")
        self.remove_btn.setObjectName("danger")
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(self.remove_btn, 5, 3)

    def _update_musp(self):
        musp = self.mus_spin.value() * (1 - self.g_spin.value()) * 10  # mm⁻¹ → cm⁻¹
        self.musp_label.setText(f"μs' (reduced) = {musp:.2f} cm⁻¹")

    def _on_preset_changed(self, name: str):
        preset = get_preset(name)
        self.mua_spin.setValue(preset['mua'])
        self.mus_spin.setValue(preset['mus'])
        self.g_spin.setValue(preset['g'])
        self.n_spin.setValue(preset['n'])
        if name != "Custom":
            self.name_edit.setText(name.split(' (')[0])
        self._update_musp()

    def get_layer_data(self):
        return {
            'name': self.name_edit.text() or f"Layer{self.idx + 1}",
            'thickness': self.thickness_spin.value(),
            'mua': self.mua_spin.value(),
            'mus': self.mus_spin.value(),
            'g': self.g_spin.value(),
            'n': self.n_spin.value(),
        }


class InputTab(QWidget):
    """매질 + 광원 설정 탭"""

    config_changed = pyqtSignal()

    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        self.state = state
        self.layer_widgets = []
        self._build()
        self.add_layer()  # 기본 layer 1개

    def _build(self):
        main_layout = QVBoxLayout(self)

        # 안내 라벨
        info = QLabel(
            "<b>매질 설정</b>: 표면(z=0)에서 깊이 방향(+z)으로 layer가 쌓입니다. "
            "광원은 표면 (0, 0, 0)에서 +z 방향으로 입사됩니다."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 8px; background: #fef3c7; border-radius: 4px;")
        main_layout.addWidget(info)

        # ─── 매질 layers ───
        layers_group = QGroupBox("Medium Layers")
        layers_vbox = QVBoxLayout(layers_group)

        # Add layer button
        add_btn = QPushButton("+ Add Layer")
        add_btn.clicked.connect(self.add_layer)
        layers_vbox.addWidget(add_btn)

        # 스크롤 영역
        self.layers_scroll = QScrollArea()
        self.layers_scroll.setWidgetResizable(True)
        self.layers_container = QWidget()
        self.layers_layout = QVBoxLayout(self.layers_container)
        self.layers_layout.setAlignment(Qt.AlignTop)
        self.layers_scroll.setWidget(self.layers_container)
        layers_vbox.addWidget(self.layers_scroll)

        main_layout.addWidget(layers_group, stretch=3)

        # ─── 광원 + 검출 설정 ───
        source_group = QGroupBox("Source & Simulation Settings")
        source_layout = QGridLayout(source_group)

        source_layout.addWidget(QLabel("Number of photons:"), 0, 0)
        self.nphotons_spin = QSpinBox()
        self.nphotons_spin.setRange(100, 10_000_000)
        self.nphotons_spin.setValue(50_000)
        self.nphotons_spin.setSingleStep(10_000)
        self.nphotons_spin.setSuffix("  (more = slower but smoother)")
        source_layout.addWidget(self.nphotons_spin, 0, 1)

        source_layout.addWidget(QLabel("Max detection offset (mm):"), 1, 0)
        self.max_offset_spin = QDoubleSpinBox()
        self.max_offset_spin.setRange(1.0, 30.0)
        self.max_offset_spin.setValue(8.0)
        self.max_offset_spin.setDecimals(1)
        source_layout.addWidget(self.max_offset_spin, 1, 1)

        source_layout.addWidget(QLabel("Detector pixel size (mm):"), 2, 0)
        self.pixel_spin = QDoubleSpinBox()
        self.pixel_spin.setRange(0.05, 5.0)
        self.pixel_spin.setValue(0.3)
        self.pixel_spin.setDecimals(2)
        source_layout.addWidget(self.pixel_spin, 2, 1)

        source_layout.addWidget(QLabel("# paths to visualize:"), 3, 0)
        self.paths_spin = QSpinBox()
        self.paths_spin.setRange(0, 100000)
        self.paths_spin.setValue(100)
        self.paths_spin.setSuffix("  (0 = no path visualization)")
        source_layout.addWidget(self.paths_spin, 3, 1)

        main_layout.addWidget(source_group, stretch=1)

        # ─── 요약 ───
        self.summary_label = QLabel("총 두께: 0.0 mm | Layer 수: 0")
        self.summary_label.setStyleSheet(
            "padding: 6px 12px; background: #ecfdf5; border-radius: 4px; font-weight: 600;"
        )
        main_layout.addWidget(self.summary_label)

    def add_layer(self):
        idx = len(self.layer_widgets)
        layer = LayerWidget(idx)
        layer.removed.connect(self.remove_layer)
        layer.thickness_spin.valueChanged.connect(self._update_summary)   # ← 이 줄 추가
        self.layer_widgets.append(layer)
        self.layers_layout.addWidget(layer)
        self._update_summary()

    def remove_layer(self, layer):
        if len(self.layer_widgets) <= 1:
            QMessageBox.warning(self, "Cannot remove",
                                "최소 1개 layer는 필요합니다.")
            return
        self.layer_widgets.remove(layer)
        self.layers_layout.removeWidget(layer)
        layer.deleteLater()
        # 인덱스 재정렬
        for i, w in enumerate(self.layer_widgets):
            w.idx = i
            w.setTitle(f"Layer {i + 1}")
        self._update_summary()

    def _update_summary(self):
        total = sum(w.get_layer_data()['thickness'] for w in self.layer_widgets)
        n = len(self.layer_widgets)
        self.summary_label.setText(f"총 두께: {total:.2f} mm | Layer 수: {n}")

    def get_simulation_config(self):
        """현재 설정을 SimulationConfig로 변환"""
        from core.monte_carlo import Layer, SimulationConfig

        layers = []
        for w in self.layer_widgets:
            d = w.get_layer_data()
            layers.append(Layer(
                name=d['name'], thickness=d['thickness'],
                mua=d['mua'], mus=d['mus'], g=d['g'], n=d['n']
            ))

        n_paths = self.paths_spin.value()
        return SimulationConfig(
            layers=layers,
            n_photons=self.nphotons_spin.value(),
            detector_radius=self.pixel_spin.value(),
            max_offset=self.max_offset_spin.value(),
            save_paths=(n_paths > 0),
            n_paths_to_save=n_paths,
        )

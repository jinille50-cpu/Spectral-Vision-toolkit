import os
import importlib
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QSizePolicy
)
from PyQt5.QtCore import Qt


# 전이학습 가중치 매핑 (모델별)
PRETRAINED_WEIGHTS = {
    # ── SMP CNN 계열 ──────────────────────────────────────────
    "UNet":       ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet", "efficientnet-b0"],
    "UNet++":     ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet"],
    "MAnet":      ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet", "resnet101-ImageNet"],
    "FPN":        ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet"],
    "PSPNet":     ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet"],
    "PAN":        ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet"],
    "Linknet":    ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet"],
    "DeepLabV3":  ["없음 (scratch)", "resnet50-ImageNet", "resnet101-ImageNet"],
    "DeepLabV3+": ["없음 (scratch)", "resnet50-ImageNet", "resnet101-ImageNet", "xception-ImageNet"],
    # ── Transformer 계열 ──────────────────────────────────────
    "SegFormer-B0": ["없음 (scratch)", "nvidia/mit-b0 (ImageNet)"],
    "SegFormer-B2": ["없음 (scratch)", "nvidia/mit-b2 (ImageNet)", "nvidia/mit-b2 (ImageNet-22K)"],
    "SegFormer-B4": ["없음 (scratch)", "nvidia/mit-b4 (ImageNet-22K)"],
    "MobileViT":    ["없음 (scratch)", "apple/deeplabv3-mobilevit-small"],
    "Mask2Former":  ["없음 (scratch)", "swin-T-ImageNet", "swin-B-ImageNet-22K"],
}

# 표시명 → smp 실제 encoder_name 매핑
ENCODER_MAP = {
    "기본값 (resnet34)":    "resnet34",
    "resnet18":             "resnet18",
    "resnet34":             "resnet34",
    "resnet50":             "resnet50",
    "resnet101":            "resnet101",
    "resnet152":            "resnet152",
    "resnext50_32x4d":      "resnext50_32x4d",
    "resnext101_32x4d":     "resnext101_32x4d",
    "efficientnet-b0":      "efficientnet-b0",
    "efficientnet-b1":      "efficientnet-b1",
    "efficientnet-b2":      "efficientnet-b2",
    "efficientnet-b3":      "efficientnet-b3",
    "efficientnet-b4":      "efficientnet-b4",
    "efficientnet-b5":      "efficientnet-b5",
    "densenet121":          "densenet121",
    "densenet169":          "densenet169",
    "densenet201":          "densenet201",
    "mobilenet_v2":         "mobilenet_v2",
    "xception":             "xception",
    "mit_b0":               "mit_b0",
    "mit_b1":               "mit_b1",
    "mit_b2":               "mit_b2",
    "mit_b3":               "mit_b3",
    "mit_b4":               "mit_b4",
    "mit_b5":               "mit_b5",
    "timm-efficientnet-b0": "timm-efficientnet-b0",
    "timm-efficientnet-b4": "timm-efficientnet-b4",
    "vgg16":                "vgg16",
    "vgg19":                "vgg19",
}


class ModelTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state     = shared_state
        self._data_tab = None
        self._build()

    def set_data_tab(self, data_tab):
        self._data_tab = data_tab
        # 데이터탭에서 이미지 경로 변경 시 자동으로 채널/크기 감지
        if hasattr(data_tab, 'train_img_edit'):
            data_tab.train_img_edit.textChanged.connect(self._on_data_path_changed)
        if hasattr(data_tab, 'img_ext'):
            data_tab.img_ext.currentTextChanged.connect(self._on_data_path_changed)

    def _on_data_path_changed(self, *args):
        """데이터탭 경로/확장자 변경 시 자동 감지 (디바운스 처리)"""
        # 경로가 유효한 폴더일 때만 시도
        img_dir, ext = self._get_img_dir_and_ext()
        if img_dir and os.path.isdir(img_dir):
            # 짧은 딜레이로 연속 입력 흡수 (선택사항)
            self._auto_detect_all()

    def _get_img_dir_and_ext(self):
        """shared_state 우선, 없으면 data_tab UI에서 직접 읽기"""
        img_dir = self.state.get("train_img_dir", "").strip()
        ext     = self.state.get("img_ext", "png")
        if not img_dir and self._data_tab is not None:
            img_dir = self._data_tab.train_img_edit.text().strip()
            ext     = self._data_tab.img_ext.currentText()
        return img_dir, ext

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)

        main = QVBoxLayout(inner)
        main.setSpacing(14)
        main.setContentsMargins(20, 20, 20, 20)

        # ── 모델 선택 ─────────────────────────────────────────────
        model_group = QGroupBox("모델 선택")
        mg = QGridLayout(model_group)

        mg.addWidget(QLabel("Segmentation 모델:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(PRETRAINED_WEIGHTS.keys()))
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        mg.addWidget(self.model_combo, 0, 1)

        # 외부 모델 파일 로드
        load_btn = QPushButton("📁  외부 모델 파일 추가")
        load_btn.setObjectName("secondary")
        load_btn.clicked.connect(self._load_external_model)
        mg.addWidget(load_btn, 0, 2)

        mg.addWidget(QLabel("전이학습 가중치:"), 1, 0)
        self.weight_combo = QComboBox()
        mg.addWidget(self.weight_combo, 1, 1)

        mg.addWidget(QLabel("인코더 백본:"), 2, 0)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(list(ENCODER_MAP.keys()))
        mg.addWidget(self.encoder_combo, 2, 1)

        hint = QLabel("※ Transformer 계열 모델은 인코더 선택 무관")
        hint.setObjectName("hint")
        mg.addWidget(hint, 3, 1)
        mg.setColumnStretch(1, 1)
        main.addWidget(model_group)

        # ── 입력 설정 ─────────────────────────────────────────────
        input_group = QGroupBox("입력 설정  (데이터 로드 시 자동 감지 → 필요 시 수동 변경)")
        ig = QGridLayout(input_group)

        ig.addWidget(QLabel("입력 채널 수:"), 0, 0)
        self.in_channels = QSpinBox()
        self.in_channels.setRange(1, 500)
        self.in_channels.setValue(3)
        ig.addWidget(self.in_channels, 0, 1)

        ig.addWidget(QLabel("Input Size (H × W):"), 1, 0)
        size_h = QHBoxLayout()
        self.input_h = QSpinBox(); self.input_h.setRange(32, 2048); self.input_h.setValue(512)
        self.input_w = QSpinBox(); self.input_w.setRange(32, 2048); self.input_w.setValue(512)
        size_h.addWidget(self.input_h)
        size_h.addWidget(QLabel("×"))
        size_h.addWidget(self.input_w)
        size_h.addStretch()
        ig.addLayout(size_h, 1, 1)

        # 통합 다시 감지 버튼
        redetect_btn = QPushButton("🔄  데이터에서 다시 감지")
        redetect_btn.setObjectName("secondary")
        redetect_btn.clicked.connect(self._auto_detect_all)
        ig.addWidget(redetect_btn, 2, 0, 1, 2)

        ig.setColumnStretch(1, 1)
        main.addWidget(input_group)

        # ── 클래스 설정 ───────────────────────────────────────────
        cls_group = QGroupBox("클래스 설정 및 가중치")
        cv = QVBoxLayout(cls_group)

        ch = QHBoxLayout()
        ch.addWidget(QLabel("클래스 수:"))
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 256)
        self.num_classes.setValue(2)
        self.num_classes.valueChanged.connect(self._update_class_table)
        ch.addWidget(self.num_classes)
        ch.addStretch()
        cv.addLayout(ch)

        hint2 = QLabel("클래스 이름과 가중치를 직접 입력하세요 (가중치 합산 무관)")
        hint2.setObjectName("hint")
        cv.addWidget(hint2)

        self.class_table = QTableWidget(2, 3)
        self.class_table.setHorizontalHeaderLabels(["클래스 ID", "클래스 이름", "손실 가중치"])
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.class_table.setMaximumHeight(200)
        self._init_class_table(2)
        cv.addWidget(self.class_table)
        main.addWidget(cls_group)

        # ── 저장 ──────────────────────────────────────────────────
        save_btn = QPushButton("✔  모델 설정 저장")
        save_btn.clicked.connect(self._save_state)
        main.addWidget(save_btn)
        main.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._on_model_changed(self.model_combo.currentText())

    def _on_model_changed(self, model_name):
        self.weight_combo.clear()
        weights = PRETRAINED_WEIGHTS.get(model_name, ["없음 (scratch)"])
        self.weight_combo.addItems(weights)

    def _load_external_model(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "모델 파일 선택", "", "Python (*.py)")
        if path:
            name = os.path.splitext(os.path.basename(path))[0]
            self.model_combo.addItem(f"[외부] {name}")
            self.model_combo.setCurrentText(f"[외부] {name}")
            self.state["external_model_path"] = path

    def _auto_detect_all(self):
        """채널 + 크기 한 번에 자동 감지 (샘플 1개만 읽음)"""
        img_dir, ext = self._get_img_dir_and_ext()
        if not img_dir or not os.path.isdir(img_dir):
            print(f"[AutoDetect] 경로 없음 또는 유효하지 않음: '{img_dir}'")
            return
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(f".{ext}")]
        if not files:
            print(f"[AutoDetect] {img_dir} 에 .{ext} 파일 없음")
            return
        try:
            import numpy as np
            sample = os.path.join(img_dir, files[0])
            if ext == "mat":
                import scipy.io as sio
                d   = sio.loadmat(sample)
                arr = list(d.values())[-1]
            elif ext == "npy":
                arr = np.load(sample)
            else:
                from PIL import Image
                arr = np.array(Image.open(sample))
            # 채널
            ch = arr.shape[2] if arr.ndim == 3 else 1
            self.in_channels.setValue(ch)
            # 크기
            h, w = arr.shape[:2]
            self.input_h.setValue(h)
            self.input_w.setValue(w)
            print(f"[AutoDetect] {files[0]}  →  채널 {ch}, 크기 {h}x{w}")
        except Exception as e:
            print(f"[AutoDetect] 감지 실패: {e}")

    def _auto_detect_channels(self):
        """기존 호환용 — _auto_detect_all로 위임"""
        self._auto_detect_all()

    def _auto_detect_size(self):
        """기존 호환용 — _auto_detect_all로 위임"""
        self._auto_detect_all()

    def _init_class_table(self, n):
        self.class_table.setRowCount(n)
        for i in range(n):
            self.class_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.class_table.setItem(i, 1, QTableWidgetItem(f"Class {i}"))
            self.class_table.setItem(i, 2, QTableWidgetItem("1.0"))

    def _update_class_table(self, n):
        old = self.class_table.rowCount()
        self.class_table.setRowCount(n)
        for i in range(old, n):
            self.class_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.class_table.setItem(i, 1, QTableWidgetItem(f"Class {i}"))
            self.class_table.setItem(i, 2, QTableWidgetItem("1.0"))

    def _save_state(self):
        weights = []
        names = []
        for i in range(self.class_table.rowCount()):
            name_item = self.class_table.item(i, 1)
            w_item    = self.class_table.item(i, 2)
            names.append(name_item.text() if name_item else f"Class {i}")
            try:
                weights.append(float(w_item.text()) if w_item else 1.0)
            except ValueError:
                weights.append(1.0)

        # 표시명 → smp 실제 encoder 이름으로 변환
        display_enc = self.encoder_combo.currentText()
        real_enc    = ENCODER_MAP.get(display_enc, "resnet34")

        self.state["model_name"]    = self.model_combo.currentText()
        self.state["pretrained"]    = self.weight_combo.currentText()
        self.state["encoder"]       = real_enc          # ← smp 실제 키값
        self.state["in_channels"]   = self.in_channels.value()
        self.state["input_h"]       = self.input_h.value()
        self.state["input_w"]       = self.input_w.value()
        self.state["num_classes"]   = self.num_classes.value()
        self.state["class_names"]   = names
        self.state["class_weights"] = weights
        print("[ModelTab] 설정 저장:",
              f"model={self.state['model_name']} | encoder={real_enc} | "
              f"ch={self.state['in_channels']} | classes={self.state['num_classes']}")

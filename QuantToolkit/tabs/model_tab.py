import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea, QLineEdit, QRadioButton, QButtonGroup,
    QFrame
)


# 회귀 모델 — 두 가지 모드
PATCH_MODELS = [
    "CNN-2D-Simple",          # 간단한 2D CNN (직접 정의)
    "CNN-2D-ResNet18",        # ResNet18 백본
    "CNN-2D-ResNet34",
    "CNN-2D-ResNet50",
    "CNN-2D-EfficientNet-B0",
    "CNN-2D-EfficientNet-B2",
    "CNN-1D-Spectral",        # 1D CNN (스펙트럼만)
    "MLP-Spectral",           # MLP (스펙트럼만)
]

DENSE_MODELS = [
    "UNet",
    "UNet++",
    "MAnet",
    "FPN",
    "Linknet",
    "PAN",
    "DeepLabV3+",
]

ENCODER_OPTIONS = [
    "resnet18", "resnet34", "resnet50", "resnet101",
    "resnext50_32x4d", "efficientnet-b0", "efficientnet-b2",
    "efficientnet-b4", "mit_b0", "mit_b2", "mit_b4",
    "mobilenet_v2",
]


class ModelTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self._data_tab = None
        self._build()

    def set_data_tab(self, data_tab):
        self._data_tab = data_tab
        if hasattr(data_tab, 'train_img_edit'):
            data_tab.train_img_edit.textChanged.connect(self._on_data_path_changed)
        if hasattr(data_tab, 'img_ext'):
            data_tab.img_ext.currentTextChanged.connect(self._on_data_path_changed)

    def _on_data_path_changed(self, *args):
        img_dir, ext = self._get_img_dir_and_ext()
        if img_dir and os.path.isdir(img_dir):
            self._auto_detect_all()

    def _get_img_dir_and_ext(self):
        img_dir = self.state.get("train_img_dir", "").strip()
        ext     = self.state.get("img_ext", "mat")
        if not img_dir and self._data_tab is not None:
            img_dir = self._data_tab.train_img_edit.text().strip()
            ext     = self._data_tab.img_ext.currentText()
        return img_dir, ext

    def _build(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        main = QVBoxLayout(inner)
        main.setSpacing(14); main.setContentsMargins(20, 20, 20, 20)

        # ── 학습 모드 ──────────────────────────────────────────
        mode_g = QGroupBox("학습 모드 선택")
        mv = QVBoxLayout(mode_g)
        self.mode_btns = QButtonGroup()
        self.rb_patch = QRadioButton(
            "패치 기반 (Patch CNN)  —  픽셀별 농도값 예측  ·  대용량 데이터 가능"
        )
        self.rb_dense = QRadioButton(
            "Dense 회귀 (U-Net)  —  전체 이미지 한 번에 농도맵 출력  ·  공간 정보 활용"
        )
        self.rb_patch.setChecked(True)
        self.mode_btns.addButton(self.rb_patch, 0)
        self.mode_btns.addButton(self.rb_dense, 1)
        mv.addWidget(self.rb_patch); mv.addWidget(self.rb_dense)
        self.mode_btns.buttonClicked.connect(self._on_mode_changed)
        main.addWidget(mode_g)

        # ── 모델 선택 ───────────────────────────────────────────
        model_g = QGroupBox("모델 구조")
        mg = QGridLayout(model_g)
        mg.addWidget(QLabel("모델:"), 0, 0)
        self.model_combo = QComboBox()
        mg.addWidget(self.model_combo, 0, 1)

        mg.addWidget(QLabel("인코더 (Dense 전용):"), 1, 0)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(ENCODER_OPTIONS)
        self.encoder_combo.setCurrentText("resnet34")
        mg.addWidget(self.encoder_combo, 1, 1)

        mg.addWidget(QLabel("Pretrained:"), 2, 0)
        self.pretrained_combo = QComboBox()
        self.pretrained_combo.addItems(["없음 (scratch)", "imagenet"])
        mg.addWidget(self.pretrained_combo, 2, 1)

        mg.setColumnStretch(1, 1)
        main.addWidget(model_g)

        # ── 입력 설정 ───────────────────────────────────────────
        input_g = QGroupBox("입력 설정  (데이터 로드 시 자동 감지 / 수동 변경 가능)")
        ig = QGridLayout(input_g)

        ig.addWidget(QLabel("입력 채널 수:"), 0, 0)
        self.in_channels = QSpinBox()
        self.in_channels.setRange(1, 500); self.in_channels.setValue(6)
        ig.addWidget(self.in_channels, 0, 1)

        ig.addWidget(QLabel("입력 크기 (H × W):"), 1, 0)
        sh = QHBoxLayout()
        self.input_h = QSpinBox(); self.input_h.setRange(16, 4096); self.input_h.setValue(512)
        self.input_w = QSpinBox(); self.input_w.setRange(16, 4096); self.input_w.setValue(512)
        sh.addWidget(self.input_h); sh.addWidget(QLabel("×")); sh.addWidget(self.input_w); sh.addStretch()
        ig.addLayout(sh, 1, 1)

        # 패치 크기 (Patch 모드 전용)
        self.patch_size_label = QLabel("패치 크기 (홀수):")
        ig.addWidget(self.patch_size_label, 2, 0)
        self.patch_size = QSpinBox(); self.patch_size.setRange(3, 63); self.patch_size.setValue(3)
        self.patch_size.setSingleStep(2)
        ig.addWidget(self.patch_size, 2, 1)

        self.max_patches_label = QLabel("이미지당 최대 패치 수:")
        ig.addWidget(self.max_patches_label, 3, 0)
        self.max_patches = QSpinBox(); self.max_patches.setRange(100, 1000000); self.max_patches.setValue(10000)
        self.max_patches.setSingleStep(1000)
        ig.addWidget(self.max_patches, 3, 1)

        redetect_btn = QPushButton("🔄  데이터에서 다시 감지"); redetect_btn.setObjectName("secondary")
        redetect_btn.clicked.connect(self._auto_detect_all)
        ig.addWidget(redetect_btn, 4, 0, 1, 2)
        ig.setColumnStretch(1, 1)
        main.addWidget(input_g)

        # ── 전처리 / Augmentation ─────────────────────────────
        pre_g = QGroupBox("전처리 및 Augmentation")
        pg = QGridLayout(pre_g)

        pg.addWidget(QLabel("정규화 방식:"), 0, 0)
        self.norm_combo = QComboBox()
        self.norm_combo.addItems([
            "없음 (이미 정규화된 데이터)",
            "SNV (픽셀별 스펙트럼 정규화)",
            "Min-Max (채널별, [0,1] 매핑)",
            "Z-score (채널별, 평균0/std1)",
        ])
        pg.addWidget(self.norm_combo, 0, 1)
        norm_hint = QLabel(
            "Scratch 학습 → SNV 권장 (분광 정보 강조)\n"
            "Pretrained 사용 → Min-Max 또는 Z-score 권장 (가중치 호환성)"
        )
        norm_hint.setObjectName("hint")
        pg.addWidget(norm_hint, 1, 0, 1, 2)

        self.use_aug = QCheckBox("Augmentation 적용  (flip/rotate + Gaussian noise + spectral scaling)")
        self.use_aug.setChecked(True)
        pg.addWidget(self.use_aug, 2, 0, 1, 2)
        aug_hint = QLabel("※ 데이터가 적을 때 (수십장 이하) 권장")
        aug_hint.setObjectName("hint")
        pg.addWidget(aug_hint, 3, 0, 1, 2)
        pg.setColumnStretch(1, 1)
        main.addWidget(pre_g)

        # ── 회귀 출력 설정 ─────────────────────────────────────
        out_g = QGroupBox("회귀 출력 설정")
        og = QGridLayout(out_g)
        og.addWidget(QLabel("출력 활성화 함수:"), 0, 0)
        self.activation = QComboBox()
        self.activation.addItems(["없음 (linear)", "sigmoid", "tanh", "ReLU"])
        og.addWidget(self.activation, 0, 1)
        hint_a = QLabel("회귀는 보통 '없음' 권장 (sigmoid는 0/1 근처 saturation 위험)")
        hint_a.setObjectName("hint")
        og.addWidget(hint_a, 1, 1)

        og.addWidget(QLabel("출력 범위 (선택):"), 2, 0)
        rh = QHBoxLayout()
        self.out_min = QDoubleSpinBox(); self.out_min.setRange(-1000, 1000); self.out_min.setDecimals(3); self.out_min.setValue(0.0)
        self.out_max = QDoubleSpinBox(); self.out_max.setRange(-1000, 1000); self.out_max.setDecimals(3); self.out_max.setValue(1.0)
        rh.addWidget(self.out_min); rh.addWidget(QLabel("~")); rh.addWidget(self.out_max); rh.addStretch()
        og.addLayout(rh, 2, 1)
        og.setColumnStretch(1, 1)
        main.addWidget(out_g)

        save_btn = QPushButton("✔  모델 설정 저장")
        save_btn.clicked.connect(self._save_state)
        main.addWidget(save_btn)
        main.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._on_mode_changed()

    def _on_mode_changed(self, btn=None):
        is_patch = self.mode_btns.checkedId() == 0
        self.model_combo.clear()
        if is_patch:
            self.model_combo.addItems(PATCH_MODELS)
            self.model_combo.setCurrentText("CNN-2D-Simple")
        else:
            self.model_combo.addItems(DENSE_MODELS)
            self.model_combo.setCurrentText("UNet")

        self.encoder_combo.setEnabled(not is_patch)

        # 패치 크기/max_patches는 Patch 모드만
        self.patch_size_label.setVisible(is_patch)
        self.patch_size.setVisible(is_patch)
        self.max_patches_label.setVisible(is_patch)
        self.max_patches.setVisible(is_patch)

    def _auto_detect_all(self):
        img_dir, ext = self._get_img_dir_and_ext()
        if not img_dir or not os.path.isdir(img_dir):
            return
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(f".{ext}")]
        if not files: return
        try:
            import numpy as np
            sample = os.path.join(img_dir, files[0])
            if ext == "mat":
                import scipy.io as sio
                d = sio.loadmat(sample)
                arr = list(d.values())[-1]
            elif ext == "npy":
                arr = np.load(sample)
            else:
                from PIL import Image
                arr = np.array(Image.open(sample))
            ch = arr.shape[2] if arr.ndim == 3 else 1
            self.in_channels.setValue(ch)
            h, w = arr.shape[:2]
            self.input_h.setValue(h); self.input_w.setValue(w)
            print(f"[AutoDetect] {files[0]} → 채널 {ch}, 크기 {h}x{w}")
        except Exception as e:
            print(f"[AutoDetect] 감지 실패: {e}")

    def _save_state(self):
        is_patch = self.mode_btns.checkedId() == 0
        norm_map = {
            0: "none",
            1: "snv",
            2: "minmax",
            3: "zscore",
        }
        norm_type = norm_map[self.norm_combo.currentIndex()]
        self.state["mode"]              = "patch" if is_patch else "dense"
        self.state["model_name"]        = self.model_combo.currentText()
        self.state["encoder"]           = self.encoder_combo.currentText()
        self.state["pretrained"]        = self.pretrained_combo.currentText()
        self.state["in_channels"]       = self.in_channels.value()
        self.state["input_h"]           = self.input_h.value()
        self.state["input_w"]           = self.input_w.value()
        self.state["patch_size"]        = self.patch_size.value()
        self.state["max_patches_per_file"] = self.max_patches.value()
        self.state["norm_type"]         = norm_type
        # 호환성 위해 use_snv도 유지
        self.state["use_snv"]           = (norm_type == "snv")
        self.state["use_augmentation"]  = self.use_aug.isChecked()
        self.state["activation"]        = self.activation.currentText()
        self.state["out_min"]           = self.out_min.value()
        self.state["out_max"]           = self.out_max.value()
        print(f"[ModelTab] {self.state['mode']} | {self.state['model_name']} | "
              f"norm={norm_type} | Aug={self.state['use_augmentation']}")

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QGroupBox, QComboBox, QListWidget,
    QListWidgetItem, QGridLayout, QSpinBox, QRadioButton,
    QButtonGroup, QAbstractItemView, QSplitter, QCheckBox,
    QDoubleSpinBox, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt


SUPPORTED_EXTS = ["png", "jpg", "jpeg", "tiff", "tif", "mat", "npy"]


class DataTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self._build()

    def _build(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        main = QVBoxLayout(inner)
        main.setSpacing(14); main.setContentsMargins(20, 20, 20, 20)

        # ── 확장자 ──────────────────────────────────────────────
        ext_g = QGroupBox("파일 확장자 설정")
        eh = QHBoxLayout(ext_g)
        eh.addWidget(QLabel("이미지 확장자:"))
        self.img_ext = QComboBox(); self.img_ext.addItems(SUPPORTED_EXTS)
        self.img_ext.setCurrentText("mat")
        eh.addWidget(self.img_ext); eh.addSpacing(20)
        eh.addWidget(QLabel("농도 라벨 확장자:"))
        self.lbl_ext = QComboBox(); self.lbl_ext.addItems(SUPPORTED_EXTS)
        self.lbl_ext.setCurrentText("png")
        eh.addWidget(self.lbl_ext); eh.addStretch()
        main.addWidget(ext_g)

        # ── 농도 라벨 처리 ─────────────────────────────────────
        conv_g = QGroupBox("농도 라벨 변환  (라벨 정수값 → 실제 농도)")
        cg = QGridLayout(conv_g)
        cg.addWidget(QLabel("배경(무시) 값:"), 0, 0)
        self.bg_value = QSpinBox()
        self.bg_value.setRange(-1, 255); self.bg_value.setValue(0)
        cg.addWidget(self.bg_value, 0, 1)
        cg.addWidget(QLabel("배경 픽셀은 학습/평가에서 자동 제외"), 0, 2)

        cg.addWidget(QLabel("라벨 → 농도 변환식:"), 1, 0)
        formula_h = QHBoxLayout()
        formula_h.addWidget(QLabel("(라벨"))
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-100, 100); self.offset.setDecimals(2); self.offset.setValue(-1.0)
        self.offset.setPrefix(" ")
        formula_h.addWidget(self.offset)
        formula_h.addWidget(QLabel(") /"))
        self.divisor = QDoubleSpinBox()
        self.divisor.setRange(0.001, 1000); self.divisor.setDecimals(3); self.divisor.setValue(10.0)
        formula_h.addWidget(self.divisor)
        formula_h.addStretch()
        cg.addLayout(formula_h, 1, 1, 1, 2)

        hint_c = QLabel("예: 라벨 1~11 → (라벨 - 1) / 10 = [0.0, 0.1, ..., 1.0]")
        hint_c.setObjectName("hint")
        cg.addWidget(hint_c, 2, 1, 1, 2)
        main.addWidget(conv_g)

        # ── 학습 방식 ───────────────────────────────────────────
        split_g = QGroupBox("학습 방식")
        sv = QVBoxLayout(split_g)
        self.split_btns = QButtonGroup()
        self.rb_manual  = QRadioButton("Train / Val 직접 분리")
        self.rb_holdout = QRadioButton("Hold-out  (전체에서 랜덤 분할)")
        self.rb_kfold   = QRadioButton("K-Fold Cross Validation")
        self.rb_manual.setChecked(True)
        for i, rb in enumerate([self.rb_manual, self.rb_holdout, self.rb_kfold]):
            self.split_btns.addButton(rb, i)
            sv.addWidget(rb)
        self.split_btns.buttonClicked.connect(self._on_split_changed)
        main.addWidget(split_g)

        # ── 경로 ────────────────────────────────────────────────
        path_g = QGroupBox("데이터 경로")
        pg = QGridLayout(path_g); pg.setSpacing(8)

        pg.addWidget(QLabel("Train 이미지:"), 0, 0)
        self.train_img_edit = QLineEdit(); self.train_img_edit.setPlaceholderText("폴더 경로")
        pg.addWidget(self.train_img_edit, 0, 1)
        btn_ti = QPushButton("찾기"); btn_ti.setObjectName("secondary")
        btn_ti.clicked.connect(lambda: self._browse(self.train_img_edit))
        pg.addWidget(btn_ti, 0, 2)

        pg.addWidget(QLabel("Train 농도 라벨:"), 1, 0)
        self.train_lbl_edit = QLineEdit(); self.train_lbl_edit.setPlaceholderText("폴더 경로")
        pg.addWidget(self.train_lbl_edit, 1, 1)
        btn_tl = QPushButton("찾기"); btn_tl.setObjectName("secondary")
        btn_tl.clicked.connect(lambda: self._browse(self.train_lbl_edit))
        pg.addWidget(btn_tl, 1, 2)

        self.val_img_label = QLabel("Val 이미지:")
        pg.addWidget(self.val_img_label, 2, 0)
        self.val_img_edit = QLineEdit()
        pg.addWidget(self.val_img_edit, 2, 1)
        self.btn_vi = QPushButton("찾기"); self.btn_vi.setObjectName("secondary")
        self.btn_vi.clicked.connect(lambda: self._browse(self.val_img_edit))
        pg.addWidget(self.btn_vi, 2, 2)

        self.val_lbl_label = QLabel("Val 농도 라벨:")
        pg.addWidget(self.val_lbl_label, 3, 0)
        self.val_lbl_edit = QLineEdit()
        pg.addWidget(self.val_lbl_edit, 3, 1)
        self.btn_vl = QPushButton("찾기"); self.btn_vl.setObjectName("secondary")
        self.btn_vl.clicked.connect(lambda: self._browse(self.val_lbl_edit))
        pg.addWidget(self.btn_vl, 3, 2)

        self.holdout_label = QLabel("Val 비율:")
        pg.addWidget(self.holdout_label, 4, 0)
        self.holdout_spin = QDoubleSpinBox()
        self.holdout_spin.setRange(0.05, 0.5); self.holdout_spin.setSingleStep(0.05)
        self.holdout_spin.setValue(0.2)
        pg.addWidget(self.holdout_spin, 4, 1)

        self.kfold_label = QLabel("Fold 수 (K):")
        pg.addWidget(self.kfold_label, 5, 0)
        self.kfold_spin = QSpinBox(); self.kfold_spin.setRange(3, 20); self.kfold_spin.setValue(5)
        pg.addWidget(self.kfold_spin, 5, 1)

        pg.setColumnStretch(1, 1)
        main.addWidget(path_g)

        # ── 파일 목록 ──────────────────────────────────────────
        list_g = QGroupBox("파일 목록 확인")
        lv = QVBoxLayout(list_g)
        scan_btn = QPushButton("📂  파일 스캔")
        scan_btn.clicked.connect(self._scan_files)
        lv.addWidget(scan_btn)

        two_col = QHBoxLayout()
        left_col = QVBoxLayout()
        ih = QLabel("🖼  이미지")
        ih.setStyleSheet("font-weight:700; color:#16a34a; font-size:12px;")
        left_col.addWidget(ih)
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(140)
        left_col.addWidget(self.file_list)
        self.img_count_label = QLabel("0개"); self.img_count_label.setObjectName("hint")
        left_col.addWidget(self.img_count_label)
        two_col.addLayout(left_col)

        right_col = QVBoxLayout()
        lh = QLabel("📊  농도 라벨")
        lh.setStyleSheet("font-weight:700; color:#059669; font-size:12px;")
        right_col.addWidget(lh)
        self.label_list = QListWidget()
        self.label_list.setMinimumHeight(140)
        right_col.addWidget(self.label_list)
        self.lbl_count_label = QLabel("0개"); self.lbl_count_label.setObjectName("hint")
        right_col.addWidget(self.lbl_count_label)
        two_col.addLayout(right_col)

        lv.addLayout(two_col)
        self.file_count_label = QLabel("이미지 0개 / 라벨 0개")
        self.file_count_label.setObjectName("hint")
        lv.addWidget(self.file_count_label)
        main.addWidget(list_g)

        save_btn = QPushButton("✔  데이터 설정 저장")
        save_btn.clicked.connect(self._save_state)
        main.addWidget(save_btn)
        main.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._on_split_changed()

    def _on_split_changed(self, btn=None):
        mode = self.split_btns.checkedId()
        manual = (mode == 0); holdout = (mode == 1); kfold = (mode == 2)
        for w in [self.val_img_label, self.val_img_edit, self.btn_vi,
                  self.val_lbl_label, self.val_lbl_edit, self.btn_vl]:
            w.setVisible(manual)
        self.holdout_label.setVisible(holdout)
        self.holdout_spin.setVisible(holdout)
        self.kfold_label.setVisible(kfold)
        self.kfold_spin.setVisible(kfold)

    def _browse(self, edit):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if path: edit.setText(path)

    def _scan_files(self):
        self.file_list.clear(); self.label_list.clear()
        img_path = self.train_img_edit.text().strip()
        lbl_path = self.train_lbl_edit.text().strip()
        img_ext  = self.img_ext.currentText().lower()
        lbl_ext  = self.lbl_ext.currentText().lower()

        if img_path and os.path.isdir(img_path):
            img_files = sorted([f for f in os.listdir(img_path)
                                if f.lower().endswith(f".{img_ext}")])
            for f in img_files:
                self.file_list.addItem(QListWidgetItem(f))
            self.img_count_label.setText(f"{len(img_files)}개")
        else:
            self.img_count_label.setText("경로 없음")

        if lbl_path and os.path.isdir(lbl_path):
            lbl_files = sorted([f for f in os.listdir(lbl_path)
                                if f.lower().endswith(f".{lbl_ext}")])
            for f in lbl_files:
                self.label_list.addItem(QListWidgetItem(f))
            self.lbl_count_label.setText(f"{len(lbl_files)}개")
        else:
            self.lbl_count_label.setText("경로 없음")

        self.file_count_label.setText(
            f"이미지 {self.file_list.count()}개 / 라벨 {self.label_list.count()}개"
        )

    def _save_state(self):
        mode_map = {0: "manual", 1: "holdout", 2: "kfold"}
        self.state["split_mode"]     = mode_map[self.split_btns.checkedId()]
        self.state["train_img_dir"]  = self.train_img_edit.text().strip()
        self.state["train_lbl_dir"]  = self.train_lbl_edit.text().strip()
        self.state["val_img_dir"]    = self.val_img_edit.text().strip()
        self.state["val_lbl_dir"]    = self.val_lbl_edit.text().strip()
        self.state["holdout_ratio"]  = self.holdout_spin.value()
        self.state["kfold_k"]        = self.kfold_spin.value()
        self.state["img_ext"]        = self.img_ext.currentText()
        self.state["lbl_ext"]        = self.lbl_ext.currentText()
        # 회귀 전용
        self.state["bg_value"]       = self.bg_value.value()
        self.state["label_offset"]   = self.offset.value()
        self.state["label_divisor"]  = self.divisor.value()
        print("[DataTab] 저장 완료:", self.state)

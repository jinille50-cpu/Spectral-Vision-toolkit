import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QGroupBox, QComboBox, QListWidget,
    QListWidgetItem, QGridLayout, QSpinBox, QRadioButton,
    QButtonGroup, QAbstractItemView, QSplitter, QCheckBox,
    QDoubleSpinBox, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt


SUPPORTED_EXTS = ["png", "jpg", "jpeg", "tiff", "tif", "mat", "npy", "hdr"]


class DataTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self._build()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)

        main = QVBoxLayout(inner)
        main.setSpacing(14)
        main.setContentsMargins(20, 20, 20, 20)

        # ── 확장자 선택 ──────────────────────────────────────────
        ext_group = QGroupBox("파일 확장자 설정")
        ext_layout = QHBoxLayout(ext_group)
        ext_layout.addWidget(QLabel("이미지 확장자:"))
        self.img_ext = QComboBox()
        self.img_ext.addItems(SUPPORTED_EXTS)
        ext_layout.addWidget(self.img_ext)
        ext_layout.addSpacing(20)
        ext_layout.addWidget(QLabel("라벨 확장자:"))
        self.lbl_ext = QComboBox()
        self.lbl_ext.addItems(SUPPORTED_EXTS)
        self.lbl_ext.setCurrentText("png")
        ext_layout.addWidget(self.lbl_ext)
        ext_layout.addStretch()
        main.addWidget(ext_group)

        # ── 학습 방식 선택 ────────────────────────────────────────
        split_group = QGroupBox("학습 방식")
        split_v = QVBoxLayout(split_group)

        self.split_btns = QButtonGroup()
        self.rb_manual   = QRadioButton("Train / Val 직접 분리  (폴더 따로 지정)")
        self.rb_holdout  = QRadioButton("Hold-out  (전체 데이터에서 랜덤 분할)")
        self.rb_kfold    = QRadioButton("Stratified K-Fold  (데이터 적을 때 권장)")
        self.rb_manual.setChecked(True)
        for i, rb in enumerate([self.rb_manual, self.rb_holdout, self.rb_kfold]):
            self.split_btns.addButton(rb, i)
            split_v.addWidget(rb)

        self.split_btns.buttonClicked.connect(self._on_split_changed)
        main.addWidget(split_group)

        # ── 경로 설정 ─────────────────────────────────────────────
        path_group = QGroupBox("데이터 경로")
        path_grid = QGridLayout(path_group)
        path_grid.setSpacing(8)

        # Train image
        path_grid.addWidget(QLabel("Train 이미지:"), 0, 0)
        self.train_img_edit = QLineEdit(); self.train_img_edit.setPlaceholderText("폴더 경로")
        path_grid.addWidget(self.train_img_edit, 0, 1)
        btn_ti = QPushButton("찾기"); btn_ti.setObjectName("secondary")
        btn_ti.clicked.connect(lambda: self._browse(self.train_img_edit))
        path_grid.addWidget(btn_ti, 0, 2)

        # Train label
        path_grid.addWidget(QLabel("Train 라벨:"), 1, 0)
        self.train_lbl_edit = QLineEdit(); self.train_lbl_edit.setPlaceholderText("폴더 경로")
        path_grid.addWidget(self.train_lbl_edit, 1, 1)
        btn_tl = QPushButton("찾기"); btn_tl.setObjectName("secondary")
        btn_tl.clicked.connect(lambda: self._browse(self.train_lbl_edit))
        path_grid.addWidget(btn_tl, 1, 2)

        # Val image (manual 전용)
        self.val_img_label = QLabel("Val 이미지:")
        path_grid.addWidget(self.val_img_label, 2, 0)
        self.val_img_edit = QLineEdit(); self.val_img_edit.setPlaceholderText("폴더 경로")
        path_grid.addWidget(self.val_img_edit, 2, 1)
        self.btn_vi = QPushButton("찾기"); self.btn_vi.setObjectName("secondary")
        self.btn_vi.clicked.connect(lambda: self._browse(self.val_img_edit))
        path_grid.addWidget(self.btn_vi, 2, 2)

        # Val label (manual 전용)
        self.val_lbl_label = QLabel("Val 라벨:")
        path_grid.addWidget(self.val_lbl_label, 3, 0)
        self.val_lbl_edit = QLineEdit(); self.val_lbl_edit.setPlaceholderText("폴더 경로")
        path_grid.addWidget(self.val_lbl_edit, 3, 1)
        self.btn_vl = QPushButton("찾기"); self.btn_vl.setObjectName("secondary")
        self.btn_vl.clicked.connect(lambda: self._browse(self.val_lbl_edit))
        path_grid.addWidget(self.btn_vl, 3, 2)

        # Hold-out 비율 (holdout 전용)
        self.holdout_label = QLabel("Val 비율:")
        path_grid.addWidget(self.holdout_label, 4, 0)
        self.holdout_spin = QDoubleSpinBox()
        self.holdout_spin.setRange(0.05, 0.5); self.holdout_spin.setSingleStep(0.05)
        self.holdout_spin.setValue(0.2); self.holdout_spin.setPrefix("  "); self.holdout_spin.setSuffix("  (20% = 0.2)")
        path_grid.addWidget(self.holdout_spin, 4, 1)

        # K-Fold K값 (kfold 전용)
        self.kfold_label = QLabel("Fold 수 (K):")
        path_grid.addWidget(self.kfold_label, 5, 0)
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(3, 20); self.kfold_spin.setValue(5)
        path_grid.addWidget(self.kfold_spin, 5, 1)

        # Test 이미지/라벨은 ④ 평가탭, ⑤ 추론탭에서 직접 지정 → 제거됨

        path_grid.setColumnStretch(1, 1)
        main.addWidget(path_group)

        # ── 파일 목록 확인 ────────────────────────────────────────
        list_group = QGroupBox("파일 목록 확인 및 순서 조정")
        list_v = QVBoxLayout(list_group)

        scan_btn = QPushButton("📂  파일 스캔")
        scan_btn.clicked.connect(self._scan_files)
        list_v.addWidget(scan_btn)

        # 좌(이미지) / 우(라벨) 2열 레이아웃
        two_col = QHBoxLayout()

        # 왼쪽 — 이미지
        left_col = QVBoxLayout()
        img_header = QLabel("🖼  이미지")
        img_header.setStyleSheet("font-weight:700; color:#2563eb; font-size:12px;")
        left_col.addWidget(img_header)
        self.file_list = QListWidget()
        self.file_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setMinimumHeight(160)
        left_col.addWidget(self.file_list)
        self.img_count_label = QLabel("0개")
        self.img_count_label.setObjectName("hint")
        left_col.addWidget(self.img_count_label)
        two_col.addLayout(left_col)

        # 오른쪽 — 라벨
        right_col = QVBoxLayout()
        lbl_header = QLabel("🏷  라벨")
        lbl_header.setStyleSheet("font-weight:700; color:#059669; font-size:12px;")
        right_col.addWidget(lbl_header)
        self.label_list = QListWidget()
        self.label_list.setMinimumHeight(160)
        right_col.addWidget(self.label_list)
        self.lbl_count_label = QLabel("0개")
        self.lbl_count_label.setObjectName("hint")
        right_col.addWidget(self.lbl_count_label)
        two_col.addLayout(right_col)

        list_v.addLayout(two_col)

        # 순서 조정 버튼 (이미지 기준)
        btn_h = QHBoxLayout()
        up_btn = QPushButton("▲ 위로"); up_btn.setObjectName("secondary")
        up_btn.clicked.connect(self._move_up)
        down_btn = QPushButton("▼ 아래로"); down_btn.setObjectName("secondary")
        down_btn.clicked.connect(self._move_down)
        del_btn = QPushButton("✕ 제거"); del_btn.setObjectName("secondary")
        del_btn.clicked.connect(self._remove_selected)
        btn_h.addWidget(up_btn); btn_h.addWidget(down_btn); btn_h.addWidget(del_btn)
        btn_h.addStretch()
        list_v.addLayout(btn_h)

        self.file_count_label = QLabel("이미지 0개 / 라벨 0개")
        self.file_count_label.setObjectName("hint")
        list_v.addWidget(self.file_count_label)
        main.addWidget(list_group)

        # ── 저장 ──────────────────────────────────────────────────
        save_btn = QPushButton("✔  데이터 설정 저장")
        save_btn.clicked.connect(self._save_state)
        main.addWidget(save_btn)
        main.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._on_split_changed()

    # ── helpers ──────────────────────────────────────────────────
    def _on_split_changed(self, btn=None):
        mode = self.split_btns.checkedId()  # 0=manual, 1=holdout, 2=kfold
        manual  = (mode == 0)
        holdout = (mode == 1)
        kfold   = (mode == 2)

        for w in [self.val_img_label, self.val_img_edit, self.btn_vi,
                  self.val_lbl_label, self.val_lbl_edit, self.btn_vl]:
            w.setVisible(manual)
        self.holdout_label.setVisible(holdout)
        self.holdout_spin.setVisible(holdout)
        self.kfold_label.setVisible(kfold)
        self.kfold_spin.setVisible(kfold)

    def _browse(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if path:
            edit.setText(path)

    def _scan_files(self):
        self.file_list.clear()
        self.label_list.clear()

        img_path = self.train_img_edit.text().strip()
        lbl_path = self.train_lbl_edit.text().strip()
        img_ext  = self.img_ext.currentText().lower()
        lbl_ext  = self.lbl_ext.currentText().lower()

        # 이미지 스캔
        if img_path and os.path.isdir(img_path):
            img_files = sorted([f for f in os.listdir(img_path)
                                 if f.lower().endswith(f".{img_ext}")])
            for f in img_files:
                self.file_list.addItem(QListWidgetItem(f))
            self.img_count_label.setText(f"{len(img_files)}개")
        else:
            img_files = []
            self.img_count_label.setText("경로 없음")

        # 라벨 스캔
        if lbl_path and os.path.isdir(lbl_path):
            lbl_files = sorted([f for f in os.listdir(lbl_path)
                                 if f.lower().endswith(f".{lbl_ext}")])
            for f in lbl_files:
                self.label_list.addItem(QListWidgetItem(f))
            self.lbl_count_label.setText(f"{len(lbl_files)}개")
        else:
            lbl_files = []
            self.lbl_count_label.setText("경로 없음")

        self.file_count_label.setText(
            f"이미지 {self.file_list.count()}개 / 라벨 {self.label_list.count()}개"
        )

    def _move_up(self):
        row = self.file_list.currentRow()
        if row > 0:
            item = self.file_list.takeItem(row)
            self.file_list.insertItem(row - 1, item)
            self.file_list.setCurrentRow(row - 1)

    def _move_down(self):
        row = self.file_list.currentRow()
        if row < self.file_list.count() - 1:
            item = self.file_list.takeItem(row)
            self.file_list.insertItem(row + 1, item)
            self.file_list.setCurrentRow(row + 1)

    def _remove_selected(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self.file_count_label.setText(f"파일 {self.file_list.count()}개")

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
        self.state["file_order"]     = [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
        ]
        print("[DataTab] 설정 저장 완료:", self.state)

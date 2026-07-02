import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QGroupBox, QComboBox, QListWidget,
    QListWidgetItem, QGridLayout, QSpinBox, QRadioButton,
    QButtonGroup, QDoubleSpinBox, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QTextEdit, QSplitter
)
from PyQt5.QtCore import Qt

# YOLO Detection: 이미지는 png/tiff/tif, 라벨은 txt 고정
IMG_EXTS = ["png", "tiff", "tif"]


class DataTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self._build()

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽: 설정 ───────────────────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        main = QVBoxLayout(inner)
        main.setSpacing(14); main.setContentsMargins(20, 20, 20, 20)

        # 확장자
        ext_group = QGroupBox("파일 형식")
        ext_layout = QHBoxLayout(ext_group)
        ext_layout.addWidget(QLabel("이미지 확장자:"))
        self.img_ext = QComboBox(); self.img_ext.addItems(IMG_EXTS)
        ext_layout.addWidget(self.img_ext)
        ext_layout.addSpacing(20)
        ext_layout.addWidget(QLabel("라벨 형식:"))
        lbl_fixed = QLabel("txt  (YOLO Detection)")
        lbl_fixed.setStyleSheet("font-weight:700; color:#059669;")
        ext_layout.addWidget(lbl_fixed)
        ext_layout.addStretch()
        main.addWidget(ext_group)

        # 학습 방식
        split_group = QGroupBox("학습 방식")
        split_v = QVBoxLayout(split_group)
        self.split_btns = QButtonGroup()
        self.rb_manual  = QRadioButton("Train / Val 직접 분리  (폴더 따로 지정)")
        self.rb_holdout = QRadioButton("Hold-out  (Train 데이터에서 랜덤 분할)")
        self.rb_kfold   = QRadioButton("K-Fold  (K회 교차검증 + 전체 데이터 재학습)")
        self.rb_manual.setChecked(True)
        for i, rb in enumerate([self.rb_manual, self.rb_holdout, self.rb_kfold]):
            self.split_btns.addButton(rb, i)
            split_v.addWidget(rb)
        self.split_btns.buttonClicked.connect(self._on_split_changed)
        main.addWidget(split_group)

        # 경로
        path_group = QGroupBox("데이터 경로")
        pg = QGridLayout(path_group); pg.setSpacing(8)

        pg.addWidget(QLabel("Train 이미지:"), 0, 0)
        self.train_img_edit = QLineEdit(); self.train_img_edit.setPlaceholderText("이미지 폴더")
        pg.addWidget(self.train_img_edit, 0, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary")
        b.clicked.connect(lambda: self._browse(self.train_img_edit)); pg.addWidget(b, 0, 2)

        pg.addWidget(QLabel("Train 라벨(txt):"), 1, 0)
        self.train_lbl_edit = QLineEdit(); self.train_lbl_edit.setPlaceholderText("txt 라벨 폴더")
        pg.addWidget(self.train_lbl_edit, 1, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary")
        b.clicked.connect(lambda: self._browse(self.train_lbl_edit)); pg.addWidget(b, 1, 2)

        # Val (manual 전용)
        self.val_img_label = QLabel("Val 이미지:")
        pg.addWidget(self.val_img_label, 2, 0)
        self.val_img_edit = QLineEdit(); self.val_img_edit.setPlaceholderText("이미지 폴더")
        pg.addWidget(self.val_img_edit, 2, 1)
        self.btn_vi = QPushButton("찾기"); self.btn_vi.setObjectName("secondary")
        self.btn_vi.clicked.connect(lambda: self._browse(self.val_img_edit)); pg.addWidget(self.btn_vi, 2, 2)

        self.val_lbl_label = QLabel("Val 라벨(txt):")
        pg.addWidget(self.val_lbl_label, 3, 0)
        self.val_lbl_edit = QLineEdit(); self.val_lbl_edit.setPlaceholderText("txt 라벨 폴더")
        pg.addWidget(self.val_lbl_edit, 3, 1)
        self.btn_vl = QPushButton("찾기"); self.btn_vl.setObjectName("secondary")
        self.btn_vl.clicked.connect(lambda: self._browse(self.val_lbl_edit)); pg.addWidget(self.btn_vl, 3, 2)

        # Hold-out 비율 (holdout 전용)
        self.holdout_label = QLabel("Val 비율:")
        pg.addWidget(self.holdout_label, 4, 0)
        self.holdout_spin = QDoubleSpinBox()
        self.holdout_spin.setRange(0.05, 0.5); self.holdout_spin.setSingleStep(0.05)
        self.holdout_spin.setValue(0.2); self.holdout_spin.setSuffix("   (20% = 0.2)")
        pg.addWidget(self.holdout_spin, 4, 1)

        # K-Fold K값 (kfold 전용)
        self.kfold_label = QLabel("Fold 수 (K):")
        pg.addWidget(self.kfold_label, 5, 0)
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 20); self.kfold_spin.setValue(5)
        pg.addWidget(self.kfold_spin, 5, 1)

        pg.setColumnStretch(1, 1)
        main.addWidget(path_group)

        # 클래스 설정
        cls_group = QGroupBox("클래스 설정  (data.yaml names)")
        cv = QVBoxLayout(cls_group)
        ch = QHBoxLayout()
        ch.addWidget(QLabel("클래스 수:"))
        self.num_classes = QSpinBox(); self.num_classes.setRange(1, 1000); self.num_classes.setValue(1)
        self.num_classes.valueChanged.connect(self._update_class_table)
        ch.addWidget(self.num_classes); ch.addStretch()
        cv.addLayout(ch)
        hint = QLabel("클래스 ID(0부터) 순서대로 이름을 입력하세요")
        hint.setObjectName("hint"); cv.addWidget(hint)
        self.class_table = QTableWidget(1, 2)
        self.class_table.setHorizontalHeaderLabels(["클래스 ID", "클래스 이름"])
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.class_table.setMaximumHeight(220)
        self._init_class_table(1)
        cv.addWidget(self.class_table)
        main.addWidget(cls_group)

        # 버튼
        btn_h = QHBoxLayout()
        scan_btn = QPushButton("📂  파일 스캔"); scan_btn.setObjectName("secondary")
        scan_btn.clicked.connect(self._scan_files)
        verify_btn = QPushButton("✓  데이터 검증")
        verify_btn.clicked.connect(self._verify)
        save_btn = QPushButton("✔  설정 저장")
        save_btn.clicked.connect(self._save_state)
        btn_h.addWidget(scan_btn); btn_h.addWidget(verify_btn); btn_h.addWidget(save_btn)
        main.addLayout(btn_h)
        main.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0, 0, 0, 0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 오른쪽: 목록 + 로그 ──────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w); rv.setContentsMargins(16, 16, 16, 16); rv.setSpacing(8)

        two_col = QHBoxLayout()
        left_col = QVBoxLayout()
        ih = QLabel("🖼  이미지"); ih.setStyleSheet("font-weight:700; color:#2563eb;")
        left_col.addWidget(ih)
        self.file_list = QListWidget(); self.file_list.setMinimumHeight(180)
        left_col.addWidget(self.file_list)
        self.img_count_label = QLabel("0개"); self.img_count_label.setObjectName("hint")
        left_col.addWidget(self.img_count_label)
        two_col.addLayout(left_col)

        right_col = QVBoxLayout()
        lh = QLabel("🏷  라벨(txt)"); lh.setStyleSheet("font-weight:700; color:#059669;")
        right_col.addWidget(lh)
        self.label_list = QListWidget(); self.label_list.setMinimumHeight(180)
        right_col.addWidget(self.label_list)
        self.lbl_count_label = QLabel("0개"); self.lbl_count_label.setObjectName("hint")
        right_col.addWidget(self.lbl_count_label)
        two_col.addLayout(right_col)
        rv.addLayout(two_col)

        log_lbl = QLabel("검증 로그"); log_lbl.setObjectName("section_title")
        rv.addWidget(log_lbl)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        rv.addWidget(self.log_box, 1)

        splitter.addWidget(right_w)
        splitter.setSizes([460, 540])
        outer.addWidget(splitter)

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

    def _browse(self, edit):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if path: edit.setText(path)

    def _init_class_table(self, n):
        self.class_table.setRowCount(n)
        for i in range(n):
            self.class_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.class_table.setItem(i, 1, QTableWidgetItem(f"class{i}"))

    def _update_class_table(self, n):
        old = self.class_table.rowCount()
        self.class_table.setRowCount(n)
        for i in range(old, n):
            self.class_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.class_table.setItem(i, 1, QTableWidgetItem(f"class{i}"))

    def get_class_names(self):
        names = []
        for i in range(self.class_table.rowCount()):
            item = self.class_table.item(i, 1)
            names.append(item.text() if item and item.text() else f"class{i}")
        return names

    def _scan_files(self):
        self.file_list.clear(); self.label_list.clear()
        img_path = self.train_img_edit.text().strip()
        lbl_path = self.train_lbl_edit.text().strip()
        ext = self.img_ext.currentText().lower()

        if img_path and os.path.isdir(img_path):
            imgs = sorted([f for f in os.listdir(img_path)
                           if f.lower().endswith(f".{ext}")])
            for f in imgs:
                self.file_list.addItem(QListWidgetItem(f))
            self.img_count_label.setText(f"{len(imgs)}개")
        else:
            self.img_count_label.setText("경로 없음")

        if lbl_path and os.path.isdir(lbl_path):
            lbls = sorted([f for f in os.listdir(lbl_path)
                           if f.lower().endswith(".txt")])
            for f in lbls:
                self.label_list.addItem(QListWidgetItem(f))
            self.lbl_count_label.setText(f"{len(lbls)}개")
        else:
            self.lbl_count_label.setText("경로 없음")

    def _verify(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from yolo_module_wrap import scan_dataset

        self.log_box.clear()
        img = self.train_img_edit.text().strip()
        lbl = self.train_lbl_edit.text().strip()
        ext = self.img_ext.currentText().lower()
        self.log_box.append("── Train 데이터 검증 ──")
        res = scan_dataset(img, lbl, ext)
        for m in res["messages"]:
            self.log_box.append(m)

        # 클래스 수 자동 제안
        if res["max_class_id"] is not None:
            suggested = res["max_class_id"] + 1
            if suggested != self.num_classes.value():
                self.log_box.append(
                    f"\n[제안] 라벨의 최대 클래스 ID가 {res['max_class_id']} → "
                    f"클래스 수를 {suggested}로 설정하는 것을 권장합니다.")

        # manual이면 Val도 검증
        if self.split_btns.checkedId() == 0:
            vi = self.val_img_edit.text().strip()
            vl = self.val_lbl_edit.text().strip()
            if vi and vl:
                self.log_box.append("\n── Val 데이터 검증 ──")
                vres = scan_dataset(vi, vl, ext)
                for m in vres["messages"]:
                    self.log_box.append(m)

    def _save_state(self):
        mode = {0: "manual", 1: "holdout", 2: "kfold"}[self.split_btns.checkedId()]
        self.state["split_mode"]    = mode
        self.state["train_img_dir"] = self.train_img_edit.text().strip()
        self.state["train_lbl_dir"] = self.train_lbl_edit.text().strip()
        self.state["val_img_dir"]   = self.val_img_edit.text().strip()
        self.state["val_lbl_dir"]   = self.val_lbl_edit.text().strip()
        self.state["holdout_ratio"] = self.holdout_spin.value()
        self.state["kfold_k"]       = self.kfold_spin.value()
        self.state["img_ext"]       = self.img_ext.currentText()
        self.state["num_classes"]   = self.num_classes.value()
        self.state["class_names"]   = self.get_class_names()
        self.log_box.append(f"\n✔ 설정 저장 완료 (mode={mode}, "
                            f"classes={self.state['class_names']})")

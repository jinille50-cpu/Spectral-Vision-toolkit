"""
테스트 평가 탭
- 학습된 모델 + 별도 테스트셋(데이터 + 정답 라벨)으로 성능 평가
- 데이터 탭과 동일한 전처리 (전치 + 크롭 + 스무딩)
- 라벨 입력: 문자열 직접 / 숫자 파일 매핑 (정답)
- 출력: Confusion Matrix + ROC Curve + Classification Report
- CM/ROC 각각 저장
"""
import os
import re
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QSpinBox, QSplitter, QCheckBox, QMessageBox, QRadioButton,
    QButtonGroup
)
from PyQt5.QtCore import Qt, pyqtSignal


class TestEvalTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.raw_matrix = None
        self.data_matrix = None
        self.data_path = None
        self.num_labels = None
        # 마지막 평가 결과 (저장용)
        self.last_cm = None
        self.last_classes = None
        self.last_y_true = None
        self.last_y_probs = None
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.has_mpl = True
            self._Canvas = FigureCanvasQTAgg; self._Figure = Figure
        except ImportError:
            self.has_mpl = False
        self.canvas = None
        self.roc_canvas = None
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # 모델 정보
        self.model_info = QLabel("학습된 모델 없음 — 먼저 학습 탭에서 모델을 만들거나 불러오세요")
        self.model_info.setStyleSheet("padding: 8px; background: #f3f4f6; border-radius: 4px;")
        main.addWidget(self.model_info)

        # 좌측(설정) + 우측(결과)
        split = QSplitter(Qt.Horizontal)

        # ===== 좌측: 데이터/라벨 설정 =====
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0)

        # 1. 테스트 데이터 불러오기
        load_g = QGroupBox("1. 테스트 데이터 불러오기 (행렬)")
        lg = QGridLayout(load_g)
        lg.addWidget(QLabel("구분자:"), 0, 0)
        self.delim_combo = QComboBox()
        self.delim_combo.addItems(["자동", "Tab", "쉼표(,)", "공백", "세미콜론(;)"])
        lg.addWidget(self.delim_combo, 0, 1)
        lg.addWidget(QLabel("헤더 행:"), 0, 2)
        self.skip_rows = QSpinBox(); self.skip_rows.setRange(0, 100); self.skip_rows.setValue(0)
        lg.addWidget(self.skip_rows, 0, 3)

        btn_load = QPushButton("테스트 데이터 파일 선택")
        btn_load.clicked.connect(self._load_data)
        lg.addWidget(btn_load, 1, 0, 1, 4)

        self.shape_label = QLabel("불러온 데이터 없음")
        self.shape_label.setStyleSheet("padding: 4px; background: #f3f4f6; border-radius: 4px;")
        lg.addWidget(self.shape_label, 2, 0, 1, 3)
        self.transpose_check = QCheckBox("전치")
        self.transpose_check.toggled.connect(self._on_transpose)
        lg.addWidget(self.transpose_check, 2, 3)
        lg.setColumnStretch(1, 1)
        ll.addWidget(load_g)

        # 2. 전처리 (학습 시와 동일)
        prep_g = QGroupBox("2. 전처리 (학습 시와 동일하게)")
        cg = QGridLayout(prep_g)
        self.use_crop = QCheckBox("크롭 사용"); self.use_crop.setChecked(True)
        self.use_crop.toggled.connect(self._apply_preprocessing)
        cg.addWidget(self.use_crop, 0, 0, 1, 4)
        cg.addWidget(QLabel("시작:"), 1, 0)
        self.crop_start = QSpinBox(); self.crop_start.setRange(0, 1000000); self.crop_start.setValue(0)
        self.crop_start.valueChanged.connect(self._apply_preprocessing)
        cg.addWidget(self.crop_start, 1, 1)
        cg.addWidget(QLabel("끝:"), 1, 2)
        self.crop_end = QSpinBox(); self.crop_end.setRange(0, 1000000); self.crop_end.setValue(0)
        self.crop_end.valueChanged.connect(self._apply_preprocessing)
        cg.addWidget(self.crop_end, 1, 3)
        self.use_smooth = QCheckBox("Moving Average 스무딩"); self.use_smooth.setChecked(True)
        self.use_smooth.toggled.connect(self._apply_preprocessing)
        cg.addWidget(self.use_smooth, 2, 0, 1, 2)
        cg.addWidget(QLabel("윈도우:"), 2, 2)
        self.smooth_window = QSpinBox(); self.smooth_window.setRange(2, 101); self.smooth_window.setValue(5)
        self.smooth_window.setSingleStep(2)
        self.smooth_window.valueChanged.connect(self._apply_preprocessing)
        cg.addWidget(self.smooth_window, 2, 3)
        cg.setColumnStretch(1, 1); cg.setColumnStretch(3, 1)
        ll.addWidget(prep_g)

        # 3. 정답 라벨 입력
        label_g = QGroupBox("3. 정답 라벨 입력 (성능 평가용)")
        lblg = QGridLayout(label_g)
        self.label_mode_group = QButtonGroup(self)
        self.rb_string = QRadioButton("문자열 직접 입력"); self.rb_string.setChecked(True)
        self.rb_numfile = QRadioButton("숫자 라벨 파일 + 매핑")
        self.label_mode_group.addButton(self.rb_string, 0)
        self.label_mode_group.addButton(self.rb_numfile, 1)
        self.rb_string.toggled.connect(self._on_label_mode)
        lblg.addWidget(self.rb_string, 0, 0)
        lblg.addWidget(self.rb_numfile, 0, 1)

        # 문자열 박스
        self.string_box = QWidget()
        sb_l = QGridLayout(self.string_box); sb_l.setContentsMargins(0,0,0,0)
        sb_l.addWidget(QLabel("적용:"), 0, 0)
        self.string_apply_mode = QComboBox()
        self.string_apply_mode.addItems(["모든 샘플 동일 라벨", "샘플별 개별 입력 (표 편집)"])
        self.string_apply_mode.currentIndexChanged.connect(self._refresh_preview)
        sb_l.addWidget(self.string_apply_mode, 0, 1)
        sb_l.addWidget(QLabel("라벨:"), 1, 0)
        self.single_label = QLineEdit("PP")
        sb_l.addWidget(self.single_label, 1, 1)
        sb_l.setColumnStretch(1, 1)
        lblg.addWidget(self.string_box, 1, 0, 1, 2)

        # 숫자 파일 박스
        self.numfile_box = QWidget()
        nb_l = QGridLayout(self.numfile_box); nb_l.setContentsMargins(0,0,0,0)
        btn_lbl = QPushButton("라벨 파일 선택 (숫자)"); btn_lbl.setObjectName("secondary")
        btn_lbl.clicked.connect(self._load_label_file)
        nb_l.addWidget(btn_lbl, 0, 0, 1, 2)
        self.label_file_info = QLabel("라벨 파일 없음"); self.label_file_info.setObjectName("hint")
        nb_l.addWidget(self.label_file_info, 1, 0, 1, 2)
        nb_l.addWidget(QLabel("숫자 -> 클래스명:"), 2, 0, 1, 2)
        self.num_map_table = QTableWidget(0, 2)
        self.num_map_table.setHorizontalHeaderLabels(["숫자", "클래스명"])
        self.num_map_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.num_map_table.setMaximumHeight(120)
        nb_l.addWidget(self.num_map_table, 3, 0, 1, 2)
        self.numfile_box.hide()
        lblg.addWidget(self.numfile_box, 2, 0, 1, 2)
        lblg.setColumnStretch(1, 1)
        ll.addWidget(label_g)

        # 미리보기 (샘플 + 라벨)
        self.preview_table = QTableWidget(0, 3)
        self.preview_table.setHorizontalHeaderLabels(["샘플 #", "정답 라벨", "길이"])
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_table.setMaximumHeight(150)
        ll.addWidget(self.preview_table)

        # 평가 실행
        self.btn_eval = QPushButton("▶  테스트 평가 실행")
        self.btn_eval.clicked.connect(self._run_eval)
        ll.addWidget(self.btn_eval)
        ll.addStretch()
        split.addWidget(left)

        # ===== 우측: 결과 =====
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0)
        self.result_label = QLabel("테스트 평가 결과가 여기 표시됩니다")
        self.result_label.setStyleSheet("font-size: 14px; padding: 8px;")
        rl.addWidget(self.result_label)

        # 옵션 + 저장
        opt_bar = QHBoxLayout()
        self.cm_normalize = QCheckBox("비율 표시 (%)"); self.cm_normalize.setChecked(True)
        self.cm_normalize.toggled.connect(self._refresh_cm)
        opt_bar.addWidget(self.cm_normalize)
        self.cm_show_count = QCheckBox("개수 표시"); self.cm_show_count.setChecked(True)
        self.cm_show_count.toggled.connect(self._refresh_cm)
        opt_bar.addWidget(self.cm_show_count)
        opt_bar.addStretch()
        self.btn_save_cm = QPushButton("💾  CM 저장"); self.btn_save_cm.setObjectName("secondary")
        self.btn_save_cm.clicked.connect(self._save_cm)
        opt_bar.addWidget(self.btn_save_cm)
        self.btn_save_roc = QPushButton("💾  ROC 저장"); self.btn_save_roc.setObjectName("secondary")
        self.btn_save_roc.clicked.connect(self._save_roc)
        opt_bar.addWidget(self.btn_save_roc)
        self.btn_save_all = QPushButton("📦  전체 결과 저장"); self.btn_save_all.setObjectName("secondary")
        self.btn_save_all.clicked.connect(self._save_all)
        opt_bar.addWidget(self.btn_save_all)
        rl.addLayout(opt_bar)

        # CM + ROC 가로 배치
        figs_h = QHBoxLayout()
        cm_g = QGroupBox("Confusion Matrix")
        cm_l = QVBoxLayout(cm_g)
        self.canvas_container = QVBoxLayout()
        cv_w = QWidget(); cv_w.setLayout(self.canvas_container)
        cm_l.addWidget(cv_w)
        figs_h.addWidget(cm_g, 1)
        roc_g = QGroupBox("ROC Curve")
        roc_l = QVBoxLayout(roc_g)
        self.roc_container = QVBoxLayout()
        roc_w = QWidget(); roc_w.setLayout(self.roc_container)
        roc_l.addWidget(roc_w)
        figs_h.addWidget(roc_g, 1)
        rl.addLayout(figs_h, 1)

        # 리포트
        self.report_box = QTextEdit()
        self.report_box.setReadOnly(True); self.report_box.setMaximumHeight(180)
        self.report_box.setStyleSheet("font-family: 'Courier New', monospace; font-size: 11px;")
        rl.addWidget(self.report_box)
        split.addWidget(right)
        split.setSizes([450, 850])
        main.addWidget(split, 1)

        # 로그
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(90)
        main.addWidget(self.log_box)

    # ════════════ 모델 정보 ════════════
    def refresh_model_info(self):
        meta = self.state.get("model_meta")
        if meta is None:
            self.model_info.setText("학습된 모델 없음 — 먼저 학습 탭에서 모델을 만들거나 불러오세요")
        else:
            self.model_info.setText(
                f"<b>{meta['model_name']}</b>  ·  norm={meta['norm_type']}  ·  "
                f"클래스 {len(meta['classes'])}개  ·  학습 CV Acc {meta['cv_acc']:.4f}"
            )

    # ════════════ 데이터 로드 ════════════
    def _load_data(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "테스트 데이터 선택", "",
            "Data files (*.txt *.csv *.dat *.asc);;All (*.*)")
        if not fp: return
        try:
            arr = self._parse_matrix(fp)
            if arr is None:
                QMessageBox.warning(self, "로드 실패", "파싱 불가")
                return
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            self.raw_matrix = arr
            self.data_path = fp
            self.transpose_check.setChecked(False)
            self._log(f"불러옴: {os.path.basename(fp)}  shape={arr.shape}")
            self._reset_crop_range()
            self._apply_preprocessing()
        except Exception as e:
            QMessageBox.critical(self, "로드 실패", str(e))

    def _parse_matrix(self, fp):
        delim_map = {"자동": None, "Tab": "\t", "쉼표(,)": ",",
                     "공백": None, "세미콜론(;)": ";"}
        delim = delim_map[self.delim_combo.currentText()]
        skip = self.skip_rows.value()
        try:
            return np.loadtxt(fp, delimiter=delim, skiprows=skip)
        except Exception:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[skip:]
            rows = []
            for ln in lines:
                ln = ln.strip()
                if not ln or ln.startswith('#') or ln.startswith('%'):
                    continue
                vals = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', ln)
                if vals:
                    rows.append([float(v) for v in vals])
            if not rows: return None
            ml = max(len(r) for r in rows)
            rows = [r for r in rows if len(r) == ml]
            return np.array(rows)

    def _reset_crop_range(self):
        if self.raw_matrix is None: return
        arr = self.raw_matrix
        n_ch = arr.shape[0] if self.transpose_check.isChecked() else arr.shape[1]
        self.crop_start.blockSignals(True); self.crop_end.blockSignals(True)
        self.crop_start.setRange(0, 1000000); self.crop_end.setRange(0, 1000000)
        self.crop_start.setValue(0); self.crop_end.setValue(n_ch)
        self.crop_start.setRange(0, max(1, n_ch - 1)); self.crop_end.setRange(0, n_ch)
        self.crop_start.blockSignals(False); self.crop_end.blockSignals(False)

    def _on_transpose(self):
        self._reset_crop_range()
        self._apply_preprocessing()

    @staticmethod
    def _moving_average(spec, window):
        if window < 2: return spec
        pad = window // 2
        padded = np.pad(spec, pad, mode='edge')
        kernel = np.ones(window) / window
        return np.convolve(padded, kernel, mode='valid')

    def _apply_preprocessing(self):
        if self.raw_matrix is None: return
        arr = self.raw_matrix.copy()
        if self.transpose_check.isChecked():
            arr = arr.T
        if self.use_crop.isChecked():
            cs, ce = self.crop_start.value(), self.crop_end.value()
            if ce <= 0 or ce > arr.shape[1]:
                ce = arr.shape[1]
            if cs < ce:
                arr = arr[:, cs:ce]
        if self.use_smooth.isChecked():
            sw = self.smooth_window.value()
            arr = np.stack([self._moving_average(arr[i], sw) for i in range(arr.shape[0])])
        self.data_matrix = arr
        n, m = arr.shape
        self.shape_label.setText(f"<b>{n} 샘플 x {m} 채널</b>")
        self._refresh_preview()

    # ════════════ 라벨 ════════════
    def _on_label_mode(self):
        is_str = self.rb_string.isChecked()
        self.string_box.setVisible(is_str)
        self.numfile_box.setVisible(not is_str)
        self._refresh_preview()

    def _load_label_file(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "라벨 파일 선택 (숫자)", "", "Label (*.txt *.csv *.dat);;All (*.*)")
        if not fp: return
        try:
            labels = np.loadtxt(fp).flatten().astype(int)
            self.num_labels = labels
            uniq = sorted(set(labels))
            self.label_file_info.setText(f"라벨 {len(labels)}개  ·  고유값: {uniq}")
            self.num_map_table.setRowCount(len(uniq))
            for i, u in enumerate(uniq):
                self.num_map_table.setItem(i, 0, QTableWidgetItem(str(u)))
                self.num_map_table.setItem(i, 1, QTableWidgetItem(f"Class{u}"))
            self._refresh_preview()
        except Exception as e:
            QMessageBox.critical(self, "라벨 로드 실패", str(e))

    def _get_labels(self, n):
        if self.rb_string.isChecked():
            if self.string_apply_mode.currentIndex() == 0:
                lbl = self.single_label.text().strip() or "Unknown"
                return [lbl] * n
            else:
                out = []
                for i in range(n):
                    it = self.preview_table.item(i, 1)
                    out.append(it.text().strip() if it else "Unknown")
                return out
        else:
            if self.num_labels is None:
                return ["Unknown"] * n
            mapping = {}
            for r in range(self.num_map_table.rowCount()):
                k = self.num_map_table.item(r, 0); v = self.num_map_table.item(r, 1)
                if k and v:
                    try: mapping[int(k.text())] = v.text().strip()
                    except ValueError: pass
            out = []
            for i in range(n):
                if i < len(self.num_labels):
                    num = int(self.num_labels[i])
                    out.append(mapping.get(num, f"Class{num}"))
                else:
                    out.append("Unknown")
            return out

    def _refresh_preview(self):
        if self.data_matrix is None:
            self.preview_table.setRowCount(0); return
        n, m = self.data_matrix.shape
        labels = self._get_labels(n)
        editable = (self.rb_string.isChecked() and self.string_apply_mode.currentIndex() == 1)
        self.preview_table.setRowCount(n)
        for i in range(n):
            self.preview_table.setItem(i, 0, QTableWidgetItem(str(i)))
            it = QTableWidgetItem(labels[i])
            if not editable:
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.preview_table.setItem(i, 1, it)
            self.preview_table.setItem(i, 2, QTableWidgetItem(str(m)))

    # ════════════ 평가 실행 ════════════
    def _run_eval(self):
        if self.state.get("trained_model") is None:
            QMessageBox.warning(self, "모델 없음", "먼저 학습 탭에서 모델을 학습하거나 불러오세요.")
            return
        if self.data_matrix is None:
            QMessageBox.warning(self, "데이터 없음", "테스트 데이터를 먼저 불러오세요.")
            return

        model = self.state["trained_model"]
        meta = self.state["model_meta"]
        scaler = meta["scaler"]
        classes = meta["classes"]
        le = meta.get("label_enc")

        n, m = self.data_matrix.shape
        # 채널 수 검사
        if self.state.get("database"):
            exp = len(self.state["database"][0]["spectrum"])
            if m != exp:
                QMessageBox.warning(self, "채널 불일치",
                    f"테스트 데이터 채널({m}) ≠ 학습 채널({exp})\n전처리(크롭/전치)를 학습과 동일하게 맞추세요.")
                return

        # 정답 라벨
        true_labels = self._get_labels(n)
        # 모델 클래스에 없는 라벨 체크
        unknown = set(true_labels) - set(classes)
        if unknown:
            box = QMessageBox.question(self, "미지의 라벨",
                f"학습 클래스에 없는 라벨이 있습니다: {unknown}\n"
                f"(이 샘플들은 평가에서 제외됩니다) 계속할까요?",
                QMessageBox.Yes | QMessageBox.No)
            if box == QMessageBox.No: return

        try:
            from sklearn.metrics import (confusion_matrix, classification_report,
                                         accuracy_score, f1_score)

            # 라벨 → 정수 인코딩 (모델 클래스 순서 기준)
            cls_to_idx = {c: i for i, c in enumerate(classes)}
            X_list, y_list = [], []
            for i in range(n):
                if true_labels[i] in cls_to_idx:
                    X_list.append(self.data_matrix[i])
                    y_list.append(cls_to_idx[true_labels[i]])
            if not X_list:
                QMessageBox.warning(self, "평가 불가", "유효한 라벨이 없습니다.")
                return
            X = np.stack(X_list).astype(np.float32)
            y_true = np.array(y_list)

            # 정규화 (학습 때 scaler 그대로)
            if scaler is not None:
                X = scaler.transform(X)

            # 예측
            y_pred = model.predict(X)
            y_probs = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)
                    y_probs = np.zeros((len(y_true), len(classes)))
                    for col, cls in enumerate(model.classes_):
                        y_probs[:, cls] = proba[:, col]
                except Exception:
                    y_probs = None

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
            report = classification_report(y_true, y_pred,
                                           labels=list(range(len(classes))),
                                           target_names=classes,
                                           zero_division=0, digits=4)

            # 결과 저장
            self.last_cm = cm
            self.last_classes = classes
            self.last_y_true = y_true
            self.last_y_probs = y_probs

            self.result_label.setText(
                f"<b>테스트 평가</b>  ·  샘플 {len(y_true)}개  ·  "
                f"<b>Accuracy: {acc:.4f}</b>  ·  F1 (macro): {f1:.4f}"
            )
            self.report_box.setPlainText(report)
            self._draw_cm(cm, classes)
            self._draw_roc(y_true, y_probs, classes)

            self._log(f"✔ 테스트 평가 완료: {len(y_true)}개 샘플, Acc {acc:.4f}, F1 {f1:.4f}")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "평가 실패", f"{e}\n{traceback.format_exc()}")

    # ════════════ Figure ════════════
    def _refresh_cm(self):
        if self.last_cm is not None:
            self._draw_cm(self.last_cm, self.last_classes)

    def _draw_cm(self, cm, classes):
        if not self.has_mpl: return
        if self.canvas:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None); self.canvas = None
        fig = self._build_cm_figure(cm, classes,
                                    self.cm_normalize.isChecked(),
                                    self.cm_show_count.isChecked())
        self.canvas = self._Canvas(fig)
        self.canvas_container.addWidget(self.canvas)
        self.canvas.draw()

    def _build_cm_figure(self, cm, classes, show_pct=True, show_count=True):
        fig = self._Figure(figsize=(5.5, 5), facecolor='white')
        ax = fig.add_subplot(111)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        im = ax.imshow(cm_norm, cmap='Purples', vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (Test Set)")
        for i in range(len(classes)):
            for j in range(len(classes)):
                parts = []
                if show_count: parts.append(f"{cm[i,j]}")
                if show_pct:   parts.append(f"({cm_norm[i,j]*100:.1f}%)")
                txt = "\n".join(parts) if parts else ""
                color = 'white' if cm_norm[i,j] > 0.5 else 'black'
                ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    def _draw_roc(self, y_true, y_probs, classes):
        if not self.has_mpl: return
        if self.roc_canvas:
            self.roc_container.removeWidget(self.roc_canvas)
            self.roc_canvas.setParent(None); self.roc_canvas = None
        fig = self._build_roc_figure(y_true, y_probs, classes)
        if fig is None: return
        self.roc_canvas = self._Canvas(fig)
        self.roc_container.addWidget(self.roc_canvas)
        self.roc_canvas.draw()

    def _build_roc_figure(self, y_true, y_probs, classes):
        if y_probs is None or y_true is None:
            return None
        try:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
        except Exception:
            return None
        n_classes = len(classes)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2:
            y_bin = np.hstack([1 - y_bin, y_bin])

        fig = self._Figure(figsize=(5.5, 5), facecolor='white')
        ax = fig.add_subplot(111)
        import matplotlib.cm as mcm
        try:
            cmap = mcm.get_cmap('tab10' if n_classes <= 10 else 'tab20')
        except Exception:
            import matplotlib
            cmap = matplotlib.colormaps['tab10' if n_classes <= 10 else 'tab20']

        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=cmap(i % 20), lw=1.8,
                        label=f"{classes[i]} (AUC={roc_auc:.3f})")
            except Exception:
                continue
        try:
            all_fpr = np.unique(np.concatenate([
                roc_curve(y_bin[:, i], y_probs[:, i])[0] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= n_classes
            ax.plot(all_fpr, mean_tpr, 'k--', lw=2,
                    label=f"Macro-avg (AUC={auc(all_fpr, mean_tpr):.3f})")
        except Exception:
            pass
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Test Set)")
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    # ════════════ 저장 ════════════
    def _save_cm(self):
        if self.last_cm is None:
            QMessageBox.warning(self, "저장 불가", "먼저 평가를 실행하세요."); return
        fp, _ = QFileDialog.getSaveFileName(self, "CM 저장", "test_confusion_matrix.png",
                                            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not fp: return
        try:
            fig = self._build_cm_figure(self.last_cm, self.last_classes,
                                        self.cm_normalize.isChecked(), self.cm_show_count.isChecked())
            fig.savefig(fp, dpi=300, bbox_inches='tight')
            import matplotlib.pyplot as plt; plt.close(fig)
            self._log(f"✔ CM 저장: {fp}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    def _save_roc(self):
        if self.last_y_probs is None:
            QMessageBox.warning(self, "저장 불가", "ROC 데이터 없음 (확률 미지원 모델이거나 미평가)"); return
        fp, _ = QFileDialog.getSaveFileName(self, "ROC 저장", "test_roc_curve.png",
                                            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not fp: return
        try:
            fig = self._build_roc_figure(self.last_y_true, self.last_y_probs, self.last_classes)
            if fig is None:
                QMessageBox.warning(self, "저장 불가", "ROC 생성 불가"); return
            fig.savefig(fp, dpi=300, bbox_inches='tight')
            import matplotlib.pyplot as plt; plt.close(fig)
            self._log(f"✔ ROC 저장: {fp}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    def _save_all(self):
        if self.last_cm is None:
            QMessageBox.warning(self, "저장 불가", "먼저 평가를 실행하세요."); return
        d = QFileDialog.getExistingDirectory(self, "결과 저장 폴더 선택")
        if not d: return
        try:
            import matplotlib.pyplot as plt
            meta = self.state.get("model_meta", {})
            stem = f"TEST_{meta.get('model_name','model')}_{meta.get('norm_type','none')}"

            fig_cm = self._build_cm_figure(self.last_cm, self.last_classes,
                                           self.cm_normalize.isChecked(), self.cm_show_count.isChecked())
            fig_cm.savefig(os.path.join(d, f"{stem}_confusion_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_cm)

            roc_saved = False
            if self.last_y_probs is not None:
                fig_roc = self._build_roc_figure(self.last_y_true, self.last_y_probs, self.last_classes)
                if fig_roc is not None:
                    fig_roc.savefig(os.path.join(d, f"{stem}_roc_curve.png"), dpi=300, bbox_inches='tight')
                    plt.close(fig_roc); roc_saved = True

            np.save(os.path.join(d, f"{stem}_cm.npy"), self.last_cm)
            with open(os.path.join(d, f"{stem}_cm.csv"), "w", encoding="utf-8") as f:
                f.write("," + ",".join(self.last_classes) + "\n")
                for i, cls in enumerate(self.last_classes):
                    f.write(cls + "," + ",".join(str(v) for v in self.last_cm[i]) + "\n")
            with open(os.path.join(d, f"{stem}_report.txt"), "w", encoding="utf-8") as f:
                f.write(self.report_box.toPlainText())

            self._log(f"✔ 전체 저장: {d}")
            n_files = 4 + (1 if roc_saved else 0)
            QMessageBox.information(self, "저장 완료", f"{n_files}개 파일 저장됨\n{d}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    def _log(self, msg):
        self.log_box.append(msg)

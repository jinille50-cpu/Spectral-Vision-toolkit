"""
예측 탭
- 학습된 모델로 새 스펙트럼 파일을 불러와서 예측
- 예측 결과 + 확률(가능하면) 표시
- 결과 내보내기 (CSV)
"""
import os
import re
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QSpinBox, QSplitter, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal


class PredictTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.results = []   # [{filename, predicted, probs(optional), spectrum}]
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.has_mpl = True
            self._Canvas = FigureCanvasQTAgg; self._Figure = Figure
        except ImportError:
            self.has_mpl = False
        self.canvas = None
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # 모델 정보
        self.model_info = QLabel("학습된 모델 없음")
        self.model_info.setStyleSheet("padding: 8px; background: #f3f4f6; border-radius: 4px;")
        main.addWidget(self.model_info)

        # 파일 로드
        load_g = QGroupBox("새 스펙트럼 파일 예측 (행렬)")
        lg = QGridLayout(load_g)
        from PyQt5.QtWidgets import QCheckBox

        lg.addWidget(QLabel("구분자:"), 0, 0)
        self.delim_combo = QComboBox()
        self.delim_combo.addItems(["자동", "Tab", "쉼표(,)", "공백", "세미콜론(;)"])
        lg.addWidget(self.delim_combo, 0, 1)
        lg.addWidget(QLabel("건너뛸 헤더 행:"), 0, 2)
        self.skip_rows = QSpinBox(); self.skip_rows.setRange(0, 100); self.skip_rows.setValue(0)
        lg.addWidget(self.skip_rows, 0, 3)

        btn_files = QPushButton("📂  파일 선택 (다중)")
        btn_files.clicked.connect(self._predict_files)
        lg.addWidget(btn_files, 1, 0, 1, 2)
        btn_folder = QPushButton("📁  폴더 선택")
        btn_folder.setObjectName("secondary")
        btn_folder.clicked.connect(self._predict_folder)
        lg.addWidget(btn_folder, 1, 2, 1, 2)

        self.transpose_check = QCheckBox("전치 (Transpose) — 데이터가 (채널 x 샘플)이면 체크")
        lg.addWidget(self.transpose_check, 2, 0, 1, 4)
        lg.setColumnStretch(1, 1)
        main.addWidget(load_g)

        # ── 전처리 옵션 (학습 시와 동일하게 적용) ──
        crop_g = QGroupBox("전처리 옵션 (학습 시와 동일하게 설정)")
        cg = QGridLayout(crop_g)
        # 크롭
        self.use_crop = QCheckBox("크롭 사용")
        self.use_crop.setChecked(True)   # 기본 ON
        cg.addWidget(self.use_crop, 0, 0, 1, 4)
        cg.addWidget(QLabel("시작:"), 1, 0)
        self.crop_start = QSpinBox(); self.crop_start.setRange(0, 1000000); self.crop_start.setValue(0)
        cg.addWidget(self.crop_start, 1, 1)
        cg.addWidget(QLabel("끝:"), 1, 2)
        self.crop_end = QSpinBox(); self.crop_end.setRange(0, 1000000); self.crop_end.setValue(0)
        cg.addWidget(self.crop_end, 1, 3)
        # 스무딩
        self.use_smooth = QCheckBox("Moving Average 스무딩")
        self.use_smooth.setChecked(True)   # 기본 ON
        cg.addWidget(self.use_smooth, 2, 0, 1, 2)
        cg.addWidget(QLabel("윈도우 크기:"), 2, 2)
        self.smooth_window = QSpinBox(); self.smooth_window.setRange(2, 101); self.smooth_window.setValue(5)
        self.smooth_window.setSingleStep(2)
        cg.addWidget(self.smooth_window, 2, 3)
        hint = QLabel("※ 데이터 탭에서 학습 때 쓴 것과 동일하게 설정해야 정확히 예측됩니다.")
        hint.setObjectName("hint")
        cg.addWidget(hint, 3, 0, 1, 4)
        cg.setColumnStretch(1, 1); cg.setColumnStretch(3, 1)
        main.addWidget(crop_g)

        # 결과 (좌우)
        split = QSplitter(Qt.Horizontal)

        # 좌측: 결과 테이블
        left_g = QGroupBox("예측 결과")
        ll = QVBoxLayout(left_g)
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(
            ["파일명", "예측 라벨", "확률", "Top-2"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.itemSelectionChanged.connect(self._on_select)
        ll.addWidget(self.result_table)

        bh = QHBoxLayout()
        btn_export = QPushButton("📊  CSV 내보내기")
        btn_export.setObjectName("secondary")
        btn_export.clicked.connect(self._export_results)
        bh.addWidget(btn_export)
        btn_clear = QPushButton("결과 비우기")
        btn_clear.setObjectName("secondary")
        btn_clear.clicked.connect(self._clear)
        bh.addWidget(btn_clear)
        ll.addLayout(bh)
        split.addWidget(left_g)

        # 우측: 스펙트럼 시각화
        right_g = QGroupBox("스펙트럼 미리보기")
        rl = QVBoxLayout(right_g)
        self.canvas_container = QVBoxLayout()
        cw = QWidget(); cw.setLayout(self.canvas_container)
        rl.addWidget(cw)
        split.addWidget(right_g)
        split.setSizes([700, 600])
        main.addWidget(split, 1)

        # 로그
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(80)
        main.addWidget(self.log_box)

    def refresh_model_info(self):
        meta = self.state.get("model_meta")
        if meta is None:
            self.model_info.setText("학습된 모델 없음")
        else:
            self.model_info.setText(
                f"<b>{meta['model_name']}</b>  ·  norm={meta['norm_type']}  ·  "
                f"클래스 {len(meta['classes'])}개  ·  "
                f"CV Acc {meta['cv_acc']:.4f} · F1 {meta['cv_f1']:.4f}"
            )

    # ────────────────────────────────────────────────────────
    def _predict_files(self):
        if self.state.get("trained_model") is None:
            QMessageBox.warning(self, "모델 없음", "먼저 모델을 학습하거나 불러오세요.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "예측할 스펙트럼 파일", "",
            "Spectrum (*.txt *.csv *.dat *.asc);;All (*.*)")
        if files: self._do_predict(files)

    def _predict_folder(self):
        if self.state.get("trained_model") is None:
            QMessageBox.warning(self, "모델 없음", "먼저 모델을 학습하거나 불러오세요.")
            return
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if not d: return
        exts = ('.txt', '.csv', '.dat', '.asc')
        files = []
        for root, _, fnames in os.walk(d):
            for fn in fnames:
                if fn.lower().endswith(exts):
                    files.append(os.path.join(root, fn))
        if files: self._do_predict(sorted(files))

    @staticmethod
    def _moving_average(spec, window):
        """엣지 패딩 적용 Moving Average"""
        if window < 2: return spec
        pad = window // 2
        padded = np.pad(spec, pad, mode='edge')
        kernel = np.ones(window) / window
        return np.convolve(padded, kernel, mode='valid')

    def _do_predict(self, files):
        model = self.state["trained_model"]
        meta  = self.state["model_meta"]
        scaler = meta["scaler"]
        classes = meta["classes"]

        # 전처리 설정
        use_transpose = self.transpose_check.isChecked()
        use_crop = self.use_crop.isChecked()
        crop_s = self.crop_start.value()
        crop_e = self.crop_end.value()
        use_smooth = self.use_smooth.isChecked()
        sw = self.smooth_window.value()

        # 기대되는 입력 길이 (학습 DB 기준)
        db = self.state.get("database", [])
        expected_len = len(db[0]["spectrum"]) if db else None

        for fp in files:
            try:
                arr = self._parse_matrix(fp)
                if arr is None:
                    self._log(f"⚠ 빈/파싱 실패: {os.path.basename(fp)}")
                    continue
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                # 전치
                if use_transpose:
                    arr = arr.T
                # 각 행 = 한 샘플
                for i in range(arr.shape[0]):
                    spec = arr[i]
                    # 크롭
                    if use_crop:
                        ce = crop_e if (0 < crop_e <= len(spec)) else len(spec)
                        cs = crop_s if crop_s < ce else 0
                        spec = spec[cs:ce]
                    # 스무딩
                    if use_smooth:
                        spec = self._moving_average(spec, sw)
                    if expected_len is None:
                        expected_len = len(spec)
                    if len(spec) != expected_len:
                        self._log(f"⚠ 길이 불일치 ({len(spec)} ≠ {expected_len}): {os.path.basename(fp)}#{i}")
                        continue
                    X = np.asarray(spec, dtype=np.float32).reshape(1, -1)
                    if scaler is not None:
                        X = scaler.transform(X)
                    pred_idx = model.predict(X)[0]
                    predicted = classes[pred_idx]

                    probs = None
                    if hasattr(model, "predict_proba"):
                        try:
                            probs = model.predict_proba(X)[0]
                        except Exception:
                            pass

                    self.results.append({
                        "filename": os.path.basename(fp) + (f" #{i}" if arr.shape[0] > 1 else ""),
                        "path":     fp,
                        "spectrum": spec,
                        "predicted":predicted,
                        "probs":    probs,
                    })
            except Exception as e:
                self._log(f"✗ {os.path.basename(fp)}: {e}")

        self._refresh_table()
        self._log(f"✔ {len(self.results)}개 예측 완료")

    def _parse_matrix(self, fp):
        """행렬로 불러오기 (data_tab과 동일)"""
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

    def _refresh_table(self):
        self.result_table.setRowCount(len(self.results))
        meta = self.state.get("model_meta")
        classes = meta["classes"] if meta else []

        for i, r in enumerate(self.results):
            self.result_table.setItem(i, 0, QTableWidgetItem(r["filename"]))
            self.result_table.setItem(i, 1, QTableWidgetItem(r["predicted"]))
            if r["probs"] is not None:
                max_p = float(r["probs"].max())
                self.result_table.setItem(i, 2, QTableWidgetItem(f"{max_p*100:.1f}%"))
                # Top-2
                top2_idx = np.argsort(r["probs"])[-2:][::-1]
                top2_str = " · ".join([f"{classes[j]}({r['probs'][j]*100:.0f}%)" for j in top2_idx])
                self.result_table.setItem(i, 3, QTableWidgetItem(top2_str))
            else:
                self.result_table.setItem(i, 2, QTableWidgetItem("-"))
                self.result_table.setItem(i, 3, QTableWidgetItem("-"))

    def _on_select(self):
        rows = sorted({i.row() for i in self.result_table.selectedIndexes()})
        if not rows or not self.has_mpl: return
        r = self.results[rows[0]]
        self._draw_spectrum(r)

    def _draw_spectrum(self, r):
        if self.canvas:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None); self.canvas = None
        fig = self._Figure(figsize=(8, 5), facecolor='white')
        ax = fig.add_subplot(111)

        wl = self.state.get("wavelengths")
        x = wl if (wl is not None and len(wl) == len(r["spectrum"])) else np.arange(len(r["spectrum"]))
        ax.plot(x, r["spectrum"], 'b-', lw=1.2)
        ax.set_xlabel("Wavelength" if wl is not None else "Channel")
        ax.set_ylabel("Intensity")

        title = f"{r['filename']}  →  Predicted: {r['predicted']}"
        if r["probs"] is not None:
            title += f" ({r['probs'].max()*100:.1f}%)"
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self.canvas = self._Canvas(fig)
        self.canvas_container.addWidget(self.canvas)
        self.canvas.draw()

    def _export_results(self):
        if not self.results:
            QMessageBox.warning(self, "내보낼 결과 없음", "예측을 먼저 수행하세요.")
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "결과 저장", "predictions.csv", "CSV (*.csv)")
        if not fp: return
        try:
            import csv
            meta = self.state["model_meta"]; classes = meta["classes"]
            with open(fp, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                header = ["filename", "predicted"]
                if any(r["probs"] is not None for r in self.results):
                    header += [f"prob_{c}" for c in classes]
                w.writerow(header)
                for r in self.results:
                    row = [r["filename"], r["predicted"]]
                    if r["probs"] is not None:
                        row += [f"{p:.4f}" for p in r["probs"]]
                    elif any(rr["probs"] is not None for rr in self.results):
                        row += [""] * len(classes)
                    w.writerow(row)
            self._log(f"✔ 저장: {fp}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    def _clear(self):
        self.results.clear()
        self.result_table.setRowCount(0)
        if self.canvas:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None); self.canvas = None

    def _log(self, msg):
        self.log_box.append(msg)

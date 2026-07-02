import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QLineEdit, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QTextEdit, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap


# ═══════════════════════════════════════════════════════════════
#  평가 Worker — YOLO.val()
# ═══════════════════════════════════════════════════════════════
class EvalWorker(QThread):
    result_ready = pyqtSignal(dict)     # 지표 dict
    log_msg      = pyqtSignal(str)
    finished     = pyqtSignal(str)      # save_dir

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            from ultralytics import YOLO
            import torch
        except ImportError as e:
            self.log_msg.emit(f"[오류] ultralytics 미설치: {e}")
            self.finished.emit(""); return

        cfg = self.config
        try:
            self.log_msg.emit(f"[평가 시작] 모델: {os.path.basename(cfg['model_path'])}")
            self.log_msg.emit(f"  data.yaml: {cfg['data_yaml']}")
            device = 0 if torch.cuda.is_available() else "cpu"
            model = YOLO(cfg["model_path"])
            metrics = model.val(
                data=cfg["data_yaml"],
                imgsz=cfg.get("imgsz", 640),
                batch=cfg.get("batch", 16),
                device=device,
                split=cfg.get("split", "val"),
                plots=True,
            )
            # 지표 추출
            box = metrics.box
            results = {
                "mAP50":     float(box.map50),
                "mAP50-95":  float(box.map),
                "Precision": float(box.mp),
                "Recall":    float(box.mr),
            }
            # per-class mAP50
            per_class = {}
            try:
                names = model.names
                for i, ap in enumerate(box.ap50):
                    cname = names.get(i, f"class{i}") if isinstance(names, dict) else str(i)
                    per_class[cname] = float(ap)
            except Exception:
                pass
            results["per_class"] = per_class

            save_dir = str(getattr(metrics, "save_dir", "") or "")
            self.log_msg.emit("  ✔ 평가 완료")
            self.result_ready.emit(results)
            self.finished.emit(save_dir)
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
            self.finished.emit("")


# ═══════════════════════════════════════════════════════════════
#  EvalTab UI
# ═══════════════════════════════════════════════════════════════
class EvalTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self.worker = None
        self._infer_tab = None
        self._build()

    def set_infer_tab(self, tab):
        self._infer_tab = tab

    def set_model_path(self, path):
        self.model_path.setText(path)

    def set_data_yaml(self, path):
        self.yaml_path.setText(path)

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽 ─────────────────────────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        lv = QVBoxLayout(inner); lv.setSpacing(10); lv.setContentsMargins(14, 14, 14, 14)

        mg = QGroupBox("모델 / 데이터")
        m = QGridLayout(mg)
        m.addWidget(QLabel("모델 (.pt):"), 0, 0)
        self.model_path = QLineEdit(); self.model_path.setPlaceholderText("학습 후 자동 반영 또는 직접 선택")
        m.addWidget(self.model_path, 0, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary")
        b.clicked.connect(self._browse_model); m.addWidget(b, 0, 2)

        m.addWidget(QLabel("data.yaml:"), 1, 0)
        self.yaml_path = QLineEdit(); self.yaml_path.setPlaceholderText("학습 시 생성된 data.yaml")
        m.addWidget(self.yaml_path, 1, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary")
        b.clicked.connect(self._browse_yaml); m.addWidget(b, 1, 2)

        m.addWidget(QLabel("Split:"), 2, 0)
        self.split_combo = QComboBox(); self.split_combo.addItems(["val", "train", "test"])
        m.addWidget(self.split_combo, 2, 1)

        m.addWidget(QLabel("Image Size:"), 3, 0)
        from PyQt5.QtWidgets import QSpinBox
        self.imgsz = QSpinBox(); self.imgsz.setRange(64, 2048); self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        m.addWidget(self.imgsz, 3, 1)
        m.setColumnStretch(1, 1)
        lv.addWidget(mg)

        self.run_btn = QPushButton("▶  평가 실행")
        self.run_btn.clicked.connect(self._run_eval)
        lv.addWidget(self.run_btn)

        res_l = QLabel("평가 결과"); res_l.setObjectName("section_title")
        lv.addWidget(res_l)
        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(["지표", "값"])
        hh = self.result_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.result_table.verticalHeader().setVisible(False)
        lv.addWidget(self.result_table)

        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(120)
        lv.addWidget(self.log_box)
        lv.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0, 0, 0, 0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 오른쪽: Ultralytics figure 표시 ───────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w); rv.setContentsMargins(14, 14, 14, 14); rv.setSpacing(8)
        fig_l = QLabel("결과 그래프  (Ultralytics 생성)"); fig_l.setObjectName("section_title")
        rv.addWidget(fig_l)

        fh = QHBoxLayout()
        fh.addWidget(QLabel("그래프 선택:"))
        self.fig_combo = QComboBox()
        self.fig_combo.currentTextChanged.connect(self._show_figure)
        fh.addWidget(self.fig_combo, 1)
        rv.addLayout(fh)

        self.fig_scroll = QScrollArea(); self.fig_scroll.setWidgetResizable(True)
        self.fig_label = QLabel("평가 실행 후 결과 그래프가 여기 표시됩니다")
        self.fig_label.setAlignment(Qt.AlignCenter)
        self.fig_label.setStyleSheet("color:#9ca3af;")
        self.fig_scroll.setWidget(self.fig_label)
        rv.addWidget(self.fig_scroll, 1)

        splitter.addWidget(right_w)
        splitter.setSizes([420, 580])
        outer.addWidget(splitter)

    # ── helpers ──────────────────────────────────────────────────
    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "모델 파일", "", "PyTorch (*.pt *.pth);;All (*)")
        if path: self.model_path.setText(path)

    def _browse_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "data.yaml", "", "YAML (*.yaml *.yml);;All (*)")
        if path: self.yaml_path.setText(path)

    def _run_eval(self):
        mp = self.model_path.text().strip()
        yp = self.yaml_path.text().strip()
        if not mp or not os.path.isfile(mp):
            self.log_box.append("[오류] 모델 파일(.pt)을 선택하세요"); return
        if not yp or not os.path.isfile(yp):
            self.log_box.append("[오류] data.yaml을 선택하세요"); return

        self.run_btn.setEnabled(False); self.log_box.clear()
        self.result_table.setRowCount(0)
        config = {
            "model_path": mp, "data_yaml": yp,
            "split": self.split_combo.currentText(),
            "imgsz": self.imgsz.value(),
        }
        self.worker = EvalWorker(config)
        self.worker.result_ready.connect(self._on_result)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_result(self, results):
        rows = [
            ("mAP50",     results.get("mAP50")),
            ("mAP50-95",  results.get("mAP50-95")),
            ("Precision", results.get("Precision")),
            ("Recall",    results.get("Recall")),
        ]
        for name, val in rows:
            r = self.result_table.rowCount()
            self.result_table.insertRow(r)
            self.result_table.setItem(r, 0, QTableWidgetItem(name))
            self.result_table.setItem(r, 1, QTableWidgetItem(f"{val:.4f}" if val is not None else "—"))
        # per-class
        for cname, ap in results.get("per_class", {}).items():
            r = self.result_table.rowCount()
            self.result_table.insertRow(r)
            self.result_table.setItem(r, 0, QTableWidgetItem(f"AP50 [{cname}]"))
            self.result_table.setItem(r, 1, QTableWidgetItem(f"{ap:.4f}"))

    def _on_finished(self, save_dir):
        self.run_btn.setEnabled(True)
        self.log_box.append("✔ 평가 완료")
        if save_dir and os.path.isdir(save_dir):
            self.log_box.append(f"📁 결과 폴더: {save_dir}")
            self._load_figures(save_dir)

    def _load_figures(self, save_dir):
        """결과 폴더에서 png 그래프 목록 채우기"""
        self._fig_dir = save_dir
        pngs = sorted([f for f in os.listdir(save_dir) if f.lower().endswith(".png")])
        self.fig_combo.blockSignals(True)
        self.fig_combo.clear()
        self.fig_combo.addItems(pngs)
        self.fig_combo.blockSignals(False)
        if pngs:
            # confusion_matrix / PR_curve 우선 표시
            preferred = next((p for p in pngs if "confusion" in p.lower()), pngs[0])
            self.fig_combo.setCurrentText(preferred)
            self._show_figure(preferred)

    def _show_figure(self, name):
        if not name or not hasattr(self, "_fig_dir"):
            return
        path = os.path.join(self._fig_dir, name)
        if not os.path.isfile(path):
            return
        pix = QPixmap(path)
        if not pix.isNull():
            self.fig_label.setPixmap(pix)
            self.fig_label.resize(pix.size())
        else:
            self.fig_label.setText(f"이미지를 열 수 없음: {name}")

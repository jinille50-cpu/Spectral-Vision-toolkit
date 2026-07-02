import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea, QTextEdit, QProgressBar, QSplitter,
    QLineEdit, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# YOLO 버전 × 크기
YOLO_VERSIONS = {
    "YOLOv8":  "yolov8{}.pt",
    "YOLOv10": "yolov10{}.pt",
    "YOLO11":  "yolo11{}.pt",
    "YOLO12":  "yolo12{}.pt",
}
YOLO_SIZES = ["n", "s", "m", "l"]


# ═══════════════════════════════════════════════════════════════
#  학습 Worker
# ═══════════════════════════════════════════════════════════════
class TrainWorker(QThread):
    # epoch, total, box_loss, cls_loss, mAP50, mAP50_95
    epoch_done = pyqtSignal(int, int, object, object, object, object)
    fold_start = pyqtSignal(int, int)   # fold_idx, total_folds (0,0 = Full 학습)
    log_msg    = pyqtSignal(str)
    finished   = pyqtSignal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from yolo_module_wrap import build_yaml, train, train_kfold

        cfg = self.config
        aug = cfg.get("aug", None)
        # train()에 공통 전달할 학습 파라미터
        base = {
            "model": cfg["model"], "epochs": cfg["epochs"], "batch": cfg["batch"],
            "imgsz": cfg["imgsz"], "lr0": cfg["lr0"], "optimizer": cfg["optimizer"],
            "patience": cfg["patience"], "project": cfg["project"],
        }
        try:
            if cfg["split_mode"] == "kfold":
                # ── K-Fold: build + K회 학습 + Full 재학습 ──────────
                self.log_msg.emit("[K-Fold] 데이터 분할 + 학습 시작...")
                kcfg = {
                    **base,
                    "kfold_k":       cfg.get("kfold_k", 5),
                    "train_img_dir": cfg["train_img_dir"],
                    "train_lbl_dir": cfg["train_lbl_dir"],
                    "img_ext":       cfg.get("img_ext", "png"),
                    "class_names":   cfg["class_names"],
                    "aug":           aug,
                    "full_retrain":  True,
                }
                result = train_kfold(
                    kcfg,
                    on_epoch=self._on_epoch,
                    on_log=self.log_msg.emit,
                    should_stop=lambda: self._stop,
                    on_fold=self.fold_start.emit,
                )
            else:
                # ── manual / holdout: 단일 data.yaml + 1회 학습 ─────
                self.log_msg.emit("[1/2] data.yaml 생성 중...")
                yaml_path = build_yaml(
                    out_dir=cfg["project"],
                    class_names=cfg["class_names"],
                    split_mode=cfg["split_mode"],
                    train_img_dir=cfg["train_img_dir"],
                    train_lbl_dir=cfg["train_lbl_dir"],
                    val_img_dir=cfg.get("val_img_dir"),
                    val_lbl_dir=cfg.get("val_lbl_dir"),
                    img_ext=cfg.get("img_ext", "png"),
                    holdout_ratio=cfg.get("holdout_ratio", 0.2),
                    aug=aug,
                    log=self.log_msg.emit,
                )
                self.log_msg.emit("[2/2] 학습 시작...")
                result = train(
                    {**base, "data_yaml": yaml_path, "name": cfg["name"]},
                    on_epoch=self._on_epoch,
                    on_log=self.log_msg.emit,
                    should_stop=lambda: self._stop,
                )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
            self.finished.emit({"success": False, "error": str(e)})

    def _on_epoch(self, ep, total, m):
        self.epoch_done.emit(ep, total, m.get("box_loss"), m.get("cls_loss"),
                             m.get("mAP50"), m.get("mAP50_95"))


# ═══════════════════════════════════════════════════════════════
#  TrainTab UI
# ═══════════════════════════════════════════════════════════════
class TrainTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self.worker = None
        self._data_tab = None
        self.box_loss_data = []
        self.cls_loss_data = []
        self.map50_data    = []
        self.map5095_data  = []
        self._last_ep      = 0   # 중복 epoch 콜백(최종 검증) 방지용
        self._build()

    def set_data_tab(self, data_tab):
        self._data_tab = data_tab

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽: 설정 ───────────────────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        lv = QVBoxLayout(inner); lv.setSpacing(12); lv.setContentsMargins(16, 16, 16, 16)

        # 모델 선택
        mg = QGroupBox("모델 선택")
        m = QGridLayout(mg)
        m.addWidget(QLabel("YOLO 버전:"), 0, 0)
        self.version_combo = QComboBox(); self.version_combo.addItems(list(YOLO_VERSIONS.keys()))
        self.version_combo.setCurrentText("YOLOv8")
        m.addWidget(self.version_combo, 0, 1)
        m.addWidget(QLabel("모델 크기:"), 1, 0)
        self.size_combo = QComboBox(); self.size_combo.addItems(YOLO_SIZES)
        m.addWidget(self.size_combo, 1, 1)
        self.model_name_lbl = QLabel("→ yolov8n.pt")
        self.model_name_lbl.setObjectName("hint")
        m.addWidget(self.model_name_lbl, 2, 0, 1, 2)
        self.version_combo.currentTextChanged.connect(self._update_model_name)
        self.size_combo.currentTextChanged.connect(self._update_model_name)
        m.setColumnStretch(1, 1)
        lv.addWidget(mg)

        # 하이퍼파라미터
        hp = QGroupBox("하이퍼파라미터")
        hg = QGridLayout(hp)
        hg.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 5000); self.epochs.setValue(100)
        hg.addWidget(self.epochs, 0, 1)
        hg.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch = QSpinBox(); self.batch.setRange(1, 256); self.batch.setValue(16)
        hg.addWidget(self.batch, 1, 1)
        hg.addWidget(QLabel("Image Size:"), 2, 0)
        self.imgsz = QSpinBox(); self.imgsz.setRange(64, 2048); self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        hg.addWidget(self.imgsz, 2, 1)
        hg.addWidget(QLabel("LR0:"), 3, 0)
        self.lr0 = QDoubleSpinBox(); self.lr0.setDecimals(5); self.lr0.setRange(1e-5, 1.0)
        self.lr0.setSingleStep(0.001); self.lr0.setValue(0.01)
        hg.addWidget(self.lr0, 3, 1)
        hg.addWidget(QLabel("Optimizer:"), 4, 0)
        self.optimizer = QComboBox()
        self.optimizer.addItems(["auto", "SGD", "Adam", "AdamW", "RMSProp"])
        hg.addWidget(self.optimizer, 4, 1)
        hg.addWidget(QLabel("Patience:"), 5, 0)
        self.patience = QSpinBox(); self.patience.setRange(0, 1000); self.patience.setValue(50)
        hg.addWidget(self.patience, 5, 1)
        lv.addWidget(hp)

        # Augmentation
        aug_g = QGroupBox("Augmentation  (Ultralytics 내장)")
        ag = QGridLayout(aug_g)
        ag.addWidget(QLabel("항목"), 0, 0)
        ag.addWidget(QLabel("사용"), 0, 1)
        ag.addWidget(QLabel("수치"), 0, 2)
        # (key, label, default, range_max)
        aug_defs = [
            ("fliplr",   "좌우 반전 (fliplr)",     0.5,  1.0),
            ("flipud",   "상하 반전 (flipud)",     0.0,  1.0),
            ("mosaic",   "Mosaic",                1.0,  1.0),
            ("mixup",    "MixUp",                 0.0,  1.0),
            ("hsv_h",    "HSV 색조 (hsv_h)",       0.015, 1.0),
            ("hsv_s",    "HSV 채도 (hsv_s)",       0.7,  1.0),
            ("hsv_v",    "HSV 명도 (hsv_v)",       0.4,  1.0),
            ("degrees",  "회전 (degrees)",         0.0,  180.0),
            ("translate","이동 (translate)",       0.1,  1.0),
            ("scale",    "스케일 (scale)",          0.5,  1.0),
        ]
        self.aug_checks = {}; self.aug_spins = {}
        for i, (key, label, val, rmax) in enumerate(aug_defs):
            r = i + 1
            ag.addWidget(QLabel(label), r, 0)
            cb = QCheckBox(); cb.setChecked(val > 0)
            ag.addWidget(cb, r, 1); self.aug_checks[key] = cb
            sp = QDoubleSpinBox(); sp.setDecimals(3); sp.setRange(0.0, rmax)
            sp.setValue(val); sp.setFixedWidth(90)
            ag.addWidget(sp, r, 2); self.aug_spins[key] = sp
        lv.addWidget(aug_g)

        # 결과 저장 경로
        pr = QGroupBox("결과 저장 폴더 (project)")
        pg = QGridLayout(pr)
        self.project_edit = QLineEdit(); self.project_edit.setText("./YOLO_runs")
        pg.addWidget(self.project_edit, 0, 0)
        bp = QPushButton("📁"); bp.setObjectName("secondary"); bp.setFixedWidth(36)
        bp.clicked.connect(self._browse_project); pg.addWidget(bp, 0, 1)
        pg.addWidget(QLabel("실험 이름 (name):"), 1, 0)
        self.name_edit = QLineEdit(); self.name_edit.setText("train")
        pg.addWidget(self.name_edit, 2, 0, 1, 2)
        lv.addWidget(pr)
        lv.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0, 0, 0, 0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 오른쪽: 모니터 ───────────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w); rv.setContentsMargins(16, 16, 16, 16); rv.setSpacing(8)

        bh = QHBoxLayout()
        self.start_btn = QPushButton("▶  학습 시작"); self.start_btn.clicked.connect(self._start_train)
        self.stop_btn = QPushButton("■  중단"); self.stop_btn.setObjectName("secondary")
        self.stop_btn.clicked.connect(self._stop_train); self.stop_btn.setEnabled(False)
        bh.addWidget(self.start_btn); bh.addWidget(self.stop_btn)
        rv.addLayout(bh)

        self.progress = QProgressBar(); self.progress.setRange(0, 100)
        rv.addWidget(self.progress)
        self.epoch_label = QLabel("대기 중..."); self.epoch_label.setObjectName("hint")
        rv.addWidget(self.epoch_label)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(6, 4), facecolor='white')
            self.ax1 = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122)
            self.canvas = Canvas(self.fig); rv.addWidget(self.canvas, 1)
            self.has_mpl = True
        except ImportError:
            rv.addWidget(QLabel("matplotlib 미설치")); self.has_mpl = False

        log_lbl = QLabel("학습 로그"); log_lbl.setObjectName("section_title")
        rv.addWidget(log_lbl)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(160)
        rv.addWidget(self.log_box)

        splitter.addWidget(right_w)
        splitter.setSizes([400, 600])
        outer.addWidget(splitter)

        self._update_model_name()

    # ── helpers ──────────────────────────────────────────────────
    def _update_model_name(self):
        v = self.version_combo.currentText()
        s = self.size_combo.currentText()
        name = YOLO_VERSIONS[v].format(s)
        self.model_name_lbl.setText(f"→ {name}")
        return name

    def _browse_project(self):
        path = QFileDialog.getExistingDirectory(self, "결과 저장 폴더")
        if path: self.project_edit.setText(path)

    def _get_data_config(self):
        """data_tab UI에서 직접 읽기 (저장 안 눌렀어도 동작)"""
        if self._data_tab is None:
            return {}
        dt = self._data_tab
        mode = {0: "manual", 1: "holdout", 2: "kfold"}[dt.split_btns.checkedId()]
        return {
            "split_mode":    mode,
            "train_img_dir": dt.train_img_edit.text().strip(),
            "train_lbl_dir": dt.train_lbl_edit.text().strip(),
            "val_img_dir":   dt.val_img_edit.text().strip(),
            "val_lbl_dir":   dt.val_lbl_edit.text().strip(),
            "holdout_ratio": dt.holdout_spin.value(),
            "kfold_k":       dt.kfold_spin.value(),
            "img_ext":       dt.img_ext.currentText(),
            "class_names":   dt.get_class_names(),
        }

    def _collect_aug(self):
        aug = {}
        for key, cb in self.aug_checks.items():
            aug[key] = self.aug_spins[key].value() if cb.isChecked() else 0.0
        return aug

    def _start_train(self):
        self.box_loss_data.clear(); self.cls_loss_data.clear()
        self.map50_data.clear(); self.map5095_data.clear()
        self._last_ep = 0
        self.log_box.clear(); self.progress.setValue(0)

        data_cfg = self._get_data_config()
        if not data_cfg.get("train_img_dir"):
            self.log_box.append("[오류] ① 데이터 탭에서 Train 이미지 폴더를 지정하세요"); return
        if not data_cfg.get("train_lbl_dir"):
            self.log_box.append("[오류] ① 데이터 탭에서 Train 라벨(txt) 폴더를 지정하세요"); return
        if data_cfg["split_mode"] == "manual" and not data_cfg.get("val_img_dir"):
            self.log_box.append("[오류] manual 방식은 Val 이미지 폴더가 필요합니다\n"
                                "→ Hold-out으로 바꾸거나 Val 폴더를 지정하세요"); return
        if not data_cfg.get("class_names"):
            self.log_box.append("[오류] 클래스 이름이 비어있습니다"); return

        config = {
            **data_cfg,
            "model":     self._update_model_name(),
            "epochs":    self.epochs.value(),
            "batch":     self.batch.value(),
            "imgsz":     self.imgsz.value(),
            "lr0":       self.lr0.value(),
            "optimizer": self.optimizer.currentText(),
            "patience":  self.patience.value(),
            "project":   self.project_edit.text().strip() or "./YOLO_runs",
            "name":      self.name_edit.text().strip() or "train",
            "aug":       self._collect_aug(),
        }
        self.state["last_project"] = config["project"]
        self.state["last_name"]    = config["name"]
        self.state["class_names"]  = config["class_names"]
        self.total_epochs = config["epochs"]

        self.log_box.append(f"[설정] 모델={config['model']}  "
                            f"분할={config['split_mode']}  클래스={config['class_names']}")
        self.worker = TrainWorker(config)
        self.worker.epoch_done.connect(self._on_epoch)
        self.worker.fold_start.connect(self._on_fold_start)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def _stop_train(self):
        if self.worker:
            self.worker.stop()
            self.epoch_label.setText("중단 요청됨 — 현재 epoch 종료 후 정지")

    def _on_fold_start(self, fold_idx, total):
        """fold 시작 시 그래프 초기화"""
        self.box_loss_data.clear(); self.cls_loss_data.clear()
        self.map50_data.clear(); self.map5095_data.clear()
        self._last_ep = 0
        self.progress.setValue(0)
        if total == 0:
            self.epoch_label.setText("Full Data 학습 중...")
        else:
            self.epoch_label.setText(f"Fold {fold_idx} / {total} 학습 중...")
        if self.has_mpl:
            self.ax1.cla(); self.ax2.cla(); self.canvas.draw()

    def _on_epoch(self, ep, total, box_loss, cls_loss, map50, map5095):
        bl = box_loss if box_loss is not None else float('nan')
        cl = cls_loss if cls_loss is not None else float('nan')
        m50 = map50 if map50 is not None else float('nan')
        m5095 = map5095 if map5095 is not None else float('nan')
        if ep == self._last_ep and self.box_loss_data:
            # 같은 epoch 재호출(최종 검증 콜백) → 마지막 점 덮어쓰기
            self.box_loss_data[-1] = bl; self.cls_loss_data[-1] = cl
            self.map50_data[-1] = m50;   self.map5095_data[-1] = m5095
        else:
            self.box_loss_data.append(bl); self.cls_loss_data.append(cl)
            self.map50_data.append(m50);   self.map5095_data.append(m5095)
            self._last_ep = ep
        self.progress.setValue(int(ep / max(total, 1) * 100))
        parts = [f"Epoch {ep}/{total}"]
        if box_loss is not None: parts.append(f"box={box_loss:.4f}")
        if cls_loss is not None: parts.append(f"cls={cls_loss:.4f}")
        if map50 is not None: parts.append(f"mAP50={map50:.4f}")
        if map5095 is not None: parts.append(f"mAP50-95={map5095:.4f}")
        self.epoch_label.setText("  |  ".join(parts))
        if self.has_mpl:
            self._update_plot()

    def _update_plot(self):
        self.ax1.cla(); self.ax2.cla()
        xs = list(range(1, len(self.box_loss_data) + 1))
        self.ax1.plot(xs, self.box_loss_data, label="box_loss", color="#2563eb", lw=1.5)
        self.ax1.plot(xs, self.cls_loss_data, label="cls_loss", color="#ef4444", lw=1.5)
        self.ax1.set_title("Loss", fontsize=10); self.ax1.legend(fontsize=8)
        self.ax1.grid(alpha=0.3); self.ax1.set_xlabel("Epoch")
        self.ax2.plot(xs, self.map50_data, label="mAP50", color="#059669", lw=1.5)
        self.ax2.plot(xs, self.map5095_data, label="mAP50-95", color="#f59e0b", lw=1.5)
        self.ax2.set_title("mAP", fontsize=10); self.ax2.legend(fontsize=8)
        self.ax2.set_ylim(0, 1); self.ax2.grid(alpha=0.3); self.ax2.set_xlabel("Epoch")
        self.fig.tight_layout(); self.canvas.draw()

    def _on_finished(self, result):
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        if result.get("success"):
            self.progress.setValue(100)
            best = result.get("best_pt", "")
            save_dir = result.get("save_dir", "")
            # K-Fold 요약 출력
            if "mean_map50" in result:
                self.log_box.append(
                    f"📊 K-Fold 평균 mAP50 = {result['mean_map50']:.4f} "
                    f"± {result['std_map50']:.4f}  |  "
                    f"mAP50-95 = {result['mean_map5095']:.4f} "
                    f"± {result['std_map5095']:.4f}")
            self.log_box.append("✔ 학습 완료")
            if best:
                self.state["best_pt"] = best
                self.state["save_dir"] = save_dir
                self.log_box.append(f"📦 Best 모델 → 평가/추론 탭에 자동 반영: {best}")
                # 평가/추론 탭에 모델 경로 공유
                self._propagate_model(best, save_dir)
        else:
            self.log_box.append(f"✘ 학습 실패: {result.get('error', '알 수 없음')}")

    def _propagate_model(self, best_pt, save_dir):
        """학습 완료 후 best.pt를 평가/추론 탭에 전달"""
        win = self.window()
        for attr in ("tab_eval", "tab_infer"):
            tab = getattr(win, attr, None)
            if tab and hasattr(tab, "set_model_path"):
                tab.set_model_path(best_pt)
        # 평가탭에 data.yaml 경로도 전달 (K-Fold면 data_full.yaml 우선)
        tab_eval = getattr(win, "tab_eval", None)
        if tab_eval and hasattr(tab_eval, "set_data_yaml"):
            proj = self.state.get("last_project", "")
            for cand in ("data_full.yaml", "data.yaml"):
                yaml_path = os.path.join(proj, cand)
                if os.path.isfile(yaml_path):
                    tab_eval.set_data_yaml(yaml_path)
                    break

"""
학습 탭
- 모델 선택: SVM, KNN, RF, MLP (ANN)
- 전처리: StandardScaler / MinMax / SNV / None
- K-Fold CV
- Confusion Matrix + Classification Report
- 모델 저장
"""
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QSpinBox, QDoubleSpinBox, QSplitter, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class TrainWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)  # 결과 dict 전달
    finished_err = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.cfg = config

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            self.finished_err.emit(f"{e}\n{traceback.format_exc()}")

    def _run(self):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import (confusion_matrix, classification_report,
                                     accuracy_score, f1_score)

        cfg = self.cfg
        X = cfg["X"]; y = cfg["y"]
        model_name = cfg["model"]
        norm_type  = cfg["norm"]
        k          = cfg["k_folds"]

        # 라벨 인코딩
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(le.classes_)
        self.progress.emit(f"클래스 {n_classes}개: {list(le.classes_)}")

        # 정규화
        def apply_norm(arr, kind):
            if kind == "StandardScaler":
                return StandardScaler()
            elif kind == "MinMaxScaler":
                return MinMaxScaler()
            elif kind == "SNV":
                # 각 스펙트럼별 (x - mean) / std — fit_transform 따로 만들어야
                class SNVScaler:
                    def fit(self, X, y=None): return self
                    def transform(self, X):
                        m = X.mean(axis=1, keepdims=True)
                        s = X.std(axis=1, keepdims=True)
                        return (X - m) / (s + 1e-8)
                    def fit_transform(self, X, y=None):
                        return self.transform(X)
                return SNVScaler()
            return None  # none

        def make_model():
            if model_name == "SVM":
                return SVC(kernel=cfg.get("svm_kernel", "rbf"),
                           C=cfg.get("svm_C", 1.0),
                           gamma=cfg.get("svm_gamma", "scale"),
                           probability=True,
                           random_state=42)
            elif model_name == "KNN":
                return KNeighborsClassifier(
                    n_neighbors=cfg.get("knn_k", 5),
                    weights=cfg.get("knn_weights", "uniform"))
            elif model_name == "Random Forest":
                return RandomForestClassifier(
                    n_estimators=cfg.get("rf_n", 200),
                    max_depth=cfg.get("rf_depth", None),
                    n_jobs=1,  # Windows 콘솔창 방지 (멀티프로세싱 비활성)
                    random_state=42)
            elif model_name == "MLP (ANN)":
                hidden = cfg.get("mlp_hidden", (128, 64))
                return MLPClassifier(
                    hidden_layer_sizes=hidden,
                    max_iter=cfg.get("mlp_iter", 500),
                    early_stopping=True,
                    random_state=42)

        # K-Fold CV
        self.progress.emit(f"\n[{k}-Fold Stratified CV 시작]")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_results = []
        all_preds = np.zeros_like(y_enc)
        all_probs = np.zeros((len(y_enc), n_classes), dtype=float)  # ROC용
        has_proba = True

        for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y_enc), 1):
            X_tr, X_vl = X[tr_idx], X[vl_idx]
            y_tr, y_vl = y_enc[tr_idx], y_enc[vl_idx]

            scaler = apply_norm(X_tr, norm_type)
            if scaler is not None:
                X_tr = scaler.fit_transform(X_tr)
                X_vl = scaler.transform(X_vl)

            model = make_model()
            model.fit(X_tr, y_tr)
            pred = model.predict(X_vl)
            all_preds[vl_idx] = pred

            # 확률 수집 (ROC Curve용)
            if has_proba and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_vl)
                    # 모델 클래스 순서를 전체 클래스 순서에 맞게 매핑
                    for col, cls in enumerate(model.classes_):
                        all_probs[vl_idx, cls] = proba[:, col]
                except Exception:
                    has_proba = False
            else:
                has_proba = False

            acc = accuracy_score(y_vl, pred)
            f1  = f1_score(y_vl, pred, average='macro', zero_division=0)
            fold_results.append({"fold": fold, "acc": acc, "f1": f1,
                                 "n_train": len(tr_idx), "n_val": len(vl_idx)})
            self.progress.emit(f"  Fold {fold}/{k} | Acc {acc:.4f} | F1 {f1:.4f} (train {len(tr_idx)} / val {len(vl_idx)})")

        # 전체 결과 (CV 기반)
        cv_acc = accuracy_score(y_enc, all_preds)
        cv_f1  = f1_score(y_enc, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(y_enc, all_preds)
        report = classification_report(y_enc, all_preds,
                                       target_names=le.classes_,
                                       zero_division=0, digits=4)

        self.progress.emit(f"\n[K-Fold 종합]")
        self.progress.emit(f"  평균 Acc: {cv_acc:.4f}")
        self.progress.emit(f"  평균 F1 (macro): {cv_f1:.4f}")

        # 전체 데이터로 final 모델 학습 (배포용)
        self.progress.emit(f"\n[Final 모델 — 전체 데이터로 학습]")
        scaler_final = apply_norm(X, norm_type)
        X_final = scaler_final.fit_transform(X) if scaler_final else X.copy()
        final_model = make_model()
        final_model.fit(X_final, y_enc)
        self.progress.emit(f"✔ Final 모델 학습 완료")

        self.finished_ok.emit({
            "model":       final_model,
            "scaler":      scaler_final,
            "label_enc":   le,
            "classes":     le.classes_.tolist(),
            "cv_acc":      cv_acc,
            "cv_f1":       cv_f1,
            "cm":          cm,
            "report":      report,
            "fold_results":fold_results,
            "y_true":      y_enc,
            "y_pred":      all_preds,
            "y_probs":     all_probs if has_proba else None,
            "model_name":  model_name,
            "norm_type":   norm_type,
        })


class TrainTab(QWidget):
    model_trained = pyqtSignal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.has_mpl = True
            self._Canvas = FigureCanvasQTAgg; self._Figure = Figure
        except ImportError:
            self.has_mpl = False
        self.canvas = None
        self.roc_canvas = None
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # 상단: DB 정보
        self.db_info = QLabel("라이브러리: 0개 샘플 · 0개 클래스")
        main.addWidget(self.db_info)

        # ── 설정 + 결과 (좌우 분할) ──
        split = QSplitter(Qt.Horizontal)

        # 좌측: 모델 설정
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0)

        # 모델 선택
        model_g = QGroupBox("모델 선택")
        mg = QGridLayout(model_g)
        mg.addWidget(QLabel("모델:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["SVM", "KNN", "Random Forest", "MLP (ANN)"])
        self.model_combo.setCurrentText("KNN")   # 기본 KNN
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        mg.addWidget(self.model_combo, 0, 1)

        mg.addWidget(QLabel("정규화:"), 1, 0)
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["StandardScaler", "MinMaxScaler", "SNV", "None"])
        self.norm_combo.setCurrentText("SNV")   # 기본 SNV
        mg.addWidget(self.norm_combo, 1, 1)

        mg.addWidget(QLabel("K-Folds:"), 2, 0)
        self.k_spin = QSpinBox(); self.k_spin.setRange(2, 20); self.k_spin.setValue(5)
        mg.addWidget(self.k_spin, 2, 1)
        ll.addWidget(model_g)

        # 하이퍼파라미터 (모델별)
        self.hp_g = QGroupBox("하이퍼파라미터")
        self.hp_layout = QGridLayout(self.hp_g)
        ll.addWidget(self.hp_g)
        self._build_hp_widgets()
        self._on_model_changed("KNN")

        # 학습 버튼
        self.btn_train = QPushButton("▶  학습 시작")
        self.btn_train.clicked.connect(self._start_train)
        ll.addWidget(self.btn_train)

        # 모델 저장/불러오기
        save_g = QGroupBox("모델 저장/불러오기")
        sg = QHBoxLayout(save_g)
        btn_save = QPushButton("💾  모델 저장"); btn_save.setObjectName("secondary")
        btn_save.clicked.connect(self._save_model)
        sg.addWidget(btn_save)
        btn_load = QPushButton("📂  모델 불러오기"); btn_load.setObjectName("secondary")
        btn_load.clicked.connect(self._load_model)
        sg.addWidget(btn_load)
        ll.addWidget(save_g)

        ll.addStretch()
        split.addWidget(left)

        # 우측: 결과
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0)

        # 결과 통계
        self.result_label = QLabel("학습 결과가 여기 표시됩니다")
        self.result_label.setStyleSheet("font-size: 14px; padding: 8px;")
        rl.addWidget(self.result_label)

        # 옵션 + 저장 버튼 (공통 상단 바)
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
        self.btn_save_all.clicked.connect(self._save_all_results)
        opt_bar.addWidget(self.btn_save_all)
        rl.addLayout(opt_bar)

        # CM (왼쪽) + ROC (오른쪽) 가로 배치
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

        # 분류 리포트
        self.report_box = QTextEdit()
        self.report_box.setReadOnly(True)
        self.report_box.setMaximumHeight(180)
        self.report_box.setStyleSheet("font-family: 'Courier New', monospace; font-size: 11px;")
        rl.addWidget(self.report_box)

        split.addWidget(right)
        split.setSizes([400, 900])
        main.addWidget(split, 1)

        # 로그
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(120)
        self.log_box.setStyleSheet("font-family: 'Courier New', monospace; font-size: 11px;")
        main.addWidget(self.log_box)

    def _build_hp_widgets(self):
        """모델별 하이퍼파라미터를 컨테이너 위젯에 미리 다 넣어둠.
        모델 변경 시 컨테이너만 show/hide → 위젯이 부모를 잃지 않음
        """
        from PyQt5.QtWidgets import QWidget

        # === SVM 컨테이너 ===
        self.svm_box = QWidget()
        svm_l = QGridLayout(self.svm_box)
        svm_l.setContentsMargins(0, 0, 0, 0)
        self.svm_kernel = QComboBox(); self.svm_kernel.addItems(["rbf", "linear", "poly", "sigmoid"])
        self.svm_C      = QDoubleSpinBox(); self.svm_C.setRange(0.001, 1000); self.svm_C.setValue(1.0); self.svm_C.setDecimals(3)
        self.svm_gamma  = QComboBox(); self.svm_gamma.addItems(["scale", "auto"])
        svm_l.addWidget(QLabel("Kernel:"), 0, 0); svm_l.addWidget(self.svm_kernel, 0, 1)
        svm_l.addWidget(QLabel("C:"), 1, 0);      svm_l.addWidget(self.svm_C, 1, 1)
        svm_l.addWidget(QLabel("Gamma:"), 2, 0);  svm_l.addWidget(self.svm_gamma, 2, 1)
        svm_l.setColumnStretch(1, 1)

        # === KNN 컨테이너 ===
        self.knn_box = QWidget()
        knn_l = QGridLayout(self.knn_box)
        knn_l.setContentsMargins(0, 0, 0, 0)
        self.knn_k       = QSpinBox(); self.knn_k.setRange(1, 50); self.knn_k.setValue(5)
        self.knn_weights = QComboBox(); self.knn_weights.addItems(["uniform", "distance"])
        knn_l.addWidget(QLabel("N neighbors:"), 0, 0); knn_l.addWidget(self.knn_k, 0, 1)
        knn_l.addWidget(QLabel("Weights:"), 1, 0);     knn_l.addWidget(self.knn_weights, 1, 1)
        knn_l.setColumnStretch(1, 1)

        # === RF 컨테이너 ===
        self.rf_box = QWidget()
        rf_l = QGridLayout(self.rf_box)
        rf_l.setContentsMargins(0, 0, 0, 0)
        self.rf_n      = QSpinBox(); self.rf_n.setRange(10, 2000); self.rf_n.setValue(200); self.rf_n.setSingleStep(50)
        self.rf_depth  = QSpinBox(); self.rf_depth.setRange(0, 100); self.rf_depth.setValue(0); self.rf_depth.setSpecialValueText("자동")
        rf_l.addWidget(QLabel("N estimators:"), 0, 0); rf_l.addWidget(self.rf_n, 0, 1)
        rf_l.addWidget(QLabel("Max depth:"), 1, 0);    rf_l.addWidget(self.rf_depth, 1, 1)
        rf_l.setColumnStretch(1, 1)

        # === MLP 컨테이너 ===
        self.mlp_box = QWidget()
        mlp_l = QGridLayout(self.mlp_box)
        mlp_l.setContentsMargins(0, 0, 0, 0)
        self.mlp_hidden = QLineEdit("128,64"); self.mlp_hidden.setPlaceholderText("128,64 (쉼표 구분)")
        self.mlp_iter   = QSpinBox(); self.mlp_iter.setRange(50, 5000); self.mlp_iter.setValue(500); self.mlp_iter.setSingleStep(50)
        mlp_l.addWidget(QLabel("Hidden layers:"), 0, 0); mlp_l.addWidget(self.mlp_hidden, 0, 1)
        mlp_l.addWidget(QLabel("Max iter:"), 1, 0);     mlp_l.addWidget(self.mlp_iter, 1, 1)
        mlp_l.setColumnStretch(1, 1)

        # hp_layout에 모든 컨테이너 추가 (기본은 다 숨김)
        self.hp_layout.addWidget(self.svm_box, 0, 0)
        self.hp_layout.addWidget(self.knn_box, 0, 0)
        self.hp_layout.addWidget(self.rf_box,  0, 0)
        self.hp_layout.addWidget(self.mlp_box, 0, 0)
        self.svm_box.hide(); self.knn_box.hide()
        self.rf_box.hide(); self.mlp_box.hide()

    def _on_model_changed(self, name):
        """컨테이너만 show/hide — 위젯 부모는 그대로 유지"""
        boxes = {
            "SVM": self.svm_box,
            "KNN": self.knn_box,
            "Random Forest": self.rf_box,
            "MLP (ANN)": self.mlp_box,
        }
        for n, b in boxes.items():
            b.setVisible(n == name)

    def refresh_db_info(self):
        db = self.state["database"]
        classes = {s["label"] for s in db}
        self.db_info.setText(
            f"<b>라이브러리:</b> {len(db)}개 샘플 · {len(classes)}개 클래스 "
            f"({', '.join(sorted(classes)[:8])}{'...' if len(classes)>8 else ''})"
        )

    # ────────────────────────────────────────────────────────
    def _start_train(self):
        db = self.state["database"]
        if len(db) < 10:
            QMessageBox.warning(self, "데이터 부족", "최소 10개 샘플이 필요합니다.")
            return
        classes = {s["label"] for s in db}
        if len(classes) < 2:
            QMessageBox.warning(self, "클래스 부족", "최소 2개 클래스가 필요합니다.")
            return

        # 클래스당 샘플 수 확인 (Stratified K-Fold용)
        from collections import Counter
        cnt = Counter(s["label"] for s in db)
        k = self.k_spin.value()
        if min(cnt.values()) < k:
            QMessageBox.warning(self, "K-Fold 불가",
                f"가장 적은 클래스가 {min(cnt.values())}개입니다. K={k}로 분할 불가.\n"
                f"K를 줄이거나 데이터를 더 추가하세요.")
            return

        X = np.stack([s["spectrum"] for s in db])
        y = np.array([s["label"]    for s in db])

        config = {
            "X": X, "y": y,
            "model": self.model_combo.currentText(),
            "norm":  self.norm_combo.currentText(),
            "k_folds": k,
            # 모델별 하이퍼파라미터
            "svm_kernel": self.svm_kernel.currentText(),
            "svm_C":      self.svm_C.value(),
            "svm_gamma":  self.svm_gamma.currentText(),
            "knn_k":      self.knn_k.value(),
            "knn_weights":self.knn_weights.currentText(),
            "rf_n":       self.rf_n.value(),
            "rf_depth":   None if self.rf_depth.value() == 0 else self.rf_depth.value(),
            "mlp_hidden": tuple(int(x.strip()) for x in self.mlp_hidden.text().split(',') if x.strip()),
            "mlp_iter":   self.mlp_iter.value(),
        }

        self.btn_train.setEnabled(False)
        self.log_box.clear()
        self.report_box.clear()

        self.worker = TrainWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.finished_err.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg):
        self.log_box.append(msg)

    def _on_finished(self, result):
        self.btn_train.setEnabled(True)

        # 결과 표시
        self.result_label.setText(
            f"<b>{result['model_name']}</b>  ·  norm={result['norm_type']}  ·  "
            f"<b>Accuracy: {result['cv_acc']:.4f}</b>  ·  F1 (macro): {result['cv_f1']:.4f}"
        )
        self.report_box.setPlainText(result["report"])
        self._draw_cm(result["cm"], result["classes"])

        # ROC 데이터 저장 + 그리기
        self.last_y_true = result.get("y_true")
        self.last_y_probs = result.get("y_probs")
        self._draw_roc(self.last_y_true, self.last_y_probs, result["classes"])

        # fold 정보도 저장 (전체 결과 저장 시 사용)
        self.last_fold_results = result.get("fold_results", [])

        # state에 저장
        self.state["trained_model"] = result["model"]
        self.state["model_meta"] = {
            "scaler":     result["scaler"],
            "label_enc":  result["label_enc"],
            "classes":    result["classes"],
            "model_name": result["model_name"],
            "norm_type":  result["norm_type"],
            "cv_acc":     result["cv_acc"],
            "cv_f1":      result["cv_f1"],
        }
        self.model_trained.emit()

    def _on_error(self, msg):
        self.btn_train.setEnabled(True)
        self.log_box.append(f"\n[오류]\n{msg}")
        QMessageBox.critical(self, "학습 실패", msg)

    def _refresh_cm(self):
        """옵션 변경 시 CM 다시 그림"""
        if hasattr(self, 'last_cm') and self.last_cm is not None:
            self._draw_cm(self.last_cm, self.last_classes)

    def _draw_cm(self, cm, classes):
        if not self.has_mpl: return
        # 마지막 결과 저장 (재그림용)
        self.last_cm = cm
        self.last_classes = classes

        if self.canvas:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None); self.canvas = None

        fig = self._build_cm_figure(cm, classes,
                                    show_pct=self.cm_normalize.isChecked(),
                                    show_count=self.cm_show_count.isChecked())
        self.canvas = self._Canvas(fig)
        self.canvas_container.addWidget(self.canvas)
        self.canvas.draw()

    def _build_cm_figure(self, cm, classes, show_pct=True, show_count=True):
        """CM figure 생성 (재사용 가능)"""
        import numpy as np
        fig = self._Figure(figsize=(5.5, 5), facecolor='white')
        ax = fig.add_subplot(111)

        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        im = ax.imshow(cm_norm, cmap='Purples', vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        for i in range(len(classes)):
            for j in range(len(classes)):
                parts = []
                if show_count: parts.append(f"{cm[i,j]}")
                if show_pct:   parts.append(f"({cm_norm[i,j]*100:.1f}%)")
                txt = "\n".join(parts) if parts else ""
                color = 'white' if cm_norm[i,j] > 0.5 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        color=color, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    # ─── ROC Curve ───
    def _refresh_roc(self):
        if hasattr(self, 'last_y_true') and self.last_y_true is not None:
            self._draw_roc(self.last_y_true, self.last_y_probs, self.last_classes)

    def _draw_roc(self, y_true, y_probs, classes):
        if not self.has_mpl: return
        if self.roc_canvas:
            self.roc_container.removeWidget(self.roc_canvas)
            self.roc_canvas.setParent(None); self.roc_canvas = None

        fig = self._build_roc_figure(y_true, y_probs, classes)
        if fig is None:
            # 확률 없는 모델 (예: 일부 SVM 설정) → 안내 표시
            from PyQt5.QtWidgets import QLabel
            return
        self.roc_canvas = self._Canvas(fig)
        self.roc_container.addWidget(self.roc_canvas)
        self.roc_canvas.draw()

    def _build_roc_figure(self, y_true, y_probs, classes):
        """One-vs-Rest ROC Curve (멀티클래스)"""
        import numpy as np
        if y_probs is None or y_true is None:
            return None
        try:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
        except Exception:
            return None

        n_classes = len(classes)
        # 이진화 (One-vs-Rest)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2:
            # binarize가 (N,1)로 나오므로 보정
            y_bin = np.hstack([1 - y_bin, y_bin])

        fig = self._Figure(figsize=(5.5, 5), facecolor='white')
        ax = fig.add_subplot(111)

        import matplotlib.cm as mcm
        try:
            cmap = mcm.get_cmap('tab10' if n_classes <= 10 else 'tab20')
        except Exception:
            import matplotlib
            cmap = matplotlib.colormaps['tab10' if n_classes <= 10 else 'tab20']

        # 클래스별 ROC
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=cmap(i % 20), lw=1.8,
                        label=f"{classes[i]} (AUC={roc_auc:.3f})")
            except Exception:
                continue

        # Macro-average ROC
        try:
            all_fpr = np.unique(np.concatenate([
                roc_curve(y_bin[:, i], y_probs[:, i])[0] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= n_classes
            macro_auc = auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr, 'k--', lw=2,
                    label=f"Macro-avg (AUC={macro_auc:.3f})")
        except Exception:
            pass

        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (One-vs-Rest)")
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────
    # 저장 함수들
    # ────────────────────────────────────────────────────────
    def _save_cm(self):
        if not hasattr(self, 'last_cm') or self.last_cm is None:
            QMessageBox.warning(self, "저장 불가", "먼저 모델을 학습하세요.")
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "Confusion Matrix 저장", "confusion_matrix.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not fp: return
        try:
            fig = self._build_cm_figure(
                self.last_cm, self.last_classes,
                show_pct=self.cm_normalize.isChecked(),
                show_count=self.cm_show_count.isChecked())
            fig.savefig(fp, dpi=300, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
            self.log_box.append(f"\n✔ CM 저장: {fp}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    def _save_roc(self):
        if not hasattr(self, 'last_y_true') or self.last_y_true is None:
            QMessageBox.warning(self, "저장 불가", "먼저 모델을 학습하세요.")
            return
        if self.last_y_probs is None:
            QMessageBox.warning(self, "저장 불가",
                "이 모델은 확률 예측을 지원하지 않아 ROC를 그릴 수 없습니다.")
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "ROC Curve 저장", "roc_curve.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not fp: return
        try:
            fig = self._build_roc_figure(self.last_y_true, self.last_y_probs, self.last_classes)
            if fig is None:
                QMessageBox.warning(self, "저장 불가", "ROC를 생성할 수 없습니다.")
                return
            fig.savefig(fp, dpi=300, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
            self.log_box.append(f"\n✔ ROC 저장: {fp}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    def _save_all_results(self):
        """학습 결과 일괄 저장 — CM + ROC + report + meta json"""
        import os
        if not hasattr(self, 'last_cm') or self.last_cm is None:
            QMessageBox.warning(self, "저장 불가", "먼저 모델을 학습하세요.")
            return
        d = QFileDialog.getExistingDirectory(self, "결과 저장 폴더 선택")
        if not d: return
        try:
            import json, numpy as np
            meta = self.state.get("model_meta", {})
            stem = f"{meta.get('model_name','model')}_{meta.get('norm_type','none')}"

            # 1) CM 그림
            fig = self._build_cm_figure(
                self.last_cm, self.last_classes,
                show_pct=self.cm_normalize.isChecked(),
                show_count=self.cm_show_count.isChecked())
            cm_path = os.path.join(d, f"{stem}_confusion_matrix.png")
            fig.savefig(cm_path, dpi=300, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)

            # 2) CM 데이터 (numpy + csv)
            np.save(os.path.join(d, f"{stem}_cm.npy"), self.last_cm)
            # CSV (라벨 헤더 포함)
            with open(os.path.join(d, f"{stem}_cm.csv"), "w", encoding="utf-8") as f:
                f.write("," + ",".join(self.last_classes) + "\n")
                for i, cls in enumerate(self.last_classes):
                    f.write(cls + "," + ",".join(str(v) for v in self.last_cm[i]) + "\n")

            # 2-b) ROC Curve (확률 있을 때만)
            roc_saved = False
            if getattr(self, 'last_y_probs', None) is not None:
                roc_fig = self._build_roc_figure(self.last_y_true, self.last_y_probs, self.last_classes)
                if roc_fig is not None:
                    roc_fig.savefig(os.path.join(d, f"{stem}_roc_curve.png"),
                                    dpi=300, bbox_inches='tight')
                    plt.close(roc_fig)
                    roc_saved = True

            # 3) Classification report (txt)
            with open(os.path.join(d, f"{stem}_report.txt"), "w", encoding="utf-8") as f:
                f.write(self.report_box.toPlainText())

            # 4) 메타 정보 (json)
            meta_info = {
                "model_name": meta.get("model_name"),
                "norm_type":  meta.get("norm_type"),
                "classes":    self.last_classes.tolist() if hasattr(self.last_classes, 'tolist') else list(self.last_classes),
                "cv_acc":     meta.get("cv_acc"),
                "cv_f1":      meta.get("cv_f1"),
                "n_samples":  int(self.last_cm.sum()),
            }
            # fold_results도 있다면 추가 (저장 안 했으면 skip)
            if hasattr(self, 'last_fold_results'):
                meta_info["fold_results"] = self.last_fold_results
            with open(os.path.join(d, f"{stem}_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta_info, f, indent=2, ensure_ascii=False)

            self.log_box.append(f"\n✔ 전체 결과 저장: {d}")
            self.log_box.append(f"  - {stem}_confusion_matrix.png")
            if roc_saved:
                self.log_box.append(f"  - {stem}_roc_curve.png")
            self.log_box.append(f"  - {stem}_cm.npy, {stem}_cm.csv")
            self.log_box.append(f"  - {stem}_report.txt")
            self.log_box.append(f"  - {stem}_meta.json")
            n_files = 5 + (1 if roc_saved else 0)
            QMessageBox.information(self, "저장 완료", f"{n_files}개 파일 저장됨\n{d}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    # ────────────────────────────────────────────────────────
    # 모델 저장/불러오기
    # ────────────────────────────────────────────────────────
    def _save_model(self):
        if self.state.get("trained_model") is None:
            QMessageBox.warning(self, "저장 불가", "먼저 모델을 학습하세요.")
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "모델 저장", "polymer_model.joblib", "Joblib (*.joblib)")
        if not fp: return
        try:
            import joblib
            meta = self.state["model_meta"]
            payload = {
                "model":      self.state["trained_model"],
                "scaler":     meta["scaler"],
                "label_enc":  meta["label_enc"],
                "classes":    meta["classes"],
                "model_name": meta["model_name"],
                "norm_type":  meta["norm_type"],
                "cv_acc":     meta["cv_acc"],
                "cv_f1":      meta["cv_f1"],
                "wavelengths": self.state.get("wavelengths"),
            }
            joblib.dump(payload, fp)
            self.log_box.append(f"\n✔ 모델 저장: {fp}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    def _load_model(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "모델 불러오기", "", "Joblib (*.joblib)")
        if not fp: return
        try:
            import joblib
            payload = joblib.load(fp)
            self.state["trained_model"] = payload["model"]
            self.state["model_meta"] = {k: payload[k] for k in
                ["scaler", "label_enc", "classes", "model_name", "norm_type", "cv_acc", "cv_f1"]}
            if payload.get("wavelengths") is not None and self.state.get("wavelengths") is None:
                self.state["wavelengths"] = payload["wavelengths"]

            self.result_label.setText(
                f"<b>{payload['model_name']}</b>  ·  norm={payload['norm_type']}  ·  "
                f"Acc {payload['cv_acc']:.4f}  ·  F1 {payload['cv_f1']:.4f}  (로드됨)"
            )
            self.log_box.append(f"\n✔ 모델 로드: {fp}")
            self.model_trained.emit()
        except Exception as e:
            QMessageBox.critical(self, "불러오기 실패", str(e))

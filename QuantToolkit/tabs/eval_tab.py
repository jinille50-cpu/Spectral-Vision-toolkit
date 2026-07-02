import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QLineEdit,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QCheckBox, QTextEdit, QFrame, QSpinBox, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# ═══════════════════════════════════════════════════════════════
#  EvalWorker
# ═══════════════════════════════════════════════════════════════
class EvalWorker(QThread):
    result_ready = pyqtSignal(dict, object, object)   # metrics, all_preds, all_targets
    log_msg      = pyqtSignal(str)
    finished     = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self._run_eval()
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()

    def _load_image(self, path):
        import numpy as np
        from PIL import Image
        import scipy.io as sio
        ext = path.rsplit(".", 1)[-1].lower()
        if ext == "mat":
            d = sio.loadmat(path)
            return np.array(list(d.values())[-1]).astype(np.float32)
        elif ext == "npy":
            return np.load(path).astype(np.float32)
        else:
            return np.array(Image.open(path)).astype(np.float32)

    def _load_label(self, path):
        import numpy as np
        from PIL import Image
        ext = path.rsplit(".", 1)[-1].lower()
        if ext == "mat":
            import scipy.io as sio
            d = sio.loadmat(path)
            return np.array(list(d.values())[-1])
        elif ext == "npy":
            return np.load(path)
        else:
            return np.array(Image.open(path))

    def _build_model(self, cfg):
        """config.json 기반 모델 재구성"""
        import torch, torch.nn as nn
        mode = cfg.get("mode", "patch")
        model_name = cfg.get("model_name", "")
        in_ch  = cfg.get("in_channels", 6)
        encoder = cfg.get("encoder", "resnet34")
        activation = cfg.get("activation", "없음 (linear)")

        if mode == "patch":
            from tabs.train_tab import _make_simple_cnn, _make_1d_cnn, _make_mlp
            if model_name == "CNN-2D-Simple":
                return _make_simple_cnn(in_ch, cfg.get("patch_size",9)), "patch"
            elif model_name == "CNN-1D-Spectral":
                return _make_1d_cnn(in_ch), "spectral"
            elif model_name == "MLP-Spectral":
                return _make_mlp(in_ch), "spectral"
            elif model_name.startswith("CNN-2D"):
                import torchvision.models as tvm
                name_map = {
                    "CNN-2D-ResNet18":"resnet18","CNN-2D-ResNet34":"resnet34",
                    "CNN-2D-ResNet50":"resnet50",
                    "CNN-2D-EfficientNet-B0":"efficientnet_b0",
                    "CNN-2D-EfficientNet-B2":"efficientnet_b2",
                }
                tv_name = name_map.get(model_name, "resnet34")
                model = getattr(tvm, tv_name)(weights=None)
                if "resnet" in tv_name:
                    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    model.fc = nn.Linear(model.fc.in_features, 1)
                else:
                    model.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False)
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
                return model, "patch"
        else:
            import segmentation_models_pytorch as smp
            SMP_MAP = {
                "UNet":"Unet","UNet++":"UnetPlusPlus","MAnet":"MAnet","FPN":"FPN",
                "Linknet":"Linknet","PAN":"PAN","DeepLabV3+":"DeepLabV3Plus",
            }
            act_map = {"없음 (linear)": None, "sigmoid":"sigmoid","tanh":"tanh","ReLU":"identity"}
            cls = SMP_MAP.get(model_name, "Unet")
            model = getattr(smp, cls)(
                encoder_name=encoder, encoder_weights=None,
                in_channels=in_ch, classes=1,
                activation=act_map.get(activation, None),
            )
            return model, "dense"

    def _run_eval(self):
        import torch, numpy as np, glob, json
        cfg = self.config

        # config.json 로드 — 모델 구조 파라미터
        config_path = cfg["config_path"]
        with open(config_path, "r", encoding="utf-8") as f:
            mc = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_msg.emit(f"[평가 시작]  디바이스: {device}")

        model, kind = self._build_model(mc)
        state = torch.load(cfg["model_path"], map_location=device, weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=False)
        model = model.to(device); model.eval()
        self.log_msg.emit(f"  모델 로드: {os.path.basename(cfg['model_path'])}")

        # 라벨 변환 파라미터 (학습 때 저장된 값 사용)
        bg      = mc.get("bg_value", 0)
        offset  = mc.get("label_offset", -1.0)
        divisor = mc.get("label_divisor", 10.0)
        if divisor == 0: divisor = 1.0

        # 데이터 로드
        img_dir = cfg["img_dir"]; lbl_dir = cfg["lbl_dir"]
        img_ext = cfg.get("img_ext","mat"); lbl_ext = cfg.get("lbl_ext","png")
        img_files = sorted(glob.glob(os.path.join(img_dir, f"*.{img_ext}")))
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, f"*.{lbl_ext}")))
        if not img_files:
            self.log_msg.emit(f"[오류] 이미지 없음: {img_dir}"); return
        self.log_msg.emit(f"  평가 이미지 {len(img_files)}개")

        mode  = mc.get("mode","patch")
        in_ch = mc.get("in_channels", 6)
        # norm_type 우선, 없으면 use_snv → snv/none
        norm_type = mc.get("norm_type")
        if norm_type is None:
            norm_type = "snv" if mc.get("use_snv", False) else "none"
        all_preds, all_targets = [], []

        if norm_type != "none":
            self.log_msg.emit(f"  정규화: {norm_type}")

        def _normalize(cube, nt):
            if nt == "snv":
                m = np.mean(cube, axis=-1, keepdims=True)
                s = np.std(cube,  axis=-1, keepdims=True)
                return (cube - m) / (s + 1e-8)
            elif nt == "minmax":
                ax = tuple(range(cube.ndim - 1))
                cmin = cube.min(axis=ax, keepdims=True)
                cmax = cube.max(axis=ax, keepdims=True)
                return (cube - cmin) / (cmax - cmin + 1e-8)
            elif nt == "zscore":
                ax = tuple(range(cube.ndim - 1))
                cmean = cube.mean(axis=ax, keepdims=True)
                cstd  = cube.std(axis=ax, keepdims=True)
                return (cube - cmean) / (cstd + 1e-8)
            return cube

        with torch.no_grad():
            for ip, lp in zip(img_files, lbl_files):
                img = self._load_image(ip)
                lbl = self._load_label(lp)
                conc = (lbl.astype(np.float32) + offset) / divisor
                mask_valid = (lbl != bg)

                if img.ndim == 2:
                    img = img[..., None]
                elif img.shape[0] < img.shape[-1]:
                    img = np.transpose(img, (1, 2, 0))

                # 정규화
                img = _normalize(img, norm_type).astype(np.float32)

                if mode == "patch":
                    patch_size = mc.get("patch_size", 9)
                    half = patch_size // 2
                    img_p = np.pad(img, ((half,half),(half,half),(0,0)), mode='edge')

                    ys, xs = np.where(mask_valid)
                    if len(ys) == 0: continue
                    BATCH = 1024
                    preds_img = []
                    for i in range(0, len(ys), BATCH):
                        ybs = ys[i:i+BATCH]; xbs = xs[i:i+BATCH]
                        patches = np.stack([
                            img_p[y:y+patch_size, x:x+patch_size, :]
                            for y, x in zip(ybs, xbs)
                        ])  # (B, P, P, C)
                        # (B, P, P, C) → (B, C, P, P)
                        X = torch.from_numpy(patches).float().permute(0, 3, 1, 2).to(device)
                        out = model(X)
                        if out.dim() > 1: out = out.squeeze(-1)
                        preds_img.append(out.cpu().numpy())
                    pred_vals = np.concatenate(preds_img)
                    targets   = conc[ys, xs]
                    all_preds.append(pred_vals)
                    all_targets.append(targets)

                else:  # dense
                    from utils.pad_utils import pad_tensor, unpad_tensor, get_pad_multiple
                    multiple = mc.get("pad_multiple") or get_pad_multiple(mc.get("model_name","UNet"))
                    if multiple < 32: multiple = 32

                    t = torch.from_numpy(img).float()
                    if t.ndim == 3:
                        t = t.permute(2, 0, 1)
                    elif t.ndim == 2:
                        t = t.unsqueeze(0)
                    t = t.unsqueeze(0).to(device)
                    t_pad, oh, ow = pad_tensor(t, multiple=multiple)
                    out = model(t_pad)
                    out = unpad_tensor(out, oh, ow).squeeze().cpu().numpy()
                    valid_pred = out[mask_valid]
                    valid_target = conc[mask_valid]
                    all_preds.append(valid_pred)
                    all_targets.append(valid_target)

        P = np.concatenate(all_preds)
        T = np.concatenate(all_targets)

        # 지표 계산
        mse  = float(((P - T)**2).mean())
        rmse = float(np.sqrt(mse))
        mae  = float(np.abs(P - T).mean())
        ss_res = float(((T - P)**2).sum())
        ss_tot = float(((T - T.mean())**2).sum()) + 1e-9
        r2 = float(1 - ss_res / ss_tot)
        # Pearson 상관계수
        from numpy import corrcoef
        pearson = float(corrcoef(P, T)[0, 1])
        # Bias (평균 잔차)
        bias = float((P - T).mean())

        metrics = {
            "R²": r2, "RMSE": rmse, "MAE": mae, "MSE": mse,
            "Pearson r": pearson, "Bias (mean residual)": bias,
            "Sample count": len(P),
        }
        self.log_msg.emit(f"  R²={r2:.4f}, RMSE={rmse:.5f}, MAE={mae:.5f}")
        self.result_ready.emit(metrics, P, T)


# ═══════════════════════════════════════════════════════════════
#  EvalTab UI
# ═══════════════════════════════════════════════════════════════
class EvalTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self.worker = None
        self._infer_tab = None
        self._last_preds = None
        self._last_targets = None
        self._build()

    def set_infer_tab(self, tab):
        self._infer_tab = tab

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # 좌측
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        lv = QVBoxLayout(inner); lv.setSpacing(10); lv.setContentsMargins(14,14,14,14)

        m_g = QGroupBox("모델 설정")
        mg = QGridLayout(m_g)
        mg.addWidget(QLabel("모델 파일 (.pth):"), 0, 0)
        self.model_path = QLineEdit()
        self.model_path.textChanged.connect(self._sync_to_infer)
        mg.addWidget(self.model_path, 0, 1)
        btn_m = QPushButton("찾기"); btn_m.setObjectName("secondary")
        btn_m.clicked.connect(self._browse_model)
        mg.addWidget(btn_m, 0, 2)

        mg.addWidget(QLabel("모델 옵션 (_config.json):"), 1, 0)
        self.config_path = QLineEdit()
        self.config_path.textChanged.connect(self._sync_to_infer)
        mg.addWidget(self.config_path, 1, 1)
        btn_cfg = QPushButton("찾기"); btn_cfg.setObjectName("secondary")
        btn_cfg.clicked.connect(self._browse_config)
        mg.addWidget(btn_cfg, 1, 2)

        mg.setColumnStretch(1, 1)
        lv.addWidget(m_g)

        d_g = QGroupBox("데이터 설정")
        dg = QGridLayout(d_g)
        dg.addWidget(QLabel("이미지 폴더:"), 0, 0)
        self.eval_img = QLineEdit()
        self.eval_img.textChanged.connect(self._sync_to_infer)
        dg.addWidget(self.eval_img, 0, 1)
        btn_i = QPushButton("찾기"); btn_i.setObjectName("secondary")
        btn_i.clicked.connect(lambda: self._browse_dir(self.eval_img))
        dg.addWidget(btn_i, 0, 2)
        dg.addWidget(QLabel("라벨 폴더:"), 1, 0)
        self.eval_lbl = QLineEdit()
        dg.addWidget(self.eval_lbl, 1, 1)
        btn_l = QPushButton("찾기"); btn_l.setObjectName("secondary")
        btn_l.clicked.connect(lambda: self._browse_dir(self.eval_lbl))
        dg.addWidget(btn_l, 1, 2)

        dg.addWidget(QLabel("이미지 확장자:"), 2, 0)
        self.img_ext = QComboBox(); self.img_ext.addItems(["mat","npy","png","jpg","tiff"])
        self.img_ext.currentTextChanged.connect(self._sync_to_infer)
        dg.addWidget(self.img_ext, 2, 1)
        dg.addWidget(QLabel("라벨 확장자:"), 3, 0)
        self.lbl_ext = QComboBox(); self.lbl_ext.addItems(["png","mat","npy","tiff"])
        dg.addWidget(self.lbl_ext, 3, 1)

        dg.setColumnStretch(1, 1)
        lv.addWidget(d_g)

        s_g = QGroupBox("저장 폴더 (Plot 저장)")
        sg = QGridLayout(s_g)
        self.save_dir = QLineEdit(); self.save_dir.setPlaceholderText("빈칸 → 저장 안함")
        sg.addWidget(self.save_dir, 0, 0)
        btn_s = QPushButton("찾기"); btn_s.setObjectName("secondary"); btn_s.setFixedWidth(44)
        btn_s.clicked.connect(lambda: self._browse_dir(self.save_dir))
        sg.addWidget(btn_s, 0, 1)
        lv.addWidget(s_g)

        plot_g = QGroupBox("저장할 Plot")
        pv = QVBoxLayout(plot_g)
        self.plot_checks = {}
        for p in ["Scatter (Actual vs Predicted)",
                  "Residual Plot",
                  "Histogram of Residuals",
                  "Predicted vs Actual per concentration level",
                  "Bland-Altman Plot"]:
            cb = QCheckBox(p); cb.setChecked(True)
            self.plot_checks[p] = cb; pv.addWidget(cb)
        lv.addWidget(plot_g)

        f_g = QGroupBox("Figure 옵션")
        fg = QGridLayout(f_g)
        fg.addWidget(QLabel("폰트 크기:"), 0, 0)
        self.font_spin = QSpinBox(); self.font_spin.setRange(6, 24); self.font_spin.setValue(10)
        fg.addWidget(self.font_spin, 0, 1)
        self.show_title = QCheckBox("제목 표시 (R²/RMSE/MAE)"); self.show_title.setChecked(True)
        fg.addWidget(self.show_title, 1, 0, 1, 2)
        fg.setColumnStretch(1, 1)
        lv.addWidget(f_g)

        self.run_btn = QPushButton("▶  평가 실행")
        self.run_btn.clicked.connect(self._run_eval)
        lv.addWidget(self.run_btn)

        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(80)
        lv.addWidget(self.log_box)
        lv.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0,0,0,0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # 우측
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(14,14,14,14); rv.setSpacing(8)
        rv.addWidget(QLabel("회귀 평가 결과"))

        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(["지표", "값"])
        hh = self.result_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setMaximumHeight(200)
        rv.addWidget(self.result_table)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
            from matplotlib.figure import Figure
            self._Canvas = Canvas; self._Figure = Figure
            self.canvas_container = QVBoxLayout()
            cw = QWidget(); cw.setLayout(self.canvas_container)
            rv.addWidget(cw, 1)
            self.canvas = None; self.has_mpl = True
        except ImportError:
            rv.addWidget(QLabel("matplotlib 없음")); self.has_mpl = False

        splitter.addWidget(right_w)
        splitter.setSizes([380, 720])
        outer.addWidget(splitter)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "모델", "", "PyTorch (*.pth *.pt)")
        if path:
            self.model_path.setText(path)
            auto = path.replace(".pth","_config.json").replace(".pt","_config.json")
            if os.path.isfile(auto):
                self.config_path.setText(auto)
                self.log_box.append(f"  ✔ config 자동 로드: {os.path.basename(auto)}")

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "config", "", "JSON (*.json)")
        if path: self.config_path.setText(path)

    def _browse_dir(self, edit):
        path = QFileDialog.getExistingDirectory(self, "폴더")
        if path: edit.setText(path)

    def _sync_to_infer(self):
        if self._infer_tab is None: return
        self._infer_tab.set_model_path(self.model_path.text().strip())
        self._infer_tab.set_config_path(self.config_path.text().strip())
        self._infer_tab.set_img_dir(self.eval_img.text().strip())
        self._infer_tab.set_img_ext(self.img_ext.currentText())
        self.state["lbl_dir"] = self.eval_lbl.text().strip()
        self.state["lbl_ext"] = self.lbl_ext.currentText()

    def _run_eval(self):
        mp = self.model_path.text().strip()
        cp = self.config_path.text().strip()
        img = self.eval_img.text().strip()
        lbl = self.eval_lbl.text().strip()
        if not mp or not cp or not img or not lbl:
            self.log_box.append("[오류] 모든 경로를 설정해주세요"); return
        self.run_btn.setEnabled(False); self.log_box.clear()
        config = {
            "model_path": mp, "config_path": cp,
            "img_dir": img, "lbl_dir": lbl,
            "img_ext": self.img_ext.currentText(),
            "lbl_ext": self.lbl_ext.currentText(),
        }
        self.worker = EvalWorker(config)
        self.worker.result_ready.connect(self._on_result)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_result(self, metrics, preds, targets):
        self._last_preds = preds
        self._last_targets = targets

        # 테이블
        self.result_table.setRowCount(0)
        for k, v in metrics.items():
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QTableWidgetItem(k))
            if isinstance(v, int):
                self.result_table.setItem(row, 1, QTableWidgetItem(f"{v:,}"))
            else:
                self.result_table.setItem(row, 1, QTableWidgetItem(f"{v:.5f}"))

        # 메인 산점도 표시
        self._draw_scatter(preds, targets, metrics)

        # 저장
        save_dir = self.save_dir.text().strip()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._save_plots(save_dir, preds, targets, metrics)

    def _draw_scatter(self, preds, targets, metrics):
        """농도 레벨별 평균±std errorbar 메인 plot (image 3 스타일)"""
        if not self.has_mpl: return
        import numpy as np
        fs = self.font_spin.value()
        show_title = self.show_title.isChecked() if hasattr(self, 'show_title') else True

        if self.canvas is not None:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None); self.canvas.deleteLater(); self.canvas = None

        # 농도 레벨별 grouping
        unique_t = np.unique(np.round(targets, 4))
        if len(unique_t) > 30:
            # 너무 많으면 binning
            bins = np.linspace(targets.min(), targets.max(), 12)
            bin_idx = np.digitize(targets, bins)
            groups = [(bins[i-1], preds[bin_idx==i]) for i in range(1, len(bins)+1) if (bin_idx==i).sum() > 0]
        else:
            groups = [(t, preds[np.abs(targets - t) < 1e-4]) for t in unique_t]
        xs    = [g[0] for g in groups]
        means = [g[1].mean() for g in groups]
        stds  = [g[1].std()  for g in groups]

        fig = self._Figure(figsize=(6, 5), facecolor='white')
        ax = fig.add_subplot(111)
        # Ideal y=x 선
        lo = min(min(xs), min(means)) - 0.05
        hi = max(max(xs), max(means)) + 0.05
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Ideal (y=x)')
        # Errorbar
        ax.errorbar(xs, means, yerr=stds, fmt='o-',
                    color='#1e40af', ecolor='#3b82f6',
                    elinewidth=2, capsize=5, capthick=1.5,
                    markersize=7, lw=2, label='Predicted mean ± std')
        ax.set_xlabel("Actual Concentration", fontsize=fs)
        ax.set_ylabel("Predicted Concentration", fontsize=fs)
        if show_title:
            ax.set_title(
                f"R²={metrics['R²']:.4f}   MAE={metrics['MAE']:.4f}   RMSE={metrics['RMSE']:.4f}",
                fontsize=fs+1
            )
        ax.legend(fontsize=fs-1); ax.grid(alpha=0.3)
        ax.tick_params(labelsize=fs-1)
        fig.tight_layout()
        self.canvas = self._Canvas(fig)
        self.canvas_container.addWidget(self.canvas)
        self.canvas.draw()

    def _save_plots(self, save_dir, preds, targets, metrics):
        import numpy as np
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            for font in ["Malgun Gothic","AppleGothic","NanumGothic","DejaVu Sans"]:
                try:
                    matplotlib.rcParams['font.family'] = font
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    break
                except Exception: continue
        except ImportError:
            return
        fs = self.font_spin.value()
        show_title = self.show_title.isChecked() if hasattr(self, 'show_title') else True
        n = len(preds)
        idx = np.random.choice(n, min(50000, n), replace=False)
        ps_s = preds[idx]; ts_s = targets[idx]
        residuals = preds - targets

        # ① Scatter (errorbar 스타일로 통일)
        if self.plot_checks["Scatter (Actual vs Predicted)"].isChecked():
            unique_t = np.unique(np.round(targets, 4))
            if len(unique_t) > 30:
                bins = np.linspace(targets.min(), targets.max(), 12)
                bin_idx = np.digitize(targets, bins)
                groups = [(bins[i-1], preds[bin_idx==i]) for i in range(1, len(bins)+1) if (bin_idx==i).sum() > 0]
            else:
                groups = [(t, preds[np.abs(targets - t) < 1e-4]) for t in unique_t]
            xs    = [g[0] for g in groups]
            means = [g[1].mean() for g in groups]
            stds  = [g[1].std()  for g in groups]
            fig, ax = plt.subplots(figsize=(6, 5))
            lo = min(min(xs), min(means)) - 0.05
            hi = max(max(xs), max(means)) + 0.05
            ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Ideal (y=x)')
            ax.errorbar(xs, means, yerr=stds, fmt='o-',
                        color='#1e40af', ecolor='#3b82f6',
                        elinewidth=2, capsize=5, capthick=1.5,
                        markersize=7, lw=2, label='Predicted mean ± std')
            ax.set_xlabel("Actual Concentration", fontsize=fs)
            ax.set_ylabel("Predicted Concentration", fontsize=fs)
            if show_title:
                ax.set_title(
                    f"R²={metrics['R²']:.4f}   MAE={metrics['MAE']:.4f}   RMSE={metrics['RMSE']:.4f}",
                    fontsize=fs+1
                )
            ax.legend(fontsize=fs-1); ax.grid(alpha=0.3)
            ax.tick_params(labelsize=fs-1)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "scatter.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.log_box.append("  💾 scatter.png")
            plt.close(fig)
            self.log_box.append("  💾 scatter.png")

        # ② Residual plot
        if self.plot_checks["Residual Plot"].isChecked():
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(ts_s, residuals[idx], s=4, alpha=0.3, color='#ef4444')
            ax.axhline(0, color='k', lw=1.5)
            ax.set_xlabel("Actual", fontsize=fs); ax.set_ylabel("Residual (Pred - Actual)", fontsize=fs)
            if show_title:
                ax.set_title("Residual Plot", fontsize=fs+1)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=fs-1)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "residual.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.log_box.append("  💾 residual.png")

        # ③ Histogram of residuals
        if self.plot_checks["Histogram of Residuals"].isChecked():
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(residuals, bins=60, color='#2563eb', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='k', lw=1.5)
            ax.set_xlabel("Residual", fontsize=fs); ax.set_ylabel("Count", fontsize=fs)
            if show_title:
                ax.set_title(f"Residual Distribution  (mean={residuals.mean():.4f}, std={residuals.std():.4f})", fontsize=fs+1)
            ax.grid(alpha=0.3); ax.tick_params(labelsize=fs-1)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "residual_hist.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.log_box.append("  💾 residual_hist.png")

        # ④ 농도 레벨별 평균 ± 표준편차
        if self.plot_checks["Predicted vs Actual per concentration level"].isChecked():
            unique_t = np.unique(np.round(targets, 4))
            if len(unique_t) > 30:
                bins = np.linspace(targets.min(), targets.max(), 12)
                bin_idx = np.digitize(targets, bins)
                groups = [(bins[i-1], preds[bin_idx==i]) for i in range(1, len(bins)+1) if (bin_idx==i).sum() > 0]
            else:
                groups = [(t, preds[np.abs(targets - t) < 1e-4]) for t in unique_t]
            xs = [g[0] for g in groups]
            means = [g[1].mean() for g in groups]
            stds  = [g[1].std()  for g in groups]
            fig, ax = plt.subplots(figsize=(6, 5))
            lo = min(min(xs), min(means)) - 0.05
            hi = max(max(xs), max(means)) + 0.05
            ax.plot([lo, hi], [lo, hi], 'r--', label='Ideal (y=x)')
            ax.errorbar(xs, means, yerr=stds, fmt='o-', capsize=5, color='#1e40af',
                        ecolor='#3b82f6', elinewidth=2, capthick=1.5, markersize=7, lw=2,
                        label='Predicted mean ± std')
            ax.set_xlabel("Actual Concentration", fontsize=fs)
            ax.set_ylabel("Predicted Concentration", fontsize=fs)
            if show_title:
                ax.set_title(f"R²={metrics['R²']:.4f}  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}",
                             fontsize=fs+1)
            ax.legend(fontsize=fs-1); ax.grid(alpha=0.3); ax.tick_params(labelsize=fs-1)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "per_level.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.log_box.append("  💾 per_level.png")

        # ⑤ Bland-Altman
        if self.plot_checks["Bland-Altman Plot"].isChecked():
            mean_pt = (preds + targets) / 2
            diff = preds - targets
            md = diff.mean(); sd = diff.std()
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(mean_pt[idx], diff[idx], s=4, alpha=0.3, color='#16a34a')
            ax.axhline(md, color='k', lw=1.5, label=f'Mean diff = {md:.4f}')
            ax.axhline(md + 1.96*sd, color='r', lw=1, ls='--', label=f'+1.96 SD')
            ax.axhline(md - 1.96*sd, color='r', lw=1, ls='--', label=f'-1.96 SD')
            ax.set_xlabel("Mean of Actual & Predicted", fontsize=fs)
            ax.set_ylabel("Predicted - Actual", fontsize=fs)
            if show_title:
                ax.set_title("Bland-Altman Plot", fontsize=fs+1)
            ax.legend(fontsize=fs-1); ax.grid(alpha=0.3); ax.tick_params(labelsize=fs-1)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "bland_altman.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.log_box.append("  💾 bland_altman.png")

    def _on_finished(self):
        self.run_btn.setEnabled(True)
        self.log_box.append("✔ 완료")

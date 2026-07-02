import os
import re
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QLineEdit,
    QScrollArea, QComboBox, QCheckBox,
    QSplitter, QSlider, QListWidget, QListWidgetItem,
    QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# ═══════════════════════════════════════════════════════════════
#  InferWorker
# ═══════════════════════════════════════════════════════════════
class InferWorker(QThread):
    progress_update = pyqtSignal(int, int)
    result_ready    = pyqtSignal(int, object, object, object)  # idx, vis, conc_map, gt
    log_msg         = pyqtSignal(str)
    finished        = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self._run_infer()
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

    def _build_model(self, mc):
        import torch.nn as nn
        mode = mc.get("mode", "patch")
        model_name = mc.get("model_name", "")
        in_ch = mc.get("in_channels", 6)
        encoder = mc.get("encoder", "resnet34")
        activation = mc.get("activation", "없음 (linear)")

        if mode == "patch":
            from tabs.train_tab import _make_simple_cnn, _make_1d_cnn, _make_mlp
            if model_name == "CNN-2D-Simple":
                return _make_simple_cnn(in_ch, mc.get("patch_size",9)), "patch"
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
            act_map = {"없음 (linear)":None,"sigmoid":"sigmoid","tanh":"tanh","ReLU":"identity"}
            cls = SMP_MAP.get(model_name, "Unet")
            model = getattr(smp, cls)(
                encoder_name=encoder, encoder_weights=None,
                in_channels=in_ch, classes=1,
                activation=act_map.get(activation, None),
            )
            return model, "dense"

    def _run_infer(self):
        import torch, numpy as np, glob, json
        cfg = self.config
        with open(cfg["config_path"], "r", encoding="utf-8") as f:
            mc = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_msg.emit(f"[추론 시작] 디바이스: {device}")

        model, kind = self._build_model(mc)
        state = torch.load(cfg["model_path"], map_location=device, weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=False)
        model = model.to(device); model.eval()
        self.log_msg.emit("  모델 로드 완료")

        bg      = mc.get("bg_value", 0)
        offset  = mc.get("label_offset", -1.0)
        divisor = mc.get("label_divisor", 10.0)
        if divisor == 0: divisor = 1.0

        img_dir  = cfg["img_dir"]
        img_ext  = cfg.get("img_ext","mat")
        lbl_dir  = cfg.get("lbl_dir","")
        lbl_ext  = cfg.get("lbl_ext","png")
        has_gt   = bool(lbl_dir and os.path.isdir(lbl_dir))
        out_dir  = cfg.get("out_dir", "./infer_results")
        os.makedirs(out_dir, exist_ok=True)

        img_files = cfg.get("img_files",[])
        if not img_files:
            img_files = sorted(glob.glob(os.path.join(img_dir, f"*.{img_ext}")))
        total = len(img_files)
        self.log_msg.emit(f"  추론 대상: {total}장")

        mode  = mc.get("mode","patch")
        in_ch = mc.get("in_channels", 6)
        norm_type = mc.get("norm_type")
        if norm_type is None:
            norm_type = "snv" if mc.get("use_snv", False) else "none"
        mask_mode = cfg.get("mask_mode", "auto")
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

        for idx, ip in enumerate(img_files):
            if self._stop: break
            img = self._load_image(ip)
            gt  = None
            if has_gt:
                base = os.path.splitext(os.path.basename(ip))[0]
                # 여러 패턴 시도:
                # 1) 동일한 파일명 (Data0001.mat → Data0001.png)
                # 2) "Data" → "Label" 치환 (Data0001 → Label0001)
                # 3) 숫자 부분만 매칭 (0001 → Label0001)
                candidates = [
                    base + f".{lbl_ext}",
                    base.replace("Data", "Label").replace("data", "label") + f".{lbl_ext}",
                ]
                m = re.search(r'\d+', base)
                if m:
                    candidates.append(f"Label{m.group()}.{lbl_ext}")
                    candidates.append(f"label{m.group()}.{lbl_ext}")

                gt_path = None
                for c in candidates:
                    p = os.path.join(lbl_dir, c)
                    if os.path.isfile(p):
                        gt_path = p; break

                if gt_path is None:
                    if idx == 0:
                        self.log_msg.emit(f"  [경고] {base} 의 GT 라벨 못 찾음 (시도: {candidates[:2]})")
                else:
                    if idx == 0:
                        self.log_msg.emit(f"  GT 매칭 패턴: {os.path.basename(gt_path)}")
                    gt_raw = self._load_label(gt_path)
                    gt = (gt_raw.astype(np.float32) + offset) / divisor
                    gt[gt_raw == bg] = np.nan

            # (H, W, C) 표준화
            if img.ndim == 2:
                img_hwc = img[..., None]
            elif img.shape[0] < img.shape[-1]:
                img_hwc = np.transpose(img, (1, 2, 0))
            else:
                img_hwc = img

            # 정규화
            img_hwc = _normalize(img_hwc, norm_type).astype(np.float32)

            # ROI 마스크
            if mask_mode == "use_gt" and gt is not None:
                roi_mask = ~np.isnan(gt)
            elif mask_mode == "threshold":
                ref = img_hwc[:, :, 0] if img_hwc.ndim == 3 else img_hwc
                th = cfg.get("mask_threshold", float(ref.mean()))
                roi_mask = ref > th
            else:
                roi_mask = np.ones(img_hwc.shape[:2], dtype=bool)

            with torch.no_grad():
                if mode == "patch":
                    patch_size = mc.get("patch_size", 9)
                    half = patch_size // 2
                    img_p = np.pad(img_hwc, ((half,half),(half,half),(0,0)), mode='edge')
                    H, W = img_hwc.shape[:2]
                    ys, xs = np.where(roi_mask)
                    conc_map = np.full((H, W), np.nan, dtype=np.float32)
                    if len(ys) > 0:
                        BATCH = 2048
                        for i in range(0, len(ys), BATCH):
                            ybs = ys[i:i+BATCH]; xbs = xs[i:i+BATCH]
                            patches = np.stack([
                                img_p[y:y+patch_size, x:x+patch_size, :]
                                for y, x in zip(ybs, xbs)
                            ])  # (B, P, P, C)
                            X = torch.from_numpy(patches).float().permute(0, 3, 1, 2).to(device)
                            out = model(X)
                            if out.dim() > 1: out = out.squeeze(-1)
                            conc_map[ybs, xbs] = out.cpu().numpy()
                else:
                    from utils.pad_utils import pad_tensor, unpad_tensor, get_pad_multiple
                    multiple = mc.get("pad_multiple") or get_pad_multiple(mc.get("model_name","UNet"))
                    if multiple < 32: multiple = 32
                    t = torch.from_numpy(img_hwc).float().permute(2, 0, 1)
                    t = t.unsqueeze(0).to(device)
                    t_pad, oh, ow = pad_tensor(t, multiple=multiple)
                    out = model(t_pad)
                    conc_map = unpad_tensor(out, oh, ow).squeeze().cpu().numpy()
                    conc_map[~roi_mask] = np.nan

            # 시각화용 원본 (대표 3채널) — SNV 전 데이터 사용
            if img.ndim == 3:
                arr = img if img.shape[-1] < img.shape[0] else np.transpose(img, (1, 2, 0))
                ch = arr.shape[-1]
                if ch >= 3:
                    b_r, b_g, b_b = ch-1, ch//2, 0
                    vis_raw = np.stack([arr[:,:,b_r], arr[:,:,b_g], arr[:,:,b_b]], axis=-1)
                else:
                    vis_raw = np.stack([arr[:,:,0]]*3, axis=-1)
            else:
                vis_raw = np.stack([img]*3, axis=-1)
            vmin, vmax = vis_raw.min(), vis_raw.max()
            if vmax > vmin:
                vis = ((vis_raw - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            else:
                vis = np.zeros_like(vis_raw, dtype=np.uint8)

            # 저장
            fname = os.path.splitext(os.path.basename(ip))[0]
            if cfg.get("save_conc_map", True):
                np.save(os.path.join(out_dir, f"{fname}_conc.npy"), conc_map)
            if cfg.get("save_color_map", True):
                self._save_overlay(vis, conc_map, gt,
                                   os.path.join(out_dir, f"{fname}_overlay.png"),
                                   cfg.get("cmap","viridis"),
                                   cfg.get("vmin",0.0), cfg.get("vmax",1.0),
                                   cfg.get("show_title_save", True),
                                   cfg.get("show_colorbar_save", True),
                                   fname=os.path.basename(ip))

            self.progress_update.emit(idx+1, total)
            self.result_ready.emit(idx, vis, conc_map, gt)

        self.log_msg.emit(f"✔ 추론 완료 → {out_dir}")

    def _save_overlay(self, vis, conc_map, gt, path, cmap_name, vmin, vmax,
                      show_title=True, show_cbar=True, fname=""):
        """Grayscale | (GT Overlay) | Predicted Overlay 형식으로 저장"""
        import numpy as np
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            has_gt = gt is not None
            n_panels = 3 if has_gt else 2

            # 회색조 배경
            if vis.ndim == 3:
                gray = vis.mean(axis=2) if vis.shape[-1] >= 3 else vis[..., 0]
            else:
                gray = vis
            gray = gray.astype(np.float32)
            gmin, gmax = gray.min(), gray.max()
            if gmax > gmin:
                gray = (gray - gmin) / (gmax - gmin)

            cmap_obj = cm.get_cmap(cmap_name).copy()
            cmap_obj.set_bad(color=(0,0,0,0), alpha=0)

            fig, axes = plt.subplots(1, n_panels, figsize=(n_panels*4.5, 4.5))
            if n_panels == 1: axes = [axes]

            # Panel 1: Grayscale
            axes[0].imshow(gray, cmap='gray', vmin=0, vmax=1)
            axes[0].axis('off')
            if show_title:
                axes[0].set_title(f"CH1 Grayscale  ({fname})" if fname else "Grayscale", fontsize=10)

            if has_gt:
                axes[1].imshow(gray, cmap='gray', vmin=0, vmax=1)
                im_gt = axes[1].imshow(gt, cmap=cmap_obj, vmin=vmin, vmax=vmax, alpha=0.75)
                axes[1].axis('off')
                if show_title:
                    axes[1].set_title(f"Ground Truth  ({fname})" if fname else "Ground Truth", fontsize=10)
                if show_cbar:
                    fig.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

                axes[2].imshow(gray, cmap='gray', vmin=0, vmax=1)
                im_pr = axes[2].imshow(conc_map, cmap=cmap_obj, vmin=vmin, vmax=vmax, alpha=0.75)
                axes[2].axis('off')
                if show_title:
                    axes[2].set_title(f"Predicted  ({fname})" if fname else "Predicted", fontsize=10)
                if show_cbar:
                    fig.colorbar(im_pr, ax=axes[2], fraction=0.046, pad=0.04)
            else:
                axes[1].imshow(gray, cmap='gray', vmin=0, vmax=1)
                im_pr = axes[1].imshow(conc_map, cmap=cmap_obj, vmin=vmin, vmax=vmax, alpha=0.75)
                axes[1].axis('off')
                if show_title:
                    axes[1].set_title(f"Predicted  ({fname})" if fname else "Predicted", fontsize=10)
                if show_cbar:
                    fig.colorbar(im_pr, ax=axes[1], fraction=0.046, pad=0.04)

            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"overlay 저장 실패: {e}")

    def _save_colormap(self, conc_map, path, cmap_name, vmin, vmax, show_title=False, show_cbar=True):
        """단일 농도맵 저장 (호환용, 사용 안 함)"""
        import numpy as np
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            fig, ax = plt.subplots(figsize=(6, 5))
            cmap_obj = cm.get_cmap(cmap_name).copy()
            cmap_obj.set_bad(color='white', alpha=0)
            im = ax.imshow(conc_map, cmap=cmap_obj, vmin=vmin, vmax=vmax)
            if show_cbar:
                plt.colorbar(im, ax=ax)
            if show_title:
                ax.set_title("Predicted Concentration")
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
#  InferTab UI
# ═══════════════════════════════════════════════════════════════
class InferTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self.worker = None
        self.preview_imgs = []
        self.preview_idx = 0
        self._zoom = 1.0
        self._build()

    def set_model_path(self, path):  self.model_path.setText(path)
    def set_config_path(self, path):
        if hasattr(self, 'config_path_edit'): self.config_path_edit.setText(path)
    def set_img_dir(self, path):     self.input_dir.setText(path)
    def set_img_ext(self, ext):
        idx = self.ext_combo.findText(ext)
        if idx >= 0: self.ext_combo.setCurrentIndex(idx)

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 좌측 ───────────────────────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        lv = QVBoxLayout(inner); lv.setSpacing(10); lv.setContentsMargins(14,14,14,14)

        m_g = QGroupBox("모델 (④ 평가탭에서 자동 반영)")
        mg = QGridLayout(m_g)
        mg.addWidget(QLabel("모델 파일:"), 0, 0)
        self.model_path = QLineEdit()
        mg.addWidget(self.model_path, 0, 1)
        btn_m = QPushButton("찾기"); btn_m.setObjectName("secondary")
        btn_m.clicked.connect(self._browse_model)
        mg.addWidget(btn_m, 0, 2)
        mg.addWidget(QLabel("config.json:"), 1, 0)
        self.config_path_edit = QLineEdit()
        mg.addWidget(self.config_path_edit, 1, 1)
        btn_c = QPushButton("찾기"); btn_c.setObjectName("secondary")
        btn_c.clicked.connect(self._browse_config)
        mg.addWidget(btn_c, 1, 2)
        mg.setColumnStretch(1, 1)
        lv.addWidget(m_g)

        in_g = QGroupBox("입력")
        ig = QGridLayout(in_g)
        ig.addWidget(QLabel("이미지 폴더:"), 0, 0)
        self.input_dir = QLineEdit()
        ig.addWidget(self.input_dir, 0, 1)
        btn_i = QPushButton("찾기"); btn_i.setObjectName("secondary")
        btn_i.clicked.connect(lambda: self._browse_dir(self.input_dir))
        ig.addWidget(btn_i, 0, 2)
        ig.addWidget(QLabel("확장자:"), 1, 0)
        self.ext_combo = QComboBox(); self.ext_combo.addItems(["mat","npy","png","jpg","tiff"])
        ig.addWidget(self.ext_combo, 1, 1)

        ig.addWidget(QLabel("GT 라벨 폴더(선택):"), 2, 0)
        self.lbl_dir_edit = QLineEdit()
        ig.addWidget(self.lbl_dir_edit, 2, 1)
        btn_l = QPushButton("찾기"); btn_l.setObjectName("secondary")
        btn_l.clicked.connect(lambda: self._browse_dir(self.lbl_dir_edit))
        ig.addWidget(btn_l, 2, 2)
        ig.setColumnStretch(1, 1)
        lv.addWidget(in_g)

        mask_g = QGroupBox("ROI 마스킹")
        mskg = QGridLayout(mask_g)
        mskg.addWidget(QLabel("마스크 모드:"), 0, 0)
        self.mask_mode = QComboBox()
        self.mask_mode.addItems(["전체 (마스킹 없음)", "GT 라벨 기반", "Threshold 기반"])
        mskg.addWidget(self.mask_mode, 0, 1)
        mskg.addWidget(QLabel("Threshold:"), 1, 0)
        self.mask_thresh = QDoubleSpinBox()
        self.mask_thresh.setRange(0, 65535); self.mask_thresh.setDecimals(2); self.mask_thresh.setValue(100.0)
        mskg.addWidget(self.mask_thresh, 1, 1)
        mskg.setColumnStretch(1, 1)
        lv.addWidget(mask_g)

        out_g = QGroupBox("출력")
        og = QGridLayout(out_g)
        og.addWidget(QLabel("저장 폴더:"), 0, 0)
        self.output_dir = QLineEdit(); self.output_dir.setPlaceholderText("./infer_results")
        og.addWidget(self.output_dir, 0, 1)
        btn_o = QPushButton("찾기"); btn_o.setObjectName("secondary")
        btn_o.clicked.connect(lambda: self._browse_dir(self.output_dir))
        og.addWidget(btn_o, 0, 2)
        self.save_conc = QCheckBox("농도맵 (.npy) 저장"); self.save_conc.setChecked(True)
        self.save_color = QCheckBox("컬러맵 (.png) 저장"); self.save_color.setChecked(True)
        og.addWidget(self.save_conc, 1, 0, 1, 3)
        og.addWidget(self.save_color, 2, 0, 1, 3)

        og.addWidget(QLabel("컬러맵:"), 3, 0)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["viridis","plasma","inferno","jet","hot","coolwarm","RdYlBu_r","turbo"])
        og.addWidget(self.cmap_combo, 3, 1)

        og.addWidget(QLabel("값 범위:"), 4, 0)
        rh = QHBoxLayout()
        self.vmin = QDoubleSpinBox(); self.vmin.setRange(-100,100); self.vmin.setDecimals(3); self.vmin.setValue(0.0)
        self.vmax = QDoubleSpinBox(); self.vmax.setRange(-100,100); self.vmax.setDecimals(3); self.vmax.setValue(1.0)
        rh.addWidget(self.vmin); rh.addWidget(QLabel("~")); rh.addWidget(self.vmax); rh.addStretch()
        og.addLayout(rh, 4, 1)
        og.setColumnStretch(1, 1)
        lv.addWidget(out_g)

        # ── Figure 옵션 (title / colorbar 토글) ──
        fig_g = QGroupBox("Figure 옵션 (논문용)")
        figg = QGridLayout(fig_g)
        self.show_title = QCheckBox("Title 표시 (Original / Predicted / GT)"); self.show_title.setChecked(True)
        self.show_colorbar = QCheckBox("Colorbar 표시"); self.show_colorbar.setChecked(True)
        figg.addWidget(self.show_title, 0, 0)
        figg.addWidget(self.show_colorbar, 1, 0)
        # 미리보기 변경 시 즉시 반영
        self.show_title.toggled.connect(self._refresh_preview)
        self.show_colorbar.toggled.connect(self._refresh_preview)
        lv.addWidget(fig_g)

        self.run_btn = QPushButton("▶  배치 추론 실행")
        self.run_btn.clicked.connect(self._run_infer)
        lv.addWidget(self.run_btn)
        self.progress = QProgressBar(); self.progress.setValue(0)
        lv.addWidget(self.progress)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(80)
        lv.addWidget(self.log_box)
        lv.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0,0,0,0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 우측 미리보기 ──────────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(14,14,14,14); rv.setSpacing(6)

        title_h = QHBoxLayout()
        title_h.addWidget(QLabel("농도맵 미리보기"))
        title_h.addStretch()
        title_h.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 300); self.zoom_slider.setValue(100); self.zoom_slider.setFixedWidth(120)
        self.zoom_pct_lbl = QLabel("100%")
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        title_h.addWidget(self.zoom_slider); title_h.addWidget(self.zoom_pct_lbl)
        rv.addLayout(title_h)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
            from matplotlib.figure import Figure
            self._Canvas = Canvas; self._Figure = Figure
            self.canvas_container = QVBoxLayout()
            cw = QWidget(); cw.setLayout(self.canvas_container)
            rv.addWidget(cw, 1)
            self.canvas = None; self.has_mpl = True
        except ImportError:
            self.has_mpl = False; rv.addWidget(QLabel("matplotlib 없음"))

        # 통계 테이블
        rv.addWidget(QLabel("ROI 통계"))
        self.stat_table = QTableWidget(0, 2)
        self.stat_table.setHorizontalHeaderLabels(["지표", "값"])
        self.stat_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.stat_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.stat_table.verticalHeader().setVisible(False)
        self.stat_table.setMaximumHeight(180)
        rv.addWidget(self.stat_table)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 이전"); self.prev_btn.setObjectName("secondary")
        self.next_btn = QPushButton("다음 ▶"); self.next_btn.setObjectName("secondary")
        self.img_idx_lbl = QLabel("0 / 0")
        self.prev_btn.clicked.connect(self._prev); self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.prev_btn); nav.addWidget(self.img_idx_lbl); nav.addWidget(self.next_btn)
        rv.addLayout(nav)

        splitter.addWidget(right_w)
        splitter.setSizes([380, 720])
        outer.addWidget(splitter)

    def _on_zoom_changed(self, v):
        self._zoom = v / 100.0
        self.zoom_pct_lbl.setText(f"{v}%")
        self._show()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "모델", "", "PyTorch (*.pth *.pt)")
        if path:
            self.model_path.setText(path)
            auto = path.replace(".pth","_config.json").replace(".pt","_config.json")
            if os.path.isfile(auto):
                self.config_path_edit.setText(auto)

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "config", "", "JSON (*.json)")
        if path: self.config_path_edit.setText(path)

    def _browse_dir(self, edit):
        path = QFileDialog.getExistingDirectory(self, "폴더")
        if path: edit.setText(path)

    def _run_infer(self):
        mp = self.model_path.text().strip()
        cp = self.config_path_edit.text().strip()
        img = self.input_dir.text().strip()
        if not mp or not cp or not img:
            self.log_box.append("[오류] 모델/config/이미지 경로를 모두 설정하세요"); return

        mask_mode_map = {0:"auto", 1:"use_gt", 2:"threshold"}
        config = {
            "model_path":   mp,
            "config_path":  cp,
            "img_dir":      img,
            "img_ext":      self.ext_combo.currentText(),
            "lbl_dir":      self.lbl_dir_edit.text().strip(),
            "lbl_ext":      "png",
            "out_dir":      self.output_dir.text().strip() or "./infer_results",
            "save_conc_map":  self.save_conc.isChecked(),
            "save_color_map": self.save_color.isChecked(),
            "cmap":         self.cmap_combo.currentText(),
            "vmin":         self.vmin.value(),
            "vmax":         self.vmax.value(),
            "mask_mode":    mask_mode_map[self.mask_mode.currentIndex()],
            "mask_threshold": self.mask_thresh.value(),
            "show_title_save":    self.show_title.isChecked(),
            "show_colorbar_save": self.show_colorbar.isChecked(),
        }
        self.preview_imgs.clear(); self.preview_idx = 0
        self.run_btn.setEnabled(False); self.log_box.clear(); self.progress.setValue(0)

        self.worker = InferWorker(config)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_result)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, cur, total):
        self.progress.setRange(0, total); self.progress.setValue(cur)

    def _on_result(self, idx, vis, conc_map, gt):
        self.preview_imgs.append((vis, conc_map, gt))
        if idx == 0:
            self.preview_idx = 0; self._show()
        self.img_idx_lbl.setText(f"{self.preview_idx+1} / {len(self.preview_imgs)}")

    def _refresh_preview(self):
        """Title/Colorbar 토글 변경 시 미리보기 즉시 갱신"""
        if self.preview_imgs:
            self._show()

    def _show(self):
        if not self.has_mpl or not self.preview_imgs: return
        import numpy as np
        vis, conc_map, gt = self.preview_imgs[self.preview_idx]
        zoom = self._zoom
        show_title = self.show_title.isChecked()
        show_cbar  = self.show_colorbar.isChecked()

        if self.canvas is not None:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None); self.canvas.deleteLater(); self.canvas = None

        has_gt = gt is not None
        n_panels = 3 if has_gt else 2
        base_w = n_panels * 4.0 * zoom
        base_h = 4.0 * zoom

        # 파일명 (제목용)
        fname = ""
        try:
            cfg = self.worker.config if hasattr(self, 'worker') and self.worker else {}
            img_files = cfg.get("img_files", [])
            if not img_files:
                import glob, os
                img_files = sorted(glob.glob(os.path.join(cfg.get("img_dir",""),
                                                          f"*.{cfg.get('img_ext','mat')}")))
            if self.preview_idx < len(img_files):
                import os as _os
                fname = _os.path.basename(img_files[self.preview_idx])
        except Exception:
            pass

        # ── 회색조 배경 만들기 ──
        if vis.ndim == 3:
            # 첫 채널 사용 (또는 luminance 변환)
            gray = vis.mean(axis=2) if vis.shape[-1] >= 3 else vis[..., 0]
        else:
            gray = vis
        gray = gray.astype(np.float32)
        gmin, gmax = gray.min(), gray.max()
        if gmax > gmin:
            gray = (gray - gmin) / (gmax - gmin)

        # 컬러맵 (NaN→투명)
        import matplotlib.cm as cm
        cmap_obj = cm.get_cmap(self.cmap_combo.currentText()).copy()
        cmap_obj.set_bad(color=(0,0,0,0), alpha=0)

        fig = self._Figure(figsize=(base_w, base_h), facecolor='white')
        axes = [fig.add_subplot(1, n_panels, i+1) for i in range(n_panels)]

        # Panel 1: Grayscale
        axes[0].imshow(gray, cmap='gray', vmin=0, vmax=1)
        axes[0].axis('off')
        if show_title:
            axes[0].set_title(f"CH1 Grayscale  ({fname})" if fname else "Grayscale", fontsize=10)

        # Panel 2: GT Overlay (있을 때)
        if has_gt:
            axes[1].imshow(gray, cmap='gray', vmin=0, vmax=1)
            im_gt = axes[1].imshow(gt, cmap=cmap_obj,
                                   vmin=self.vmin.value(), vmax=self.vmax.value(),
                                   alpha=0.75)
            axes[1].axis('off')
            if show_title:
                axes[1].set_title(f"Ground Truth  ({fname})" if fname else "Ground Truth", fontsize=10)
            if show_cbar:
                fig.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

            # Panel 3: Predicted Overlay
            axes[2].imshow(gray, cmap='gray', vmin=0, vmax=1)
            im_pr = axes[2].imshow(conc_map, cmap=cmap_obj,
                                   vmin=self.vmin.value(), vmax=self.vmax.value(),
                                   alpha=0.75)
            axes[2].axis('off')
            if show_title:
                axes[2].set_title(f"Predicted Concentration  ({fname})" if fname else "Predicted", fontsize=10)
            if show_cbar:
                fig.colorbar(im_pr, ax=axes[2], fraction=0.046, pad=0.04)
        else:
            # GT 없을 때: Panel 2가 예측
            axes[1].imshow(gray, cmap='gray', vmin=0, vmax=1)
            im_pr = axes[1].imshow(conc_map, cmap=cmap_obj,
                                   vmin=self.vmin.value(), vmax=self.vmax.value(),
                                   alpha=0.75)
            axes[1].axis('off')
            if show_title:
                axes[1].set_title(f"Predicted Concentration  ({fname})" if fname else "Predicted", fontsize=10)
            if show_cbar:
                fig.colorbar(im_pr, ax=axes[1], fraction=0.046, pad=0.04)

        fig.tight_layout(pad=0.5)
        self.canvas = self._Canvas(fig)
        self.canvas_container.addWidget(self.canvas)
        self.canvas.draw()

        # 통계 업데이트
        self._update_stats(conc_map, gt)

    def _update_stats(self, conc_map, gt):
        import numpy as np
        self.stat_table.setRowCount(0)
        valid = conc_map[~np.isnan(conc_map)]
        if len(valid) == 0: return
        stats = [
            ("픽셀 수",    f"{len(valid):,}"),
            ("평균",       f"{valid.mean():.5f}"),
            ("표준편차",   f"{valid.std():.5f}"),
            ("최솟값",     f"{valid.min():.5f}"),
            ("최댓값",     f"{valid.max():.5f}"),
            ("중앙값",     f"{np.median(valid):.5f}"),
        ]
        if gt is not None:
            valid_gt = gt[~np.isnan(gt)]
            if len(valid_gt) > 0:
                # 두 마스크의 교집합
                joint = ~np.isnan(conc_map) & ~np.isnan(gt)
                if joint.sum() > 0:
                    p = conc_map[joint]; t = gt[joint]
                    mae = float(np.abs(p - t).mean())
                    rmse = float(np.sqrt(((p - t)**2).mean()))
                    ss_res = ((t - p)**2).sum()
                    ss_tot = ((t - t.mean())**2).sum() + 1e-9
                    r2 = float(1 - ss_res / ss_tot)
                    stats.extend([
                        ("─" * 8 + " GT 비교 " + "─" * 8, ""),
                        ("R²",   f"{r2:.5f}"),
                        ("RMSE", f"{rmse:.5f}"),
                        ("MAE",  f"{mae:.5f}"),
                    ])
        for k, v in stats:
            row = self.stat_table.rowCount()
            self.stat_table.insertRow(row)
            self.stat_table.setItem(row, 0, QTableWidgetItem(k))
            self.stat_table.setItem(row, 1, QTableWidgetItem(v))

    def _prev(self):
        if self.preview_idx > 0:
            self.preview_idx -= 1; self._show()
            self.img_idx_lbl.setText(f"{self.preview_idx+1} / {len(self.preview_imgs)}")

    def _next(self):
        if self.preview_idx < len(self.preview_imgs) - 1:
            self.preview_idx += 1; self._show()
            self.img_idx_lbl.setText(f"{self.preview_idx+1} / {len(self.preview_imgs)}")

    def _on_finished(self):
        self.run_btn.setEnabled(True)

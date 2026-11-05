"""
Train Tab: self-supervised denoising training (N2V / N2F / N2N2).
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QProgressBar, QMessageBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import build_model
from dataset import build_dataset
from n2v_core import n2v_loss, n2n2_loss


METHOD_INFO = {
    "N2V":  "Noise2Void (blind-spot mask)",
    "N2F":  "Noise2Fast (checkerboard split)",
    "N2N2": "Neighbor2Neighbor (random subsample)",
}


class TrainWorker(QThread):
    progress = pyqtSignal(int, int, float)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            cfg = self.cfg
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log.emit(f"Device: {device}")
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = True

            method = cfg["method"]   # 'N2V' / 'N2F' / 'N2N2'
            mode = cfg["mode"]       # '2D' / '3D'

            self.log.emit(f"Method: {method} ({METHOD_INFO[method]})")
            self.log.emit(f"Mode: {mode}")

            # Dataset
            ds_kwargs = {
                "mask_ratio": cfg["mask_ratio"],
                "spectral_chunk": cfg["spectral_chunk"],
                "bands_per_sample": cfg["bands_per_sample"],
            }
            dataset = build_dataset(method, mode, cfg["patch_dir"], **ds_kwargs)
            self.log.emit(f"Dataset size: {len(dataset)}")

            n_workers = min(4, max(1, (os.cpu_count() or 1) - 1))
            loader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                num_workers=n_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=(n_workers > 0),
                drop_last=True,
            )

            model = build_model(mode).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt, step_size=max(1, cfg["epochs"] // 3), gamma=0.5
            )

            # Mixed precision
            use_amp = (device.type == "cuda")
            try:
                scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
                amp_autocast = lambda: torch.amp.autocast('cuda', enabled=use_amp)
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
                amp_autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)

            best_loss = float("inf")
            os.makedirs(os.path.dirname(cfg["save_path"]) or ".", exist_ok=True)

            for ep in range(1, cfg["epochs"] + 1):
                if self._stop:
                    self.log.emit("[Stopped by user]")
                    break
                model.train()
                ep_loss_t = torch.zeros(1, device=device)
                n_batches = 0

                for batch in loader:
                    if self._stop:
                        break

                    opt.zero_grad(set_to_none=True)

                    # ───── Method별 forward + loss ─────
                    if method == "N2V":
                        masked, target, mask = batch
                        masked = masked.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        mask = mask.to(device, non_blocking=True)
                        with amp_autocast():
                            pred = model(masked)
                            loss = n2v_loss(pred, target, mask)

                    elif method == "N2F":
                        sub_in, sub_target = batch
                        sub_in = sub_in.to(device, non_blocking=True)
                        sub_target = sub_target.to(device, non_blocking=True)
                        with amp_autocast():
                            pred = model(sub_in)
                            loss = ((pred - sub_target) ** 2).mean()

                    elif method == "N2N2":
                        g1, g2 = batch
                        g1 = g1.to(device, non_blocking=True)
                        g2 = g2.to(device, non_blocking=True)
                        with amp_autocast():
                            fg1 = model(g1)
                            fg2 = model(g2)
                            loss = n2n2_loss(fg1, g2, g1=g1, fg2=fg2,
                                             gamma=cfg.get("gamma", 1.0))
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()

                    ep_loss_t += loss.detach()
                    n_batches += 1

                avg = (ep_loss_t / max(1, n_batches)).item()
                scheduler.step()
                self.progress.emit(ep, cfg["epochs"], avg)
                self.log.emit(f"Epoch {ep}/{cfg['epochs']}  loss={avg:.6f}")

                if avg < best_loss:
                    best_loss = avg
                    torch.save({
                        "state_dict": model.state_dict(),
                        "method": method,
                        "mode": mode,
                        "spectral_chunk": cfg.get("spectral_chunk"),
                        "patch_size": cfg["patch_size"],
                    }, cfg["save_path"])

                    cfg_path = cfg["save_path"].replace(".pth", "_config.json")
                    with open(cfg_path, "w") as f:
                        json.dump({
                            "method": method,
                            "mode": mode,
                            "spectral_chunk": cfg.get("spectral_chunk"),
                            "patch_size": cfg["patch_size"],
                            "best_loss": best_loss,
                        }, f, indent=2)

            self.finished_ok.emit(cfg["save_path"])
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class TrainTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.predict_tab = None
        self.worker = None
        self.losses = []
        self._build_ui()
        self._update_param_visibility()

    def set_predict_tab(self, p):
        self.predict_tab = p

    def notify_data_ready(self):
        pd = self.state.get("patch_dir")
        if pd:
            self.lbl_patch.setText(pd)
            nb = self.state.get("n_bands", 0)
            self.lbl_bands.setText(f"Bands: {nb}")
            if nb and nb < self.sp_chunk.value():
                self.sp_chunk.setValue(min(64, nb))

    def _build_ui(self):
        layout = QHBoxLayout(self)
        left = QVBoxLayout()

        # Patch folder
        gb_data = QGroupBox("Patch Folder")
        d_l = QVBoxLayout(gb_data)
        h = QHBoxLayout()
        self.btn_patch = QPushButton("Select")
        self.btn_patch.setObjectName("secondary")
        self.btn_patch.clicked.connect(self._select_patch)
        self.lbl_patch = QLabel("(from Data tab)")
        self.lbl_patch.setStyleSheet("color: #6b7280;")
        h.addWidget(self.btn_patch)
        h.addWidget(self.lbl_patch, 1)
        d_l.addLayout(h)
        self.lbl_bands = QLabel("Bands: -")
        d_l.addWidget(self.lbl_bands)
        left.addWidget(gb_data)

        # Method & Mode
        gb_method = QGroupBox("Method & Mode")
        f_method = QFormLayout(gb_method)

        self.cb_method = QComboBox()
        self.cb_method.addItems([
            "N2V (Noise2Void)",
            "N2F (Noise2Fast)",
            "N2N2 (Neighbor2Neighbor)",
        ])
        self.cb_method.currentIndexChanged.connect(self._update_param_visibility)
        f_method.addRow("Method:", self.cb_method)

        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["2D (band-wise)", "3D (spatial+spectral)"])
        self.cb_mode.currentIndexChanged.connect(self._update_param_visibility)
        f_method.addRow("Mode:", self.cb_mode)

        self.lbl_method_info = QLabel("")
        self.lbl_method_info.setStyleSheet("color: #6b7280; font-size: 9pt;")
        self.lbl_method_info.setWordWrap(True)
        f_method.addRow(self.lbl_method_info)

        left.addWidget(gb_method)

        # Training params
        gb_m = QGroupBox("Training Parameters")
        f = QFormLayout(gb_m)

        self.sp_chunk = QSpinBox()
        self.sp_chunk.setRange(8, 512)
        self.sp_chunk.setSingleStep(8)
        self.sp_chunk.setValue(64)
        self.lbl_chunk = QLabel("Spectral chunk (3D):")
        f.addRow(self.lbl_chunk, self.sp_chunk)

        self.sp_epochs = QSpinBox()
        self.sp_epochs.setRange(1, 1000)
        self.sp_epochs.setValue(50)
        f.addRow("Epochs:", self.sp_epochs)

        self.sp_bs = QSpinBox()
        self.sp_bs.setRange(1, 64)
        self.sp_bs.setValue(8)
        f.addRow("Batch size:", self.sp_bs)

        self.sp_bps = QSpinBox()
        self.sp_bps.setRange(1, 64)
        self.sp_bps.setValue(8)
        self.lbl_bps = QLabel("Bands per sample (2D):")
        f.addRow(self.lbl_bps, self.sp_bps)

        self.sp_lr = QDoubleSpinBox()
        self.sp_lr.setDecimals(5)
        self.sp_lr.setRange(1e-5, 1e-1)
        self.sp_lr.setSingleStep(1e-4)
        self.sp_lr.setValue(1e-3)
        f.addRow("Learning rate:", self.sp_lr)

        self.sp_mask = QDoubleSpinBox()
        self.sp_mask.setDecimals(4)
        self.sp_mask.setRange(0.0001, 0.1)
        self.sp_mask.setSingleStep(0.001)
        self.sp_mask.setValue(0.005)
        self.lbl_mask = QLabel("Mask ratio (N2V):")
        f.addRow(self.lbl_mask, self.sp_mask)

        self.sp_gamma = QDoubleSpinBox()
        self.sp_gamma.setDecimals(2)
        self.sp_gamma.setRange(0.0, 10.0)
        self.sp_gamma.setSingleStep(0.5)
        self.sp_gamma.setValue(1.0)
        self.lbl_gamma = QLabel("γ regularizer (N2N2):")
        f.addRow(self.lbl_gamma, self.sp_gamma)

        left.addWidget(gb_m)

        # Save Path
        gb_save = QGroupBox("Save Path")
        sl = QHBoxLayout(gb_save)
        self.btn_save = QPushButton("Choose")
        self.btn_save.setObjectName("secondary")
        self.btn_save.clicked.connect(self._select_save)
        self.lbl_save = QLabel("./model.pth")
        self.lbl_save.setStyleSheet("color: #6b7280;")
        sl.addWidget(self.btn_save)
        sl.addWidget(self.lbl_save, 1)
        left.addWidget(gb_save)

        # Buttons
        h2 = QHBoxLayout()
        self.btn_train = QPushButton("Start Training")
        self.btn_train.setMinimumHeight(36)
        self.btn_train.clicked.connect(self._train)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("secondary")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        h2.addWidget(self.btn_train, 2)
        h2.addWidget(self.btn_stop, 1)
        left.addLayout(h2)

        self.pb = QProgressBar()
        left.addWidget(self.pb)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        left.addWidget(self.log, 1)

        # Right: loss plot
        right = QVBoxLayout()
        title = QLabel("Training Loss")
        title.setObjectName("section_title")
        right.addWidget(title)
        self.fig = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, alpha=0.3)
        right.addWidget(self.canvas, 1)

        layout.addLayout(left, 2)
        layout.addLayout(right, 3)

    def _get_method(self):
        return ["N2V", "N2F", "N2N2"][self.cb_method.currentIndex()]

    def _get_mode(self):
        return ["2D", "3D"][self.cb_mode.currentIndex()]

    def _update_param_visibility(self):
        method = self._get_method()
        mode = self._get_mode()

        # Method 정보 표시
        info = METHOD_INFO[method]
        if method != "N2V" and mode == "3D":
            info += "\n  ⚠ N2F/N2N2은 2D 모드만 지원. 자동으로 2D로 진행."
        self.lbl_method_info.setText(info)

        # 3D만 적용되는 항목
        is_3d = (method == "N2V" and mode == "3D")
        self.lbl_chunk.setVisible(is_3d)
        self.sp_chunk.setVisible(is_3d)

        # bands_per_sample은 2D 모드에서만
        is_2d = not is_3d
        self.lbl_bps.setVisible(is_2d)
        self.sp_bps.setVisible(is_2d)

        # mask_ratio는 N2V에서만
        is_n2v = (method == "N2V")
        self.lbl_mask.setVisible(is_n2v)
        self.sp_mask.setVisible(is_n2v)

        # gamma는 N2N2에서만
        is_n2n2 = (method == "N2N2")
        self.lbl_gamma.setVisible(is_n2n2)
        self.sp_gamma.setVisible(is_n2n2)

    def _select_patch(self):
        d = QFileDialog.getExistingDirectory(self, "Select patch folder")
        if not d:
            return
        self.lbl_patch.setText(d)
        self.state["patch_dir"] = d
        files = [f for f in os.listdir(d) if f.endswith(".npy")]
        if files:
            sample = np.load(os.path.join(d, files[0]))
            self.lbl_bands.setText(f"Bands: {sample.shape[2]}")
            self.state["n_bands"] = sample.shape[2]
            self.state["patch_size"] = sample.shape[0]

    def _select_save(self):
        f, _ = QFileDialog.getSaveFileName(
            self, "Save model as", "model.pth", "PyTorch (*.pth)"
        )
        if f:
            self.lbl_save.setText(f)

    def _train(self):
        pd = self.state.get("patch_dir")
        if not pd:
            QMessageBox.warning(self, "Warning", "No patch folder set.")
            return

        method = self._get_method()
        mode = self._get_mode()

        # N2F/N2N2은 3D 미지원 → 강제 2D
        if method != "N2V" and mode == "3D":
            mode = "2D"
            self.log.append(f"[Note] {method}: forced 2D mode")

        cfg = {
            "patch_dir": pd,
            "method": method,
            "mode": mode,
            "spectral_chunk": self.sp_chunk.value(),
            "epochs": self.sp_epochs.value(),
            "batch_size": self.sp_bs.value(),
            "bands_per_sample": self.sp_bps.value(),
            "lr": self.sp_lr.value(),
            "mask_ratio": self.sp_mask.value(),
            "gamma": self.sp_gamma.value(),
            "patch_size": self.state.get("patch_size", 128),
            "save_path": self.lbl_save.text(),
        }
        self.losses = []
        self.ax.clear()
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log.clear()
        self.pb.setValue(0)

        self.worker = TrainWorker(cfg)
        self.worker.progress.connect(self._on_prog)
        self.worker.log.connect(self.log.append)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self.btn_stop.setEnabled(False)

    def _on_prog(self, ep, total, loss):
        self.pb.setValue(int(100 * ep / total))
        self.losses.append(loss)
        self.ax.clear()
        self.ax.plot(range(1, len(self.losses) + 1), self.losses, "b-o", markersize=3)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def _on_done(self, path):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log.append(f"\n[Done] Model saved → {path}")
        self.state["model_path"] = path
        if self.predict_tab is not None:
            self.predict_tab.notify_model_ready(path)
        QMessageBox.information(self, "Done", f"Training complete.\nModel: {path}")

    def _on_fail(self, msg):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log.append(f"\n[Error] {msg}")
        QMessageBox.critical(self, "Error", msg)

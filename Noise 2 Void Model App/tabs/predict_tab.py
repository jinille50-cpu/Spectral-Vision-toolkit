"""
Predict Tab: load model, run inference on new .mat files.
Works for N2V / N2F / N2N2 (model API is identical).
"""
import os
import sys
import json
import numpy as np
import scipy.io as sio
import torch

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QFormLayout, QSpinBox, QTextEdit,
    QProgressBar, QMessageBox, QListWidget, QComboBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import build_model
from utils import (
    load_mat_image, find_mat_files, pad_to_size,
    extract_patches, reconstruct_from_patches,
)


class PredictWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        try:
            cfg = self.cfg
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            ckpt = torch.load(cfg["model_path"], map_location=device, weights_only=False)
            mode = ckpt.get("mode", "2D")
            method = ckpt.get("method", "N2V")
            spectral_chunk = ckpt.get("spectral_chunk") or 64
            patch_size = ckpt.get("patch_size", 128)

            model = build_model(mode).to(device)
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            model.load_state_dict(sd, strict=False)
            model.eval()

            self.progress.emit(0, f"Method: {method}, Mode: {mode}, patch={patch_size}")

            os.makedirs(cfg["out_dir"], exist_ok=True)
            out_files = []
            total = len(cfg["mat_files"])

            for i, mp in enumerate(cfg["mat_files"]):
                self.progress.emit(
                    int(100 * i / total),
                    f"[{i+1}/{total}] {os.path.basename(mp)}"
                )
                img = load_mat_image(mp)
                orig_H, orig_W, C = img.shape

                img_p = pad_to_size(img, target_size=patch_size)
                patches, positions = extract_patches(
                    img_p, patch_size=patch_size, stride=patch_size // 2
                )

                denoised_patches = []
                BATCH_2D = 64
                for p in patches:
                    if mode == "2D":
                        # band 일괄 배치
                        bands = p.transpose(2, 0, 1).astype(np.float32)
                        m = bands.mean(axis=(1, 2), keepdims=True)
                        s = bands.std(axis=(1, 2), keepdims=True) + 1e-8
                        bands_n = (bands - m) / s

                        out_bands = np.empty_like(bands_n)
                        with torch.no_grad():
                            for s_idx in range(0, C, BATCH_2D):
                                e_idx = min(s_idx + BATCH_2D, C)
                                seg = bands_n[s_idx:e_idx]
                                t = torch.from_numpy(seg).unsqueeze(1).to(
                                    device, non_blocking=True)
                                pred = model(t).squeeze(1).cpu().numpy()
                                out_bands[s_idx:e_idx] = pred

                        out_bands = out_bands * s + m
                        out_p = out_bands.transpose(1, 2, 0)
                        denoised_patches.append(out_p)
                    else:
                        # 3D
                        out_p = np.zeros_like(p)
                        chunk = spectral_chunk
                        with torch.no_grad():
                            starts = list(range(0, C, chunk))
                            if starts and starts[-1] + chunk > C:
                                starts[-1] = max(0, C - chunk)
                            for s_idx in starts:
                                e_idx = min(s_idx + chunk, C)
                                actual = e_idx - s_idx
                                seg = p[:, :, s_idx:s_idx + chunk]
                                if seg.shape[2] < chunk:
                                    pad_c = chunk - seg.shape[2]
                                    seg = np.pad(
                                        seg, ((0, 0), (0, 0), (0, pad_c)), mode="reflect"
                                    )
                                seg_t = np.transpose(seg, (2, 0, 1)).astype(np.float32)
                                m, s = seg_t.mean(), seg_t.std() + 1e-8
                                seg_n = (seg_t - m) / s
                                t = torch.from_numpy(seg_n).unsqueeze(0).unsqueeze(0).to(device)
                                pred = model(t).squeeze().cpu().numpy()
                                pred = pred * s + m
                                pred = np.transpose(pred, (1, 2, 0))[:, :, :actual]
                                out_p[:, :, s_idx:s_idx + actual] = pred
                        denoised_patches.append(out_p)

                full = reconstruct_from_patches(
                    denoised_patches, positions,
                    full_shape=img_p.shape, patch_size=patch_size,
                )
                full = full[:orig_H, :orig_W, :]

                base = os.path.splitext(os.path.basename(mp))[0]
                out_path = os.path.join(cfg["out_dir"], f"{base}_denoised.mat")
                sio.savemat(out_path, {"denoised": full, "method": method})
                out_files.append(out_path)
                self.progress.emit(
                    int(100 * (i + 1) / total), f"  Saved → {out_path}"
                )

            self.finished_ok.emit(out_files)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class PredictTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.mat_files = []
        self.current_orig = None
        self.current_denoised = None
        self._build_ui()

    def notify_model_ready(self, path):
        self.lbl_model.setText(path)
        self._update_model_info(path)

    def _update_model_info(self, path):
        """모델 metadata 표시"""
        try:
            cfg_path = path.replace(".pth", "_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    info = json.load(f)
                txt = (f"Method: {info.get('method', '?')}, "
                       f"Mode: {info.get('mode', '?')}, "
                       f"Patch: {info.get('patch_size', '?')}")
                self.lbl_model_info.setText(txt)
            else:
                self.lbl_model_info.setText("(no config.json)")
        except Exception:
            self.lbl_model_info.setText("(config read failed)")

    def _build_ui(self):
        layout = QHBoxLayout(self)
        left = QVBoxLayout()

        gb_m = QGroupBox("1. Model")
        m_l = QVBoxLayout(gb_m)
        h_m = QHBoxLayout()
        self.btn_model = QPushButton("Load .pth")
        self.btn_model.setObjectName("secondary")
        self.btn_model.clicked.connect(self._select_model)
        self.lbl_model = QLabel("(not loaded)")
        self.lbl_model.setStyleSheet("color: #6b7280;")
        h_m.addWidget(self.btn_model)
        h_m.addWidget(self.lbl_model, 1)
        m_l.addLayout(h_m)
        self.lbl_model_info = QLabel("")
        self.lbl_model_info.setStyleSheet("color: #2563eb; font-size: 9pt;")
        m_l.addWidget(self.lbl_model_info)
        left.addWidget(gb_m)

        gb_in = QGroupBox("2. Input .mat folder")
        in_l = QVBoxLayout(gb_in)
        h = QHBoxLayout()
        self.btn_in = QPushButton("Select")
        self.btn_in.setObjectName("secondary")
        self.btn_in.clicked.connect(self._select_in)
        self.lbl_in = QLabel("(not selected)")
        self.lbl_in.setStyleSheet("color: #6b7280;")
        h.addWidget(self.btn_in)
        h.addWidget(self.lbl_in, 1)
        in_l.addLayout(h)
        self.list_files = QListWidget()
        self.list_files.setMaximumHeight(120)
        self.list_files.itemClicked.connect(self._preview_input)
        in_l.addWidget(self.list_files)
        left.addWidget(gb_in)

        gb_out = QGroupBox("3. Output folder")
        out_l = QHBoxLayout(gb_out)
        self.btn_out = QPushButton("Select")
        self.btn_out.setObjectName("secondary")
        self.btn_out.clicked.connect(self._select_out)
        self.lbl_out = QLabel("(not selected)")
        self.lbl_out.setStyleSheet("color: #6b7280;")
        out_l.addWidget(self.btn_out)
        out_l.addWidget(self.lbl_out, 1)
        left.addWidget(gb_out)

        self.btn_run = QPushButton("Run Inference")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.clicked.connect(self._run)
        left.addWidget(self.btn_run)

        self.pb = QProgressBar()
        left.addWidget(self.pb)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        left.addWidget(self.log)

        right = QVBoxLayout()
        ttl = QLabel("Preview (click a file)")
        ttl.setObjectName("section_title")
        right.addWidget(ttl)

        h_pv = QHBoxLayout()
        h_pv.addWidget(QLabel("Band:"))
        self.sp_band = QSpinBox()
        self.sp_band.setRange(0, 9999)
        self.sp_band.setValue(0)
        self.sp_band.valueChanged.connect(self._update_preview)
        h_pv.addWidget(self.sp_band)
        h_pv.addStretch()
        right.addLayout(h_pv)

        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        right.addWidget(self.canvas, 1)

        layout.addLayout(left, 2)
        layout.addLayout(right, 3)

    def _select_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select model .pth", "", "PyTorch (*.pth)")
        if f:
            self.lbl_model.setText(f)
            self.state["model_path"] = f
            self._update_model_info(f)

    def _select_in(self):
        d = QFileDialog.getExistingDirectory(self, "Select .mat folder")
        if not d:
            return
        self.lbl_in.setText(d)
        self.mat_files = find_mat_files(d)
        self.list_files.clear()
        for f in self.mat_files:
            self.list_files.addItem(os.path.basename(f))

    def _select_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.lbl_out.setText(d)

    def _preview_input(self, item):
        idx = self.list_files.row(item)
        if idx < 0 or idx >= len(self.mat_files):
            return
        try:
            img = load_mat_image(self.mat_files[idx])
            self.current_orig = img
            self.current_denoised = None
            base = os.path.splitext(os.path.basename(self.mat_files[idx]))[0]
            out_dir = self.lbl_out.text()
            if out_dir != "(not selected)":
                cand = os.path.join(out_dir, f"{base}_denoised.mat")
                if os.path.exists(cand):
                    try:
                        d = sio.loadmat(cand)
                        self.current_denoised = d.get("denoised")
                    except Exception:
                        pass
            self.sp_band.setMaximum(img.shape[2] - 1)
            self.sp_band.setValue(img.shape[2] // 2)
            self._update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _update_preview(self):
        if self.current_orig is None:
            return
        b = self.sp_band.value()
        b = min(b, self.current_orig.shape[2] - 1)
        self.fig.clear()

        if self.current_denoised is not None:
            ax1 = self.fig.add_subplot(2, 2, 1)
            ax2 = self.fig.add_subplot(2, 2, 2)
            ax3 = self.fig.add_subplot(2, 1, 2)
            ax1.imshow(self.current_orig[:, :, b], cmap="gray")
            ax1.set_title(f"Original (band {b})")
            ax1.axis("off")
            ax2.imshow(self.current_denoised[:, :, b], cmap="gray")
            ax2.set_title("Denoised")
            ax2.axis("off")
            cy = self.current_orig.shape[0] // 2
            cx = self.current_orig.shape[1] // 2
            ax3.plot(self.current_orig[cy, cx, :], "k-", alpha=0.5, label="Original")
            ax3.plot(self.current_denoised[cy, cx, :], "b-", label="Denoised")
            ax3.set_xlabel("Band")
            ax3.set_ylabel("Intensity")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax = self.fig.add_subplot(111)
            ax.imshow(self.current_orig[:, :, b], cmap="gray")
            ax.set_title(f"Original (band {b})")
            ax.axis("off")
        self.canvas.draw()

    def _run(self):
        if self.lbl_model.text() == "(not loaded)":
            QMessageBox.warning(self, "Warning", "Load a model first.")
            return
        if not self.mat_files:
            QMessageBox.warning(self, "Warning", "Select input folder.")
            return
        if self.lbl_out.text() == "(not selected)":
            QMessageBox.warning(self, "Warning", "Select output folder.")
            return

        self.btn_run.setEnabled(False)
        self.log.clear()
        cfg = {
            "model_path": self.lbl_model.text(),
            "mat_files": self.mat_files,
            "out_dir": self.lbl_out.text(),
        }
        self.worker = PredictWorker(cfg)
        self.worker.progress.connect(self._on_prog)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_prog(self, val, msg):
        self.pb.setValue(val)
        self.log.append(msg)

    def _on_done(self, files):
        self.btn_run.setEnabled(True)
        self.pb.setValue(100)
        self.log.append(f"\n[Done] {len(files)} files saved.")
        QMessageBox.information(self, "Done", f"{len(files)} files saved.")

    def _on_fail(self, msg):
        self.btn_run.setEnabled(True)
        self.log.append(f"\n[Error] {msg}")
        QMessageBox.critical(self, "Error", msg)

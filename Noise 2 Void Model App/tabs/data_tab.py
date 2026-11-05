"""
Data Tab: select folder of .mat files, inspect shape, generate patches.
"""
import os
import sys
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QFormLayout, QSpinBox, QListWidget,
    QTextEdit, QProgressBar, QMessageBox
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_mat_image, find_mat_files, pad_to_size, extract_patches,
)


class PatchWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str, int, int)  # output_dir, n_patches, n_bands
    failed = pyqtSignal(str)

    def __init__(self, mat_files, out_dir, patch_size, stride):
        super().__init__()
        self.mat_files = mat_files
        self.out_dir = out_dir
        self.patch_size = patch_size
        self.stride = stride

    def run(self):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            total = len(self.mat_files)
            patch_count = 0
            n_bands = None

            for i, mat_path in enumerate(self.mat_files):
                self.progress.emit(
                    int(100 * i / total),
                    f"[{i+1}/{total}] {os.path.basename(mat_path)} loading..."
                )
                img = load_mat_image(mat_path)
                H_orig, W_orig, C = img.shape

                if n_bands is None:
                    n_bands = C
                elif C != n_bands:
                    self.progress.emit(
                        int(100 * i / total),
                        f"  Skipped: band mismatch ({C} vs {n_bands})"
                    )
                    continue

                # padding (작은 이미지는 patch_size로 확장)
                img = pad_to_size(img, target_size=self.patch_size)
                if img.shape[:2] != (H_orig, W_orig):
                    self.progress.emit(
                        int(100 * i / total),
                        f"  Padded: {H_orig}×{W_orig} → {img.shape[0]}×{img.shape[1]}"
                    )

                patches, positions = extract_patches(
                    img, patch_size=self.patch_size, stride=self.stride
                )

                base = os.path.splitext(os.path.basename(mat_path))[0]
                for j, p in enumerate(patches):
                    fname = f"{base}_y{positions[j][0]:04d}_x{positions[j][1]:04d}.npy"
                    np.save(os.path.join(self.out_dir, fname), p.astype(np.float32))
                    patch_count += 1

                self.progress.emit(
                    int(100 * (i + 1) / total),
                    f"  → {len(patches)} patches saved"
                )

            self.finished_ok.emit(self.out_dir, patch_count, n_bands or 0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class DataTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.train_tab = None
        self.mat_files = []
        self._build_ui()

    def set_train_tab(self, t):
        self.train_tab = t

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 1. Input folder
        gb_in = QGroupBox("1. Input Folder (.mat files)")
        in_l = QVBoxLayout(gb_in)
        h = QHBoxLayout()
        self.btn_in = QPushButton("Select Folder")
        self.btn_in.clicked.connect(self._select_in)
        self.lbl_in = QLabel("(not selected)")
        self.lbl_in.setStyleSheet("color: #6b7280;")
        h.addWidget(self.btn_in)
        h.addWidget(self.lbl_in, 1)
        in_l.addLayout(h)

        self.list_files = QListWidget()
        self.list_files.setMaximumHeight(120)
        in_l.addWidget(self.list_files)
        self.lbl_shape = QLabel("Shape: -")
        self.lbl_shape.setStyleSheet("color: #2563eb; font-weight: 600;")
        in_l.addWidget(self.lbl_shape)

        hint = QLabel(
            "Note: Apply your own preprocessing (e.g. AsLS, SNV, baseline) "
            "before loading .mat files."
        )
        hint.setStyleSheet("color: #6b7280; font-style: italic; font-size: 9pt;")
        hint.setWordWrap(True)
        in_l.addWidget(hint)

        layout.addWidget(gb_in)

        # 2. Patch settings
        gb_p = QGroupBox("2. Patch Settings")
        form = QFormLayout(gb_p)
        self.sp_patch = QSpinBox()
        self.sp_patch.setRange(32, 512)
        self.sp_patch.setValue(128)
        self.sp_patch.setSingleStep(32)
        form.addRow("Patch size:", self.sp_patch)

        self.sp_stride = QSpinBox()
        self.sp_stride.setRange(8, 512)
        self.sp_stride.setValue(64)
        form.addRow("Stride (overlap):", self.sp_stride)

        info = QLabel(
            "Images smaller than patch size are auto-padded (reflect). "
            "Patch size must be a multiple of 8."
        )
        info.setStyleSheet("color: #6b7280; font-size: 9pt;")
        info.setWordWrap(True)
        form.addRow(info)

        layout.addWidget(gb_p)

        # 3. Output folder
        gb_out = QGroupBox("3. Output Folder (patches)")
        out_l = QHBoxLayout(gb_out)
        self.btn_out = QPushButton("Select Folder")
        self.btn_out.clicked.connect(self._select_out)
        self.lbl_out = QLabel("(not selected)")
        self.lbl_out.setStyleSheet("color: #6b7280;")
        out_l.addWidget(self.btn_out)
        out_l.addWidget(self.lbl_out, 1)
        layout.addWidget(gb_out)

        # 4. Run
        self.btn_run = QPushButton("Generate Patches")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.clicked.connect(self._run)
        layout.addWidget(self.btn_run)

        self.pb = QProgressBar()
        self.pb.setValue(0)
        layout.addWidget(self.pb)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        layout.addWidget(self.log)

        layout.addStretch()

    def _select_in(self):
        d = QFileDialog.getExistingDirectory(self, "Select .mat folder")
        if not d:
            return
        self.lbl_in.setText(d)
        self.mat_files = find_mat_files(d)
        self.list_files.clear()
        for f in self.mat_files:
            self.list_files.addItem(os.path.basename(f))
        if self.mat_files:
            try:
                img = load_mat_image(self.mat_files[0])
                self.lbl_shape.setText(
                    f"Shape: {img.shape[0]} × {img.shape[1]} × {img.shape[2]}  "
                    f"({len(self.mat_files)} files)"
                )
            except Exception as e:
                self.lbl_shape.setText(f"Shape: error - {e}")

    def _select_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not d:
            return
        self.lbl_out.setText(d)

    def _run(self):
        if not self.mat_files:
            QMessageBox.warning(self, "Warning", "Select input folder first.")
            return
        out = self.lbl_out.text()
        if out == "(not selected)":
            QMessageBox.warning(self, "Warning", "Select output folder first.")
            return

        ps = self.sp_patch.value()
        if ps % 8 != 0:
            QMessageBox.warning(
                self, "Warning",
                f"Patch size must be a multiple of 8. Got {ps}."
            )
            return

        self.btn_run.setEnabled(False)
        self.log.clear()

        self.worker = PatchWorker(
            self.mat_files,
            out,
            self.sp_patch.value(),
            self.sp_stride.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_progress(self, val, msg):
        self.pb.setValue(val)
        self.log.append(msg)

    def _on_done(self, out_dir, n_patches, n_bands):
        self.pb.setValue(100)
        self.log.append(f"\n[Complete] {n_patches} patches saved → {out_dir}")
        self.log.append(f"Spectral bands: {n_bands}")
        self.state["patch_dir"] = out_dir
        self.state["n_bands"] = n_bands
        self.state["patch_size"] = self.sp_patch.value()
        if self.train_tab is not None:
            self.train_tab.notify_data_ready()
        self.btn_run.setEnabled(True)
        QMessageBox.information(
            self, "Done",
            f"{n_patches} patches saved.\nGo to Train tab next."
        )

    def _on_fail(self, msg):
        self.btn_run.setEnabled(True)
        self.log.append(f"\n[Error] {msg}")
        QMessageBox.critical(self, "Error", msg)

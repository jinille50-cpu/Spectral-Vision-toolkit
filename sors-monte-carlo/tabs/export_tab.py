"""
Tab 3: Export — 시뮬레이션 결과 저장 (npz, csv, png)
"""

from pathlib import Path
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QFileDialog, QMessageBox, QTextEdit
)


class ExportTab(QWidget):
    """결과 저장 탭"""

    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        self.state = state
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "<b>시뮬레이션 결과를 저장</b>합니다. Simulation 탭에서 실행을 먼저 완료하세요."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 8px; background: #fef3c7; border-radius: 4px;")
        layout.addWidget(info)

        # ─── Export buttons ───
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)

        self.btn_npz = QPushButton("Save full result (.npz) — raw data, for Python")
        self.btn_npz.clicked.connect(self.save_npz)
        export_layout.addWidget(self.btn_npz)

        self.btn_csv = QPushButton("Save offset profile (.csv) — for Excel/MATLAB")
        self.btn_csv.clicked.connect(self.save_csv)
        export_layout.addWidget(self.btn_csv)

        self.btn_detections = QPushButton("Save detected photons (.csv)")
        self.btn_detections.clicked.connect(self.save_detections)
        export_layout.addWidget(self.btn_detections)

        layout.addWidget(export_group)

        # ─── Status ───
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.status, stretch=1)

        self._log("Export tab ready. Run a simulation first.")

    def _log(self, msg):
        self.status.append(msg)

    def _get_result(self):
        result = self.state.get('last_result')
        if result is None:
            QMessageBox.warning(self, "No result", "먼저 시뮬레이션을 실행하세요.")
            return None
        return result

    def save_npz(self):
        result = self._get_result()
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save NPZ", "sors_mc_result.npz", "NumPy Compressed (*.npz)"
        )
        if not path:
            return
        np.savez_compressed(
            path,
            detected_x=result.detected_x,
            detected_y=result.detected_y,
            detected_weight=result.detected_weight,
            detected_max_depth=result.detected_max_depth,
            n_simulated=result.n_simulated,
            n_detected=result.n_detected,
            total_thickness=result.total_thickness,
        )
        self._log(f"[OK] Saved NPZ: {path}")

    def save_csv(self):
        result = self._get_result()
        if result is None:
            return
        from core.monte_carlo import compute_offset_profile
        centers, intensity, mean_depth = compute_offset_profile(result, n_bins=40)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save offset profile CSV", "offset_profile.csv", "CSV (*.csv)"
        )
        if not path:
            return
        import csv
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['offset_mm', 'intensity', 'mean_depth_mm'])
            for c, i, d in zip(centers, intensity, mean_depth):
                w.writerow([f"{c:.4f}", f"{i:.6f}", f"{d:.4f}"])
        self._log(f"[OK] Saved offset profile: {path}")

    def save_detections(self):
        result = self._get_result()
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save detections CSV", "detections.csv", "CSV (*.csv)"
        )
        if not path:
            return
        import csv
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['x_mm', 'y_mm', 'weight', 'max_depth_mm'])
            for x, y, weight, depth in zip(
                result.detected_x, result.detected_y,
                result.detected_weight, result.detected_max_depth
            ):
                w.writerow([f"{x:.4f}", f"{y:.4f}", f"{weight:.6f}", f"{depth:.4f}"])
        self._log(f"[OK] Saved {len(result.detected_x):,} detections: {path}")

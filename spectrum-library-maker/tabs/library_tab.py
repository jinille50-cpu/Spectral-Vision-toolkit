"""
라이브러리 탭
- 현재 DB 상태 (클래스별 개수, 통계)
- 샘플 검색/삭제/수정
- 스펙트럼 시각화 (라벨별 색)
- 저장/불러오기 (.npz, .csv, .json)
"""
import os
import json
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QLineEdit, QComboBox,
    QListWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QSplitter, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal


class LibraryTab(QWidget):
    database_changed = pyqtSignal()

    def __init__(self, state):
        super().__init__()
        self.state = state

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.has_mpl = True
            self._Canvas = FigureCanvasQTAgg
            self._Figure = Figure
        except ImportError:
            self.has_mpl = False
        self.canvas = None

        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # ── 상단: 통계 + 액션 ──
        top_g = QGroupBox("라이브러리 상태")
        tg = QHBoxLayout(top_g)
        self.summary_label = QLabel("샘플 0개 · 클래스 0개")
        tg.addWidget(self.summary_label)
        tg.addStretch()

        btn_save = QPushButton("💾  저장 (.npz)")
        btn_save.clicked.connect(self._save_db)
        tg.addWidget(btn_save)

        btn_load = QPushButton("📂  불러오기 (.npz)")
        btn_load.setObjectName("secondary")
        btn_load.clicked.connect(self._load_db)
        tg.addWidget(btn_load)

        btn_csv = QPushButton("📊  CSV 내보내기")
        btn_csv.setObjectName("secondary")
        btn_csv.clicked.connect(self._export_csv)
        tg.addWidget(btn_csv)

        btn_clear = QPushButton("⚠  전체 삭제")
        btn_clear.setObjectName("danger")
        btn_clear.clicked.connect(self._clear_all)
        tg.addWidget(btn_clear)
        main.addWidget(top_g)

        # ── 메인: 좌측(클래스 통계) + 우측(시각화) ──
        split = QSplitter(Qt.Horizontal)

        # 좌측: 클래스 통계
        left_g = QGroupBox("클래스별 통계")
        ll = QVBoxLayout(left_g)
        self.class_table = QTableWidget(0, 3)
        self.class_table.setHorizontalHeaderLabels(["클래스", "샘플 수", "원본 라벨"])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.class_table.itemSelectionChanged.connect(self._on_class_selected)
        ll.addWidget(self.class_table)

        bh = QHBoxLayout()
        btn_remove_class = QPushButton("선택 클래스 삭제")
        btn_remove_class.setObjectName("danger")
        btn_remove_class.clicked.connect(self._remove_class)
        bh.addWidget(btn_remove_class)

        btn_rename = QPushButton("선택 클래스 이름 변경")
        btn_rename.setObjectName("secondary")
        btn_rename.clicked.connect(self._rename_class)
        bh.addWidget(btn_rename)
        ll.addLayout(bh)
        split.addWidget(left_g)

        # 우측: 시각화
        right_g = QGroupBox("스펙트럼 시각화")
        rl = QVBoxLayout(right_g)

        opt_h = QHBoxLayout()
        self.viz_all = QCheckBox("모든 클래스 표시"); self.viz_all.setChecked(True)
        self.viz_all.toggled.connect(self._refresh_plot)
        opt_h.addWidget(self.viz_all)

        self.viz_mean = QCheckBox("각 클래스 평균만"); self.viz_mean.setChecked(False)
        self.viz_mean.toggled.connect(self._refresh_plot)
        opt_h.addWidget(self.viz_mean)

        self.viz_norm = QCheckBox("정규화 표시 (각 스펙트럼)")
        self.viz_norm.toggled.connect(self._refresh_plot)
        opt_h.addWidget(self.viz_norm)
        opt_h.addStretch()
        rl.addLayout(opt_h)

        self.canvas_container = QVBoxLayout()
        canvas_w = QWidget(); canvas_w.setLayout(self.canvas_container)
        rl.addWidget(canvas_w, 1)
        split.addWidget(right_g)
        split.setSizes([400, 800])
        main.addWidget(split, 1)

        # 로그
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(80)
        main.addWidget(self.log_box)

    # ────────────────────────────────────────────────────────
    def refresh(self):
        """DB 변경 시 호출"""
        db = self.state["database"]
        classes = {}
        raw_per_class = {}
        for s in db:
            classes[s["label"]] = classes.get(s["label"], 0) + 1
            raw_per_class.setdefault(s["label"], set()).add(s.get("raw_label", ""))

        n_samples = len(db)
        n_classes = len(classes)
        self.summary_label.setText(
            f"<b>샘플 {n_samples}개</b>  ·  <b>클래스 {n_classes}개</b>  ·  "
            f"채널 {len(db[0]['spectrum']) if db else 0}"
        )

        # 클래스 테이블
        self.class_table.setRowCount(n_classes)
        for i, (cls, cnt) in enumerate(sorted(classes.items())):
            raws = ', '.join(sorted(raw_per_class[cls]))
            if len(raws) > 60: raws = raws[:60] + "..."
            self.class_table.setItem(i, 0, QTableWidgetItem(cls))
            self.class_table.setItem(i, 1, QTableWidgetItem(str(cnt)))
            self.class_table.setItem(i, 2, QTableWidgetItem(raws))

        self._refresh_plot()

    # ────────────────────────────────────────────────────────
    def _on_class_selected(self):
        # 선택된 클래스만 표시 옵션
        if not self.viz_all.isChecked():
            self._refresh_plot()

    def _selected_classes(self):
        rows = {i.row() for i in self.class_table.selectedIndexes()}
        return [self.class_table.item(r, 0).text() for r in rows
                if self.class_table.item(r, 0)]

    def _refresh_plot(self):
        if not self.has_mpl: return
        db = self.state["database"]
        if not db: 
            if self.canvas:
                self.canvas_container.removeWidget(self.canvas)
                self.canvas.setParent(None); self.canvas = None
            return

        # 표시할 클래스 결정
        if self.viz_all.isChecked():
            classes = sorted({s["label"] for s in db})
        else:
            classes = self._selected_classes()
            if not classes:
                classes = sorted({s["label"] for s in db})

        # 클래스별 그룹
        by_class = {c: [] for c in classes}
        for s in db:
            if s["label"] in by_class:
                by_class[s["label"]].append(s["spectrum"])

        wl = self.state.get("wavelengths")
        if self.canvas:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None); self.canvas = None

        fig = self._Figure(figsize=(8, 5), facecolor='white')
        ax = fig.add_subplot(111)

        # 색상 팔레트
        import matplotlib.cm as cm
        try:
            cmap = cm.get_cmap('tab10' if len(classes) <= 10 else 'tab20')
        except Exception:
            import matplotlib
            cmap = matplotlib.colormaps['tab10' if len(classes) <= 10 else 'tab20']

        for ci, cls in enumerate(classes):
            specs = by_class[cls]
            if not specs: continue
            arr = np.stack(specs)  # (N, L)
            if self.viz_norm.isChecked():
                arr = (arr - arr.min(axis=1, keepdims=True)) / \
                      (arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True) + 1e-9)
            color = cmap(ci % 20)
            x = wl if (wl is not None and len(wl) == arr.shape[1]) else np.arange(arr.shape[1])

            if self.viz_mean.isChecked():
                mean = arr.mean(axis=0); std = arr.std(axis=0)
                ax.plot(x, mean, color=color, lw=1.8, label=f"{cls} (n={len(specs)})")
                ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
            else:
                # 일부만 표시 (속도)
                show_n = min(len(specs), 30)
                for i in range(show_n):
                    ax.plot(x, arr[i], color=color, lw=0.6, alpha=0.4,
                            label=f"{cls}" if i == 0 else None)

        ax.set_xlabel("Wavelength / Channel" if wl is None else "Wavelength")
        ax.set_ylabel("Intensity (norm.)" if self.viz_norm.isChecked() else "Intensity")
        ax.set_title(f"{len(classes)} class(es) · {sum(len(s) for s in by_class.values())} spectra")
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        self.canvas = self._Canvas(fig)
        self.canvas_container.addWidget(self.canvas)
        self.canvas.draw()

    # ────────────────────────────────────────────────────────
    # 저장/불러오기
    # ────────────────────────────────────────────────────────
    def _save_db(self):
        db = self.state["database"]
        if not db:
            QMessageBox.warning(self, "저장 불가", "라이브러리가 비어있습니다.")
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "라이브러리 저장", "polymer_library.npz", "NPZ files (*.npz)")
        if not fp: return
        try:
            X = np.stack([s["spectrum"] for s in db])
            labels    = np.array([s["label"]     for s in db])
            raw_labels= np.array([s.get("raw_label","") for s in db])
            sources   = np.array([s.get("source","")     for s in db])
            wl = self.state.get("wavelengths")
            kwargs = {"X": X, "labels": labels,
                      "raw_labels": raw_labels, "sources": sources}
            if wl is not None: kwargs["wavelengths"] = np.asarray(wl)
            np.savez_compressed(fp, **kwargs)
            self._log(f"✔ 저장 완료: {fp} ({len(db)}개 샘플)")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    def _load_db(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "라이브러리 불러오기", "", "NPZ files (*.npz)")
        if not fp: return
        try:
            data = np.load(fp, allow_pickle=True)
            X = data["X"]
            labels = data["labels"]
            raw_labels = data["raw_labels"] if "raw_labels" in data.files else labels
            sources = data["sources"] if "sources" in data.files else [""] * len(X)
            wl = data["wavelengths"] if "wavelengths" in data.files else None

            # 기존 DB 덮어쓸지/추가할지
            if self.state["database"]:
                box = QMessageBox.question(self, "불러오기 방법",
                    "기존 라이브러리에 추가하시겠습니까?\n(취소하면 기존 데이터 삭제 후 불러옵니다)",
                    QMessageBox.Yes | QMessageBox.No)
                if box == QMessageBox.No:
                    self.state["database"] = []
                    self.state["wavelengths"] = None

            for i in range(len(X)):
                self.state["database"].append({
                    "spectrum":  np.asarray(X[i], dtype=np.float32),
                    "label":     str(labels[i]),
                    "raw_label": str(raw_labels[i]),
                    "source":    str(sources[i]),
                })
            if wl is not None and self.state.get("wavelengths") is None:
                self.state["wavelengths"] = np.asarray(wl)

            self._log(f"✔ 불러오기 완료: {len(X)}개 (총 {len(self.state['database'])}개)")
            self.refresh()
            self.database_changed.emit()
        except Exception as e:
            QMessageBox.critical(self, "불러오기 실패", str(e))

    def _export_csv(self):
        db = self.state["database"]
        if not db:
            QMessageBox.warning(self, "내보내기 불가", "라이브러리가 비어있습니다.")
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "CSV 내보내기", "polymer_library.csv", "CSV (*.csv)")
        if not fp: return
        try:
            import csv
            wl = self.state.get("wavelengths")
            n_ch = len(db[0]["spectrum"])
            with open(fp, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                # 헤더
                header = ["label", "raw_label", "source"]
                if wl is not None and len(wl) == n_ch:
                    header += [f"{x:.2f}" for x in wl]
                else:
                    header += [f"ch{i+1}" for i in range(n_ch)]
                w.writerow(header)
                # 데이터
                for s in db:
                    w.writerow([s["label"], s.get("raw_label",""), s.get("source","")]
                               + list(s["spectrum"]))
            self._log(f"✔ CSV 저장: {fp}")
        except Exception as e:
            QMessageBox.critical(self, "내보내기 실패", str(e))

    def _clear_all(self):
        if not self.state["database"]: return
        box = QMessageBox.question(self, "전체 삭제",
            f"라이브러리 {len(self.state['database'])}개 샘플 모두 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No)
        if box == QMessageBox.Yes:
            self.state["database"] = []
            self.state["wavelengths"] = None
            self.refresh()
            self.database_changed.emit()
            self._log("✔ 라이브러리 비웠습니다.")

    def _remove_class(self):
        classes = self._selected_classes()
        if not classes:
            QMessageBox.warning(self, "선택 없음", "삭제할 클래스를 선택하세요.")
            return
        box = QMessageBox.question(self, "클래스 삭제",
            f"{', '.join(classes)} 클래스를 모두 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No)
        if box != QMessageBox.Yes: return
        before = len(self.state["database"])
        self.state["database"] = [s for s in self.state["database"]
                                  if s["label"] not in classes]
        removed = before - len(self.state["database"])
        self._log(f"✔ {removed}개 샘플 삭제")
        self.refresh()
        self.database_changed.emit()

    def _rename_class(self):
        classes = self._selected_classes()
        if len(classes) != 1:
            QMessageBox.warning(self, "선택 오류", "정확히 1개 클래스만 선택하세요.")
            return
        old = classes[0]
        from PyQt5.QtWidgets import QInputDialog
        new, ok = QInputDialog.getText(self, "이름 변경", f"'{old}' → 새 이름:")
        if not ok or not new.strip(): return
        new = new.strip()
        cnt = 0
        for s in self.state["database"]:
            if s["label"] == old:
                s["label"] = new; cnt += 1
        self._log(f"✔ '{old}' → '{new}' ({cnt}개)")
        self.refresh()
        self.database_changed.emit()

    def _log(self, msg):
        self.log_box.append(msg)

"""
데이터 추가 탭 (재설계)
- 데이터: 행렬 형태 1파일 불러오기 (여러 샘플)
- 전치 토글로 방향 확인 (샘플 × 채널)
- 라벨 입력 2가지:
    A. 문자열 직접 입력 (각 샘플마다 / 일괄)
    B. 숫자 라벨 파일 + 클래스명 매핑
- 전처리: 크롭 + Moving Average
"""
import os
import re
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QSpinBox, QSplitter, QCheckBox, QMessageBox, QRadioButton,
    QButtonGroup
)
from PyQt5.QtCore import Qt, pyqtSignal


class DataTab(QWidget):
    database_changed = pyqtSignal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.raw_matrix = None
        self.data_matrix = None
        self.data_path = None
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # ── 1. 데이터 파일 불러오기 ──
        load_g = QGroupBox("1. 데이터 파일 불러오기 (행렬)")
        lg = QGridLayout(load_g)

        lg.addWidget(QLabel("구분자:"), 0, 0)
        self.delim_combo = QComboBox()
        self.delim_combo.addItems(["자동", "Tab", "쉼표(,)", "공백", "세미콜론(;)"])
        lg.addWidget(self.delim_combo, 0, 1)

        lg.addWidget(QLabel("건너뛸 헤더 행:"), 0, 2)
        self.skip_rows = QSpinBox(); self.skip_rows.setRange(0, 100); self.skip_rows.setValue(0)
        lg.addWidget(self.skip_rows, 0, 3)

        btn_load = QPushButton("데이터 파일 선택")
        btn_load.clicked.connect(self._load_data)
        lg.addWidget(btn_load, 1, 0, 1, 4)

        self.shape_label = QLabel("불러온 데이터 없음")
        self.shape_label.setStyleSheet("padding: 4px; background: #f3f4f6; border-radius: 4px;")
        lg.addWidget(self.shape_label, 2, 0, 1, 3)

        self.transpose_check = QCheckBox("전치 (Transpose)")
        self.transpose_check.toggled.connect(self._on_transpose)
        lg.addWidget(self.transpose_check, 2, 3)

        dir_hint = QLabel("※ 데이터는 (샘플 수 x 채널 수) 형태여야 함. 방향이 반대면 전치 체크.")
        dir_hint.setObjectName("hint")
        lg.addWidget(dir_hint, 3, 0, 1, 4)
        lg.setColumnStretch(1, 1)
        main.addWidget(load_g)

        # ── 2. 전처리 ──
        prep_g = QGroupBox("2. 전처리 옵션")
        cg = QGridLayout(prep_g)
        self.use_crop = QCheckBox("크롭 사용"); self.use_crop.setChecked(True)
        self.use_crop.toggled.connect(self._apply_preprocessing)
        cg.addWidget(self.use_crop, 0, 0, 1, 4)
        cg.addWidget(QLabel("시작 인덱스:"), 1, 0)
        self.crop_start = QSpinBox(); self.crop_start.setRange(0, 1000000); self.crop_start.setValue(0)
        self.crop_start.valueChanged.connect(self._apply_preprocessing)
        cg.addWidget(self.crop_start, 1, 1)
        cg.addWidget(QLabel("끝 인덱스:"), 1, 2)
        self.crop_end = QSpinBox(); self.crop_end.setRange(0, 1000000); self.crop_end.setValue(0)
        self.crop_end.valueChanged.connect(self._apply_preprocessing)
        cg.addWidget(self.crop_end, 1, 3)

        self.use_smooth = QCheckBox("Moving Average 스무딩 적용"); self.use_smooth.setChecked(True)
        self.use_smooth.toggled.connect(self._apply_preprocessing)
        cg.addWidget(self.use_smooth, 2, 0, 1, 4)
        cg.addWidget(QLabel("윈도우 크기 (홀수 권장):"), 3, 0)
        self.smooth_window = QSpinBox(); self.smooth_window.setRange(2, 101); self.smooth_window.setValue(5)
        self.smooth_window.setSingleStep(2)
        self.smooth_window.valueChanged.connect(self._apply_preprocessing)
        cg.addWidget(self.smooth_window, 3, 1)
        cg.setColumnStretch(1, 1); cg.setColumnStretch(3, 1)
        main.addWidget(prep_g)

        # ── 3. 라벨 입력 ──
        label_g = QGroupBox("3. 라벨 입력")
        lblg = QGridLayout(label_g)

        self.label_mode_group = QButtonGroup(self)
        self.rb_string = QRadioButton("문자열 직접 입력")
        self.rb_string.setChecked(True)
        self.rb_numfile = QRadioButton("숫자 라벨 파일 + 클래스명 매핑")
        self.label_mode_group.addButton(self.rb_string, 0)
        self.label_mode_group.addButton(self.rb_numfile, 1)
        self.rb_string.toggled.connect(self._on_label_mode)
        lblg.addWidget(self.rb_string, 0, 0)
        lblg.addWidget(self.rb_numfile, 0, 1)

        # 방식 A: 문자열
        self.string_box = QWidget()
        sb_l = QGridLayout(self.string_box); sb_l.setContentsMargins(0, 0, 0, 0)
        sb_l.addWidget(QLabel("라벨 적용:"), 0, 0)
        self.string_apply_mode = QComboBox()
        self.string_apply_mode.addItems([
            "모든 샘플 동일 라벨",
            "샘플별 개별 입력 (아래 표에서 편집)",
        ])
        self.string_apply_mode.currentIndexChanged.connect(self._on_string_apply_mode)
        sb_l.addWidget(self.string_apply_mode, 0, 1)
        sb_l.addWidget(QLabel("라벨:"), 1, 0)
        self.single_label = QLineEdit("PP"); self.single_label.setPlaceholderText("예: PP")
        sb_l.addWidget(self.single_label, 1, 1)
        sb_l.setColumnStretch(1, 1)
        lblg.addWidget(self.string_box, 1, 0, 1, 2)

        # 방식 B: 숫자 라벨 파일
        self.numfile_box = QWidget()
        nb_l = QGridLayout(self.numfile_box); nb_l.setContentsMargins(0, 0, 0, 0)
        btn_load_lbl = QPushButton("라벨 파일 선택 (숫자)")
        btn_load_lbl.setObjectName("secondary")
        btn_load_lbl.clicked.connect(self._load_label_file)
        nb_l.addWidget(btn_load_lbl, 0, 0, 1, 2)
        self.label_file_info = QLabel("라벨 파일 없음")
        self.label_file_info.setObjectName("hint")
        nb_l.addWidget(self.label_file_info, 1, 0, 1, 2)
        nb_l.addWidget(QLabel("숫자 -> 클래스명 매핑:"), 2, 0, 1, 2)
        self.num_map_table = QTableWidget(0, 2)
        self.num_map_table.setHorizontalHeaderLabels(["숫자 라벨", "클래스명"])
        self.num_map_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.num_map_table.setMaximumHeight(150)
        nb_l.addWidget(self.num_map_table, 3, 0, 1, 2)
        self.numfile_box.hide()
        lblg.addWidget(self.numfile_box, 2, 0, 1, 2)
        lblg.setColumnStretch(1, 1)
        main.addWidget(label_g)

        # ── 4. 미리보기 + 추가 ──
        prev_g = QGroupBox("4. 미리보기 및 추가")
        pv = QVBoxLayout(prev_g)
        self.preview_table = QTableWidget(0, 3)
        self.preview_table.setHorizontalHeaderLabels(["샘플 #", "라벨", "스펙트럼 길이"])
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_table.setMaximumHeight(200)
        pv.addWidget(self.preview_table)

        ph = QHBoxLayout()
        btn_preview = QPushButton("미리보기 갱신"); btn_preview.setObjectName("secondary")
        btn_preview.clicked.connect(self._refresh_preview)
        ph.addWidget(btn_preview)
        self.btn_add = QPushButton("라이브러리에 추가")
        self.btn_add.clicked.connect(self._add_to_library)
        ph.addWidget(self.btn_add)
        self.status_label = QLabel("라이브러리: 0개 샘플")
        ph.addWidget(self.status_label); ph.addStretch()
        pv.addLayout(ph)
        main.addWidget(prev_g)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(90)
        main.addWidget(self.log_box)

    # ============ 데이터 로드 ============
    def _load_data(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "데이터 파일 선택", "",
            "Data files (*.txt *.csv *.dat *.asc);;All (*.*)")
        if not fp: return
        try:
            arr = self._parse_matrix(fp)
            if arr is None:
                QMessageBox.warning(self, "로드 실패", "데이터를 파싱할 수 없습니다.")
                return
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            self.raw_matrix = arr
            self.data_path = fp
            self.transpose_check.setChecked(False)
            self._log(f"불러옴: {os.path.basename(fp)}  shape={arr.shape}")
            # 크롭 범위를 데이터 채널 수에 맞게 자동 설정 (0 ~ 마지막)
            self._reset_crop_range()
            self._apply_preprocessing()
        except Exception as e:
            QMessageBox.critical(self, "로드 실패", str(e))

    def _reset_crop_range(self):
        """현재 데이터(전치 반영)의 채널 수에 맞춰 크롭 범위를 0~끝으로 설정"""
        if self.raw_matrix is None:
            return
        arr = self.raw_matrix
        n_channels = arr.shape[0] if self.transpose_check.isChecked() else arr.shape[1]
        # 시그널 일시 차단 (자동 설정 중 _apply_preprocessing 중복 호출 방지)
        self.crop_start.blockSignals(True)
        self.crop_end.blockSignals(True)
        # range를 먼저 넓게 잡은 뒤 값 설정 (range/value 충돌 방지)
        self.crop_start.setRange(0, 1000000)
        self.crop_end.setRange(0, 1000000)
        self.crop_start.setValue(0)
        self.crop_end.setValue(n_channels)
        # 그 다음 적절한 상한으로 조정 (최소는 0 유지 → 0이면 전체 사용)
        self.crop_start.setRange(0, max(1, n_channels - 1))
        self.crop_end.setRange(0, n_channels)
        self.crop_start.blockSignals(False)
        self.crop_end.blockSignals(False)
        self._log(f"  크롭 범위 자동 설정: 0 ~ {n_channels}")

    def _parse_matrix(self, fp):
        delim_map = {"자동": None, "Tab": "\t", "쉼표(,)": ",",
                     "공백": None, "세미콜론(;)": ";"}
        delim = delim_map[self.delim_combo.currentText()]
        skip = self.skip_rows.value()
        try:
            return np.loadtxt(fp, delimiter=delim, skiprows=skip)
        except Exception:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[skip:]
            rows = []
            for ln in lines:
                ln = ln.strip()
                if not ln or ln.startswith('#') or ln.startswith('%'):
                    continue
                vals = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', ln)
                if vals:
                    rows.append([float(v) for v in vals])
            if not rows: return None
            ml = max(len(r) for r in rows)
            rows = [r for r in rows if len(r) == ml]
            return np.array(rows)

    def _on_transpose(self):
        # 전치하면 채널 수가 바뀌므로 크롭 범위도 다시 맞춤
        self._reset_crop_range()
        self._apply_preprocessing()

    @staticmethod
    def _moving_average(spec, window):
        if window < 2: return spec
        pad = window // 2
        padded = np.pad(spec, pad, mode='edge')
        kernel = np.ones(window) / window
        return np.convolve(padded, kernel, mode='valid')

    def _apply_preprocessing(self):
        if self.raw_matrix is None:
            return
        arr = self.raw_matrix.copy()
        if self.transpose_check.isChecked():
            arr = arr.T
        if self.use_crop.isChecked():
            cs, ce = self.crop_start.value(), self.crop_end.value()
            # 끝이 0이면 전체 사용 (기본값)
            if ce <= 0 or ce > arr.shape[1]:
                ce = arr.shape[1]
            if cs >= ce:
                self._log(f"경고: 크롭 시작({cs}) >= 끝({ce}), 크롭 건너뜀")
            else:
                arr = arr[:, cs:ce]
        if self.use_smooth.isChecked():
            sw = self.smooth_window.value()
            arr = np.stack([self._moving_average(arr[i], sw) for i in range(arr.shape[0])])
        self.data_matrix = arr
        n_samples, n_channels = arr.shape
        self.shape_label.setText(f"<b>{n_samples} 샘플 x {n_channels} 채널</b>  (전처리 후)")
        self._refresh_preview()

    # ============ 라벨 입력 ============
    def _on_label_mode(self):
        is_string = self.rb_string.isChecked()
        self.string_box.setVisible(is_string)
        self.numfile_box.setVisible(not is_string)
        self._refresh_preview()

    def _on_string_apply_mode(self):
        self._refresh_preview()

    def _load_label_file(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "라벨 파일 선택 (숫자)", "",
            "Label files (*.txt *.csv *.dat);;All (*.*)")
        if not fp: return
        try:
            labels = np.loadtxt(fp).flatten().astype(int)
            self.num_labels = labels
            uniq = sorted(set(labels))
            self.label_file_info.setText(f"라벨 {len(labels)}개  ·  고유값: {uniq}")
            self.num_map_table.setRowCount(len(uniq))
            for i, u in enumerate(uniq):
                self.num_map_table.setItem(i, 0, QTableWidgetItem(str(u)))
                self.num_map_table.setItem(i, 1, QTableWidgetItem(f"Class{u}"))
            self._log(f"라벨 파일: {len(labels)}개, 고유값 {len(uniq)}개")
            self._refresh_preview()
        except Exception as e:
            QMessageBox.critical(self, "라벨 로드 실패", str(e))

    def _get_labels_for_samples(self, n_samples):
        if self.rb_string.isChecked():
            if self.string_apply_mode.currentIndex() == 0:
                lbl = self.single_label.text().strip() or "Unknown"
                return [lbl] * n_samples
            else:
                labels = []
                for i in range(n_samples):
                    item = self.preview_table.item(i, 1)
                    labels.append(item.text().strip() if item else "Unknown")
                return labels
        else:
            if not hasattr(self, 'num_labels'):
                return ["Unknown"] * n_samples
            mapping = {}
            for r in range(self.num_map_table.rowCount()):
                k_item = self.num_map_table.item(r, 0)
                v_item = self.num_map_table.item(r, 1)
                if k_item and v_item:
                    try:
                        mapping[int(k_item.text())] = v_item.text().strip()
                    except ValueError:
                        pass
            labels = []
            for i in range(n_samples):
                if i < len(self.num_labels):
                    num = int(self.num_labels[i])
                    labels.append(mapping.get(num, f"Class{num}"))
                else:
                    labels.append("Unknown")
            return labels

    # ============ 미리보기 & 추가 ============
    def _refresh_preview(self):
        if self.data_matrix is None:
            self.preview_table.setRowCount(0)
            return
        n_samples, n_channels = self.data_matrix.shape
        labels = self._get_labels_for_samples(n_samples)
        editable = (self.rb_string.isChecked() and
                    self.string_apply_mode.currentIndex() == 1)
        self.preview_table.setRowCount(n_samples)
        for i in range(n_samples):
            self.preview_table.setItem(i, 0, QTableWidgetItem(str(i)))
            lbl_item = QTableWidgetItem(labels[i])
            if not editable:
                lbl_item.setFlags(lbl_item.flags() & ~Qt.ItemIsEditable)
            self.preview_table.setItem(i, 1, lbl_item)
            self.preview_table.setItem(i, 2, QTableWidgetItem(str(n_channels)))

    def _add_to_library(self):
        if self.data_matrix is None:
            QMessageBox.warning(self, "추가 불가", "먼저 데이터를 불러오세요.")
            return
        n_samples, n_channels = self.data_matrix.shape
        labels = self._get_labels_for_samples(n_samples)
        if self.state["database"]:
            ref_len = len(self.state["database"][0]["spectrum"])
            if n_channels != ref_len:
                box = QMessageBox.question(self, "채널 수 불일치",
                    f"기존 라이브러리는 {ref_len}채널인데 새 데이터는 {n_channels}채널입니다.\n"
                    f"그래도 추가하시겠습니까?",
                    QMessageBox.Yes | QMessageBox.No)
                if box == QMessageBox.No: return
        src = os.path.basename(self.data_path) if self.data_path else "manual"
        added = 0
        for i in range(n_samples):
            self.state["database"].append({
                "spectrum":  np.asarray(self.data_matrix[i], dtype=np.float32),
                "label":     labels[i],
                "raw_label": labels[i],
                "source":    f"{src}#{i}",
            })
            added += 1
        self._log(f"{added}개 샘플 추가 (총 {len(self.state['database'])}개)")
        self.status_label.setText(f"라이브러리: {len(self.state['database'])}개 샘플")
        self.database_changed.emit()

    def _log(self, msg):
        self.log_box.append(msg)

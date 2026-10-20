"""UI 스모크 테스트: 창 생성 -> 예제 파일 로드 -> crop/스무딩 -> 일괄 저장까지 실제로 돌려본다."""
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
EXAMPLE = os.path.join(os.path.dirname(V2), "Example", "PI")
sys.path.insert(0, V2)

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402

import main as m  # noqa: E402


@pytest.fixture(scope="module")
def app():
    # 테스트가 사용자 설정(QSettings)을 덮어쓰지 않도록 별도 키 사용
    m.MainWindow.APP = "RamanDataProcessingV2-test"
    yield m.create_app([])
    m.QSettings(m.MainWindow.ORG, m.MainWindow.APP).clear()


@pytest.fixture
def win(app):
    w = m.MainWindow()
    w.setAttribute(Qt.WA_DontShowOnScreen, True)
    w.show()
    yield w
    w.close()


@pytest.fixture
def example_folder(tmp_path):
    if not os.path.isdir(EXAMPLE):
        pytest.skip("Example/PI folder not found")
    files = sorted(f for f in os.listdir(EXAMPLE) if f.lower().endswith(".csv"))[:4]
    for f in files:
        shutil.copy(os.path.join(EXAMPLE, f), tmp_path / f)
    return tmp_path, files


def test_buttons_disabled_before_load(win):
    assert not win.btn_apply.isEnabled()
    assert not win.btn_save_all.isEnabled()
    # 로드 전 Reset/Trim 을 눌러도 죽지 않아야 함 (원본 버그 회귀 방지)
    win.on_reset_range()
    win.on_trim_names()
    win.refresh_filelist()


def test_load_crop_smooth_and_batch_save(app, win, example_folder):
    folder, files = example_folder
    win.laser.setCurrentText("785")
    win.chk_multi.setChecked(True)
    win.n_frames.setValue(10)
    win.load_file(str(folder / files[0]))
    app.processEvents()
    assert win.result is not None and len(win.result.x) == 1340
    assert win.btn_apply.isEnabled()

    win.x_start.setValue(400)
    win.x_end.setValue(1800)
    win.on_apply_range()
    assert win.result.x.min() >= 400 and win.result.x.max() <= 1800
    lo, hi = win.region.getRegion()
    assert (lo, hi) == pytest.approx((400, 1800))

    for method in m.core.SMOOTHING_METHODS:
        win.sm_method.setCurrentText(method)
        assert np.isfinite(win.result.corrected).all()

    # 잘못된 레이저 값은 크래시 대신 상태바 경고
    win.laser.setCurrentText("abc")
    assert win.statusBar().currentMessage().startswith("⚠")
    win.laser.setCurrentText("785")

    # 일괄 저장
    win.on_goto_save()
    assert win.tabs.currentIndex() == 1
    assert len(win.filelist) == len(files)
    win.suffix.setText(" 1s 10 avg")
    win.on_trim_names()
    names = win.txt_after.toPlainText().split("\n")
    assert names[0] == files[0].split(" 1s 10 avg")[0]

    m.QMessageBox.information = staticmethod(lambda *a, **k: None)  # 팝업 차단
    win.on_save_all()

    merge = folder / "Merge"
    assert (merge / "Merge_Raw.txt").exists() and (merge / "Merge_AsLS.txt").exists()
    data = np.loadtxt(merge / "Merge_AsLS.txt", skiprows=1)
    assert data.shape[1] == 1 + len(files)
    meta = json.loads((merge / "metadata.json").read_text(encoding="utf-8"))
    assert meta["crop"] == [400.0, 1800.0]
    assert meta["multiframe"] == {"enabled": True, "frames": 10}
    assert meta["skipped"] == []
    assert len(os.listdir(folder / "AsLS")) == len(files)

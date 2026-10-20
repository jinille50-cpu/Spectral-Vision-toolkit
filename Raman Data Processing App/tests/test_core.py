import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core
from core import ProcessParams, process_spectrum


def synthetic(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x_nm = np.linspace(540, 600, n)
    y = 0.3 * np.arange(n) + 500 * np.exp(-((np.arange(n) - 400) / 15) ** 2) + rng.normal(0, 3, n)
    return x_nm, y


def test_nm_to_raman_shift_matches_original_formula():
    x = np.array([540.0, 580.0])
    expected = 1 / (532 * 1e-7) - 1 / (x * 1e-7)
    np.testing.assert_allclose(core.nm_to_raman_shift(x, 532), expected)


def test_asls_tracks_linear_background():
    x, y = synthetic()
    z = core.AsLS(y, lam_exp=5, p_exp=3)
    # 피크 밖 구간에서는 베이스라인이 선형 배경을 따라가야 함
    outside = np.r_[0:300, 600:1000]
    assert np.abs(z[outside] - 0.3 * np.arange(1000)[outside]).mean() < 10


def test_fit_calibration_matches_polyfit_and_rejects_wrong_count():
    measured = np.array([800.0, 1027.0, 1265.0, 1443.0])
    s, b = core.fit_calibration(measured)
    es, eb = np.polyfit(measured, core.CYH_REFERENCE, 1)
    assert s == pytest.approx(es) and b == pytest.approx(eb)
    with pytest.raises(ValueError):
        core.fit_calibration(measured[:3])


def test_average_frames_removes_cosmic_ray():
    n_frames, n_px = 10, 200
    Y = np.tile(np.linspace(100, 200, n_px), (n_frames, 1))
    Y[3, 50] += 5000  # cosmic ray in one frame
    x = np.tile(np.arange(n_px, dtype=float), n_frames)
    xm, ym = core.average_frames(x, Y.ravel(), n_frames)
    assert len(xm) == n_px
    assert ym[50] == pytest.approx(Y[0, 50], abs=1e-6)
    with pytest.raises(ValueError):
        core.average_frames(x, Y.ravel()[:-1], n_frames)


def test_sgfilter_accepts_even_window():
    _, y = synthetic()
    out = core.SGfilter(y, window_length=10)
    assert out.shape == y.shape


@pytest.mark.parametrize("method", core.SMOOTHING_METHODS)
def test_pipeline_all_smoothing_methods(method):
    x, y = synthetic()
    r = process_spectrum(x, y, ProcessParams(smoothing=method, span=5))
    assert r.x.shape == r.corrected.shape == r.baseline.shape
    assert r.xlabel.startswith("Raman")


def test_pipeline_crop_and_errors():
    x, y = synthetic()
    r = process_spectrum(x, y, ProcessParams(crop=(500, 1500)))
    assert r.x.min() >= 500 and r.x.max() <= 1500
    with pytest.raises(ValueError):
        process_spectrum(x, y, ProcessParams(crop=(100000, 100001)))


def test_read_xy_skips_headers(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("Wavelength,Intensity\n# comment\n540.0, 10\n541.0\t11\n\n542,12\n")
    arr = core.read_xy(p)
    assert arr.shape == (3, 2)


def test_unique_names():
    assert core.unique_names(["a", "a", "", "a"]) == ["a", "a_2", "unnamed", "a_3"]

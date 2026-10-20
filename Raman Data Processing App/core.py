# -*- coding: utf-8 -*-
"""Raman 스펙트럼 처리 코어 (GUI 비의존).

원본 main.py의 처리 함수를 그대로 옮기고, sklearn/pandas 의존성만 numpy로 대체했다.
처리 순서: Raman shift 변환 -> (멀티프레임 평균) -> crop -> AsLS 베이스라인 -> 스무딩
"""
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pywt
from scipy import sparse, stats
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve

# 사이클로헥산 레퍼런스 피크 (cm^-1)
CYH_REFERENCE = np.array([801.3, 1028.3, 1266.4, 1444.4])

LASER_PRESETS = ["488", "532", "632.8", "785", "830"]
SMOOTHING_METHODS = ["Raw Data", "S-G filter", "Moving Average", "Wavelet Denoising"]


# ---------------------------------------------------------------- 파일 입출력
def read_xy(path):
    """앞의 두 값이 숫자인 줄만 읽음. 헤더나 빈 줄은 건너뜀. (N, 2) 배열 반환."""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = re.split(r"[,\s]+", line.strip())
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                continue
    return np.array(rows)


def unique_names(names):
    """같은 이름이 있으면 _2, _3 을 붙여 겹치지 않게 만듦."""
    used, out = set(), []
    for n in names:
        base = n.strip() or "unnamed"
        name, k = base, 2
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        out.append(name)
    return out


# ---------------------------------------------------------------- 알고리즘
def nm_to_raman_shift(x_nm, laser_nm):
    """파장(nm) -> 라만 시프트(cm^-1)."""
    x_nm = np.asarray(x_nm, dtype=float)
    return 1 / (float(laser_nm) * 1e-7) - 1 / (x_nm * 1e-7)


def AsLS(y, lam_exp=5, p_exp=3):
    """AsLS baseline. lam = 10**lam_exp, p = 10**(-p_exp). 반환값: baseline z"""
    lam = 10.0 ** lam_exp
    p = 10.0 ** (-p_exp)
    y = np.asarray(y, dtype=float)
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    H = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(10):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + H).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def SGfilter(data, window_length=11, polyorder=3):
    """Savitzky-Golay. 창 길이는 홀수여야 하므로 짝수면 1 올림, 최소 polyorder+2."""
    w = int(window_length)
    if w % 2 == 0:
        w += 1
    w = max(w, polyorder + 2)
    return savgol_filter(data, window_length=w, polyorder=polyorder)


def MovingAverage(data, window_length=5):
    """이동 평균 (입력과 같은 길이로 반환)."""
    return uniform_filter1d(np.asarray(data, dtype=float), size=max(1, int(window_length)), mode="nearest")


def WaveletDenoising(data, wavelet="sym8", level=4):
    """웨이블릿 노이즈 제거 (입력과 같은 길이로 반환)."""
    data = np.asarray(data, dtype=float)
    level = min(int(level), pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len))
    level = max(level, 1)
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[: len(data)]


def grubbs_mean(Yf, alpha=0.01, max_remove=3):
    """Yf: (프레임 수, 픽셀 수). 픽셀마다 프레임 방향으로 위쪽 이상치(코스믹 레이)를
    Grubbs 검정으로 제거한 뒤 평균. 반환: 평균 스펙트럼, 제거된 점 개수"""
    Yf = np.array(Yf, dtype=float)
    removed = 0
    cols = np.arange(Yf.shape[1])
    for _ in range(max_remove):
        n = np.sum(~np.isnan(Yf), axis=0)
        mean = np.nanmean(Yf, axis=0)
        std = np.nanstd(Yf, axis=0, ddof=1)
        idx = np.nanargmax(Yf, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            G = (Yf[idx, cols] - mean) / std
        t = stats.t.ppf(1 - alpha / n, n - 2)
        Gcrit = (n - 1) / np.sqrt(n) * np.sqrt(t**2 / (n - 2 + t**2))
        out = (G > Gcrit) & (n > 3)
        if not out.any():
            break
        Yf[idx[out], cols[out]] = np.nan
        removed += int(out.sum())
    return np.nanmean(Yf, axis=0), removed


def average_frames(x, y, n_frames):
    """세로로 n_frames개 쌓인 스펙트럼을 (첫 프레임 x, Grubbs 평균 y)로 반환."""
    n = int(n_frames)
    if n < 1 or len(y) % n != 0:
        raise ValueError(f"Data length {len(y)} is not divisible by {n} frames.")
    x = np.asarray(x, dtype=float).reshape(n, -1)[0]
    y_mean, _ = grubbs_mean(np.asarray(y, dtype=float).reshape(n, -1))
    return x, y_mean


def fit_calibration(measured, reference=CYH_REFERENCE):
    """측정 피크 위치 -> 레퍼런스 선형 피팅. (slope, intercept) 반환."""
    measured = np.asarray(measured, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(measured) != len(reference):
        raise ValueError(f"Need {len(reference)} peaks, got {len(measured)}.")
    slope, intercept = np.polyfit(measured, reference, 1)
    return float(slope), float(intercept)


def find_cyh_peaks(y, step):
    """캘리브레이션 다이얼로그용 피크 검출. prominence = step * 500 (원본과 동일)."""
    idx, _ = find_peaks(np.asarray(y, dtype=float), prominence=step * 500)
    return idx


# ---------------------------------------------------------------- 파이프라인
@dataclass
class ProcessParams:
    raman_shift: bool = True
    laser_nm: float = 532.0
    slope: float = 1.0
    intercept: float = 0.0
    multiframe: bool = False
    n_frames: int = 10
    crop: Optional[Tuple[float, float]] = None
    lam_exp: int = 5
    p_exp: int = 3
    smoothing: str = "Raw Data"
    span: int = 11


@dataclass
class ProcessResult:
    x: np.ndarray
    y: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    xlabel: str


def process_spectrum(x_axis, y_axis, P: ProcessParams) -> ProcessResult:
    """원본 MainWindow.process 와 동일한 순서. 실패 시 ValueError."""
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)

    # 1) X축 변환 (+ 캘리브레이션)
    if P.raman_shift:
        X = nm_to_raman_shift(x_axis, P.laser_nm)
        X = P.slope * X + P.intercept
        xlabel = "Raman shift (cm⁻¹)"
    else:
        X = x_axis
        xlabel = "Wavelength (nm)"
    Y = y_axis

    # 1-1) 멀티프레임 평균
    if P.multiframe:
        X, Y = average_frames(X, Y, P.n_frames)

    # 2) crop
    if P.crop is not None:
        start, end = sorted(P.crop)
        mask = (X >= start) & (X <= end)
        if mask.sum() < 10:
            raise ValueError("Fewer than 10 points in the selected X range.")
        X, Y = X[mask], Y[mask]

    # 3) baseline
    baseline = AsLS(Y, lam_exp=P.lam_exp, p_exp=P.p_exp)
    Ycorr = Y - baseline

    # 4) smoothing
    if P.smoothing == "S-G filter":
        Ycorr = SGfilter(Ycorr, window_length=P.span)
    elif P.smoothing == "Moving Average":
        Ycorr = MovingAverage(Ycorr, window_length=P.span)
    elif P.smoothing == "Wavelet Denoising":
        Ycorr = WaveletDenoising(Ycorr, level=P.span)

    return ProcessResult(X, Y, baseline, Ycorr, xlabel)

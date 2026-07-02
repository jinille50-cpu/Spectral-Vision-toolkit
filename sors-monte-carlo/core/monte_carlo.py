"""
Monte Carlo simulation dispatcher.
===================================
Automatically selects backend:
  - PMCX (MCX): GPU-accelerated, requires NVIDIA CUDA
                Reference: Fang & Boas, Opt. Express 17, 20178 (2009)
  - Native:     Pure NumPy vectorized, CPU only
                Standard MCML algorithm (Wang, Jacques, Zheng 1995)

Use BACKEND='pmcx' or 'native' explicitly, or 'auto' for automatic selection.
"""

import os

# Backend 선택 (환경변수 또는 default 'auto')
BACKEND = os.environ.get('MC_BACKEND', 'auto').lower()


def _try_pmcx():
    """PMCX 사용 가능 여부 확인"""
    try:
        import pmcx
        # GPU 사용 가능 여부 빠르게 체크 (실제 실행 시도)
        import numpy as np
        test_cfg = {
            'nphoton': 10,
            'vol': np.ones((10, 10, 10), dtype='uint8'),
            'tstart': 0, 'tend': 5e-9, 'tstep': 5e-9,
            'srcpos': [5, 5, 0],
            'srcdir': [0, 0, 1],
            'prop': [[0, 0, 1, 1], [0.01, 1, 0.9, 1.4]],
            'issrcfrom0': 1,
            'autopilot': 1,
        }
        try:
            pmcx.run(test_cfg)
            return True, "PMCX with GPU"
        except RuntimeError as e:
            msg = str(e)
            if 'CUDA' in msg or 'cuda' in msg:
                return False, f"PMCX installed but GPU not available: {e}"
            return False, f"PMCX runtime error: {e}"
    except ImportError:
        return False, "PMCX not installed (pip install pmcx)"


# Backend 선택
_pmcx_available = False
_backend_used = None

if BACKEND == 'auto':
    _ok, _msg = _try_pmcx()
    if _ok:
        from .monte_carlo_pmcx import (
            Layer, SimulationConfig, SimulationResult,
            simulate, compute_offset_profile
        )
        _pmcx_available = True
        _backend_used = 'pmcx'
    else:
        from .monte_carlo_native import (
            Layer, SimulationConfig, SimulationResult,
            simulate, compute_offset_profile
        )
        _backend_used = 'native'
elif BACKEND == 'pmcx':
    from .monte_carlo_pmcx import (
        Layer, SimulationConfig, SimulationResult,
        simulate, compute_offset_profile
    )
    _backend_used = 'pmcx'
elif BACKEND == 'native':
    from .monte_carlo_native import (
        Layer, SimulationConfig, SimulationResult,
        simulate, compute_offset_profile
    )
    _backend_used = 'native'
else:
    raise ValueError(f"Unknown MC_BACKEND: {BACKEND}. Use 'pmcx', 'native', or 'auto'.")


def get_backend_info():
    """현재 사용 중인 백엔드 정보 반환"""
    if _backend_used == 'pmcx':
        return {
            'backend': 'PMCX (MCX)',
            'description': 'GPU-accelerated, NVIDIA CUDA required',
            'reference': 'Fang & Boas, Opt. Express 17, 20178 (2009)',
            'speed': 'Very fast (~1M photons/sec on GPU)',
        }
    else:
        return {
            'backend': 'Native (NumPy)',
            'description': 'Pure CPU, vectorized NumPy',
            'reference': 'Standard MCML (Wang, Jacques, Zheng 1995)',
            'speed': 'Moderate (~25K photons/sec on CPU)',
        }

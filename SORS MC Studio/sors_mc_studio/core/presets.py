"""
Optical Property Presets (mm⁻¹ units)
======================================
문헌값 기반 표준 매질 광학 특성.
참고:
  - Bashkatov et al. (2011) "Optical properties of human tissues"
  - Jacques (2013) "Optical properties of biological tissues: a review"
  - Aernouts et al. (2014) "Dependent scattering in Intralipid..."
  - Sandell & Zhu (2011) for PET-like polymers
"""

PRESETS = {
    "Chicken breast (785nm)": {
        "mua": 0.005, "mus": 12.0, "g": 0.92, "n": 1.40,
        "description": "닭가슴살, 785nm laser",
        "ref": "Bashkatov 2011"
    },
    "Chicken breast (830nm)": {
        "mua": 0.003, "mus": 10.0, "g": 0.93, "n": 1.40,
        "description": "닭가슴살, 830nm laser (사용자 케이스 근접)",
        "ref": "Bashkatov 2011, interpolated"
    },
    "Beef muscle (830nm)": {
        "mua": 0.045, "mus": 13.0, "g": 0.94, "n": 1.41,
        "description": "쇠고기 근육, 830nm",
        "ref": "Jacques 2013"
    },
    "Pork muscle (830nm)": {
        "mua": 0.03, "mus": 11.0, "g": 0.92, "n": 1.40,
        "description": "돼지고기 근육, 830nm",
        "ref": "Bashkatov 2011"
    },
    "Fat (subcutaneous, 830nm)": {
        "mua": 0.01, "mus": 12.0, "g": 0.90, "n": 1.44,
        "description": "피하지방, 830nm",
        "ref": "Bashkatov 2011"
    },
    "PET (transparent)": {
        "mua": 0.005, "mus": 1.5, "g": 0.7, "n": 1.58,
        "description": "투명 PET 플라스틱",
        "ref": "estimated, low scattering"
    },
    "PET (opaque/white)": {
        "mua": 0.01, "mus": 3.5, "g": 0.65, "n": 1.58,
        "description": "불투명 PET",
        "ref": "estimated"
    },
    "Skin (830nm)": {
        "mua": 0.04, "mus": 15.0, "g": 0.85, "n": 1.37,
        "description": "피부 (epidermis+dermis), 830nm",
        "ref": "Bashkatov 2011"
    },
    "Air (no medium)": {
        "mua": 0.0, "mus": 0.001, "g": 1.0, "n": 1.0,
        "description": "공기 (참고용)",
        "ref": "-"
    },
    "Custom": {
        "mua": 0.03, "mus": 10.0, "g": 0.9, "n": 1.4,
        "description": "사용자 정의",
        "ref": "-"
    },
}


def get_preset_names():
    """탭에서 사용할 preset 이름 리스트"""
    return list(PRESETS.keys())


def get_preset(name: str):
    """preset 가져오기 (없으면 Custom 반환)"""
    return PRESETS.get(name, PRESETS["Custom"])

# -*- mode: python ; coding: utf-8 -*-
# 빌드:  pyinstaller RamanApp_v2.spec   (v2 폴더에서 실행)
# 원본 spec 과 달리 console=False 라 검은 콘솔 창이 뜨지 않는다.

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['pywt._extensions._cwt'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pandas', 'sklearn', 'tkinter', 'PyQt5'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='RamanApp_v2',
    debug=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name='RamanApp_v2')

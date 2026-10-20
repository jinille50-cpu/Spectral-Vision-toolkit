# Raman Data Processing App v2

원본(상위 폴더 `main.py` + `ui_raman.py`)의 처리 흐름은 그대로 두고 화면을 코드 기반 레이아웃으로 다시 짠 버전.

## 실행
```
pip install -r requirements.txt
python main.py
```

## 구조
| 파일 | 역할 |
|---|---|
| `core.py` | 처리 알고리즘 (AsLS, 스무딩, Grubbs 프레임 평균, 캘리브레이션 피팅, 파일 읽기). GUI 비의존 |
| `theme.py` | 색상 토큰 + Qt 스타일시트 |
| `calibration_dialog.py` | 사이클로헥산 X축 캘리브레이션 다이얼로그 (pyqtgraph) |
| `main.py` | 메인 창 (Processing / Batch save 탭) |
| `tests/` | `python -m pytest tests` |
| `RamanApp_v2.spec` | PyInstaller 빌드 설정 (`pyinstaller RamanApp_v2.spec`) |

## 원본 대비 달라진 점
- 창 크기에 따라 레이아웃이 늘어남 (원본은 973x655 고정)
- 컨트롤이 Data / X axis / Baseline / Smoothing / X range / Calibration 그룹으로 구분
- raw 플롯에서 영역을 드래그해 crop 범위 지정 가능 (Apply 로 적용)
- 캘리브레이션이 matplotlib 창 대신 앱 안 다이얼로그에서 진행, 잔차 표시
- Esc 로 앱이 꺼지지 않음 (QMainWindow)
- 로드 전 버튼은 비활성, 잘못된 입력은 상태바 경고 (크래시 없음)
- sklearn / pandas / matplotlib 의존성 제거
- 설정(레이저, p, λ, 스무딩, 프레임 수, 접미사) 자동 저장/복원

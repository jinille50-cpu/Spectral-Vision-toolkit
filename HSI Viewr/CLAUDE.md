# HSI Viewer

공통 규칙은 상위 폴더 `../CLAUDE.md`를 따른다.

## 이 앱이 하는 일
- 초분광 큐브(MAT, HDF5, TIFF, NPY, ENVI)를 열어 false color, 단일 밴드, PCA 합성으로 보고, 점이나 영역의 스펙트럼을 확인하는 PySide6 + pyqtgraph 뷰어 (v3.2.0).
- SAM3 텍스트 프롬프트로 조각을 자동 분할하고, 조각별 스펙트럼으로 클러스터링(K 추천 포함), 레퍼런스 스펙트럼 각도(SAM) 분류, YOLO 세그멘테이션 라벨 내보내기, 폴더 단위 배치 처리를 한다.

## 실행
```
cd "HSI Viewer"
python "HSI_Viewer New V1.py"
python "HSI_Viewer New V1.py" path\to\cube.tiff   # 파일을 바로 열기
```

## 주요 파일
- `HSI_Viewer New V1.py`: 진입점 (런처). `hsi_viewer.app.main()` 호출
- `hsi_viewer/app.py`: 메인 창. 좌측 이미지, 우측 상단 그래프, 우측 하단 옵션 두 칸, 파일 열기와 저장
- `hsi_viewer/io_utils.py`: 큐브 불러오기 → (H, W, B) float32, 파장 파일 읽기
- `hsi_viewer/processing.py`: false color, 단일 밴드, PCA 합성, 점/영역 스펙트럼, SNV, 스펙트럼 각도, 밴드 연산
- `hsi_viewer/sam_engine.py`: SAM3 엔진 (QThread에서 별도 프로세스 제어), 조각(Piece) 후처리
- `hsi_viewer/sam_server.py`: 별도 프로세스에서 도는 SAM3 추론 서버 (ultralytics)
- `hsi_viewer/clustering.py`: 조각별 특징 추출, KMeans 등 클러스터링, K 추천, 픽셀 클러스터링
- `hsi_viewer/references.py`: 레퍼런스 스펙트럼 라이브러리와 SAM 기반 재질 분류
- `hsi_viewer/batch.py`: 폴더 배치 처리 (로드 → false color → SAM3 → 콜백)
- `hsi_viewer/yolo_export.py`: 마스크 → YOLO 폴리곤 라벨
- `hsi_viewer/theme.py`: 다크 graphite 테마 (MC Studio들과 같은 색 토큰)
- `hsi_viewer/widgets/`: `image_view.py` 이미지 캔버스, `spectrum_view.py` 스펙트럼 그래프, `plot_area.py` 그래프 전환, `panels.py` 옵션 패널과 고급 설정, `dialogs.py` 배치와 K 추천 창, `common.py` 공용 위젯

## 데이터
- 입력 데이터 위치 — 읽기만: 사용자가 파일 열기나 배치 입력 폴더로 고른 큐브
- 결과 저장 위치: 사용자가 저장 창이나 배치 출력 폴더로 고른 위치
- 모델: 앱 폴더의 `sam3.pt` (약 3.4 GB, `sam_engine.DEFAULT_MODEL`). 옮기거나 지우지 않는다

## 환경
- Python 버전 / 가상환경 경로: Python 3.10.11, 가상환경 없음 (다른 앱과 같은 시스템 Python)
- 주요 패키지: PySide6, pyqtgraph, numpy, scipy, h5py, tifffile, spectral(ENVI), scikit-learn, ultralytics(SAM3), opencv-python

## 앱별 주의사항
- SAM3는 별도 프로세스로 돈다. 추론 코드를 고칠 때는 `sam_server.py`, 결과 처리는 `sam_engine.py`
- 무거운 import(h5py, sklearn, ultralytics, cv2)는 함수 안에서 늦게 불러온다. 이 방식을 유지한다
- `theme.py`는 SORS MC Studio, Raman MC Studio에도 같은 색 토큰이 있다. 한쪽을 고쳐도 다른 쪽에는 반영되지 않는다

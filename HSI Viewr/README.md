# HSI Viewer

초분광(Hyperspectral) 이미지 뷰어. 이미지를 클릭해 스펙트럼을 보고, **SAM3 텍스트 프롬프트로 객체를 자동 추출**한 뒤
**스펙트럼 기준으로 K개 그룹**으로 묶고, 잘못 묶인 객체는 클릭으로 바로 고칩니다.
SWIR 플라스틱 분류 연구용으로 만들었지만 (H, W, Band) 형태의 어떤 큐브든 열 수 있습니다.

![preview](HSI_Viewer_preview.png)

> **English summary** — A PySide6 + pyqtgraph viewer for hyperspectral cubes (TIFF / NPY / MAT / HDF5 / ENVI).
> Click pixels or draw ROIs to plot spectra, pick false-color bands by dragging cursors on the plot,
> run **SAM3 text-prompt segmentation** (in a separate process, GUI never freezes), cluster the detected
> objects into K groups by their spectra (KMeans / Ward / GMM / Spectral, piece- or pixel-level),
> fix assignments with a right-click, and export CSV / NPY / PNG. UI text is Korean.

## 주요 기능

| 영역 | 기능 |
|---|---|
| 이미지 | False color (R/G/B 밴드 선택) · 단일 밴드 · PCA 합성, 대비 클립, 반전, 휠 확대 |
| 스펙트럼 | 포인트 클릭 / ROI 평균 ± σ, 비닝, SNV, 마우스 아래 픽셀 미리보기, 그래프 위 R/G/B 커서 드래그로 밴드 선택 |
| SAM3 | 텍스트 프롬프트(여러 개는 쉼표) → 마스크, 신뢰도 · NMS · 겹침 · 최소 면적 필터, 별도 프로세스 실행 |
| 클러스터링 | 객체별 대표 스펙트럼(SNV → median)을 KMeans / Agglomerative(Ward) / Gaussian Mixture / Spectral 로 K개 분류, 픽셀 단위 모드, 실루엣 점수 |
| 수정 | 객체 우클릭 → 다른 클러스터로 이동 / 새 클러스터 / 삭제 (평균 스펙트럼 자동 재계산) |
| 내보내기 | 객체 표 CSV, 클러스터 평균±표준편차 CSV, 객체별 스펙트럼 CSV, 라벨/클러스터 맵 .npy, 오버레이 PNG, False color PNG, 화면 캡처 |

## 설치

```bash
git clone <this repo>
cd <repo>
pip install -r requirements.txt
```

- Python 3.10+, Windows 11에서 개발·테스트 (Linux/macOS도 동작할 것으로 예상하지만 미검증).
- GPU: SAM3 추론에 CUDA 권장 (RTX 4060 Ti 16 GB 기준 이미지당 약 0.2초). `torch`는 CUDA 버전에 맞게 별도 설치하세요.
- **SAM3 가중치 `sam3.pt`**(약 3.4 GB)는 저장소에 포함되지 않습니다. Ultralytics SAM3 문서를 따라 받은 뒤
  `HSI_Viewer.py` 와 같은 폴더에 두세요. 파일이 없어도 뷰어/스펙트럼 기능은 모두 동작하고 SAM3 버튼만 비활성화됩니다.

## 실행

```bash
python HSI_Viewer.py                       # 빈 뷰어
python HSI_Viewer.py path/to/cube.tiff     # 파일을 바로 열며 실행
```

지원 형식: `.tif/.tiff`, `.npy`, `.mat` (v7 / v7.3), `.h5/.hdf5`, ENVI `.hdr`.
밴드 축은 자동으로 (H, W, B)로 맞추고, 파일에 파장 정보가 있으면 X축을 nm로 표시합니다.

## 화면 구성

```
┌──────────────────────────┬──────────────────────────────────┐
│                          │  [포인트 스펙트럼 | 클러스터    ] │
│        이미지            │           그래프 (가로로 넓게)    │
│                          ├────────────────┬─────────────────┤
│                          │ 이미지·스펙트럼 │ SAM3 탐지       │
└──────────────────────────┴────────────────┴─────────────────┘
```

| 영역 | 조작 |
|---|---|
| 이미지 | 좌클릭: 포인트 추가 · 드래그: 포인트 이동 · 우클릭: 포인트 삭제 · 휠: 확대 · 객체 위 좌클릭 = 선택(그래프에서 굵게), 우클릭 = 클러스터 변경 / 삭제 |
| 그래프 | 포인트/ROI 스펙트럼. R/G/B 세로 커서를 드래그하면 합성 밴드가 바뀜. 마우스 아래 픽셀은 점선 미리보기(축은 고정). 클러스터링 후 "클러스터 스펙트럼" 탭에 평균 ± 1σ — 포인트 그래프와 같은 스케일(세기, SNV 켜면 SNV) |
| 이미지·스펙트럼 | 보기 모드, R/G/B 밴드, 대비 클립, 반전, 비닝, SNV, +ROI, 포인트 지우기, 스펙트럼 저장 |
| SAM3 | 프롬프트(영문), 신뢰도, K, 방법, 고급…, 탐지 + 클러스터링, 클러스터링만, 오버레이(객체별/클러스터별, 불투명도), 내보내기 |

### 단축키

| 키 | 기능 |
|---|---|
| Ctrl+O | 파일 열기 (드래그 앤 드롭도 가능) |
| F | 화면 맞춤 |
| 1 / 2 / 3 | False color / 단일 밴드 / PCA |
| O | 오버레이 켜기/끄기 |
| Enter (프롬프트 칸) | 탐지 + 클러스터링 |
| Ctrl+E | 스펙트럼 저장 |
| Ctrl+Shift+C | 포인트/ROI 모두 지우기 |

## SAM3 → 클러스터링 흐름

1. False color 밴드/대비를 객체가 잘 보이게 맞춥니다. **이 합성 이미지가 SAM3 입력**입니다.
2. 프롬프트에 찾을 대상 입력(예: `plastic piece`), K 설정, **탐지 + 클러스터링**.
   첫 실행 때 별도 프로세스에서 모델을 올립니다(10~60초). 신뢰도를 낮추면 더 많이 잡고, 오탐이 많으면 올립니다.
3. 객체마다 대표 스펙트럼(픽셀별 SNV → median)을 구해 K개로 묶습니다. 결과는 오버레이(`번호:C#`)와 클러스터 스펙트럼 탭에 표시됩니다.
   그래프는 클러스터링용 정규화 공간이 아니라 **원 세기**(SNV를 켜면 SNV)로 그리므로 포인트 스펙트럼과 바로 비교할 수 있습니다.
4. 잘못 묶인 객체는 **우클릭 → C#로 이동 / 새 클러스터 / 삭제**. 평균과 실루엣은 자동 재계산.
5. K나 방법만 바꿀 때는 **클러스터링만** (SAM3 재실행 없음).
6. **고급…**: NMS IoU, 겹침 IoU, 최소 면적, SNV, 배경 정규화(기본 꺼짐 — 흰 배경/레퍼런스가 있을 때만 권장), 픽셀 단위 클러스터링, 대표값(median/mean).

### 내보내기 파일

| 항목 | 내용 |
|---|---|
| `*_objects.csv` | id, prompt, cluster, score, area_px, 중심, bbox |
| `*_cluster_spectra.csv` | 밴드(또는 nm), 클러스터별 `mean`/`std`(원 세기) + `mean_norm`(클러스터링 공간) |
| `*_object_spectra.csv` | 객체별 median 스펙트럼 (열 이름 `obj{id}_C{cluster}`) |
| `*_label_map.npy` | (H, W) int32, 0 = 배경, i = 객체 id |
| `*_cluster_map.npy` | (H, W) int32, -1 = 배경, c = 클러스터 번호 |
| `*_overlay.png` | 현재 캔버스 (오버레이 포함) |

## 구조

```
HSI_Viewer.py                실행 파일
requirements.txt
hsi_viewer/
  app.py                     메인 윈도우
  theme.py                   다크 테마 / 팔레트
  io_utils.py                파일 로더 (MAT, HDF5, TIFF, NPY, ENVI)
  processing.py              False color / PCA / 비닝 / SNV
  sam_engine.py              SAM3 facade (QThread ↔ 자식 프로세스 파이프)
  sam_server.py              SAM3 자식 프로세스 (ultralytics SAM3SemanticPredictor)
  clustering.py              특징 추출, KMeans/Ward/GMM/Spectral, 픽셀 단위, 수동 편집 재계산
  widgets/
    image_view.py            pyqtgraph 캔버스 (마커, ROI, 오버레이, 우클릭 메뉴)
    spectrum_view.py         포인트 / 클러스터 그래프
    plot_area.py             그래프 전환 토글
    panels.py                옵션 카드 두 개 + 고급 설정 대화상자
    common.py                공용 위젯
  references.py, batch.py, yolo_export.py, widgets/dialogs.py
                             레퍼런스 분류·배치 처리·YOLO 내보내기 모듈 (현재 UI에는 연결 안 됨)
```

## 라이선스

이 저장소의 코드는 자유롭게 사용·수정할 수 있습니다. SAM3 가중치는 Meta의 라이선스를, `ultralytics` 는 AGPL-3.0을 따릅니다.

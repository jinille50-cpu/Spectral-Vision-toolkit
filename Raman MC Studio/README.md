# Raman MC Studio

SORS 라만 신호 패턴 Monte Carlo (1.0, PySide6). SORS MC Studio 에서 라만 패턴 화면만 떼어 낸 앱입니다.
HSI Viewer 와 같은 graphite + teal 테마입니다.

## 실행

```
cd "Raman MC Studio"
python run.py
```

필요 패키지: PySide6, matplotlib, numpy, scipy, pmcx (GPU).

## 화면 구성

`raman_mc_studio/raman_pattern/config.json` 으로 계산합니다. '전체 실행' 한 번에 Case 1, 2, 3 과 조직 배경을 모두 계산합니다.

- **Case 1 (1D)**: 레이저가 PET 바로 위에 고정, 깊이별 offset(채널) 패턴. 점 측정을 겹쳐 그릴 수 있습니다.
- **Case 2 (2D, x–z)**: CCD 축과 평행한 x 방향으로 PET 중심을 지나는 라인 스캔. 레이저 위치별 패턴과 |x| 구간 기여율.
- **Case 3 (3D, x–y–z)**: y 방향을 더한 XY 래스터. 레이저–PET 거리 r 구간 기여율 (반경 9 mm 원 안).
- **조직 배경 B(d)**

왼쪽은 공통 설정과 Case 별 설정, 오른쪽 위 버튼으로 Case 를 고르고 그 아래 목록에서 그림을 고릅니다.

## 폴더

- `raman_mc_studio/raman_pattern/` : 라만 패턴 코드.
  `config.json` 의 `fluence_cache` 는 `results/paper/fluence_paper_cone_20M.npy` (약 100 MB) 를 가리킵니다.
  이 파일이 없으면 처음 실행할 때 플루언스를 다시 계산합니다 (GPU, 2천만 광자 약 1분).
  새 결과와 캐시는 `raman_pattern/results/` 에 저장됩니다.

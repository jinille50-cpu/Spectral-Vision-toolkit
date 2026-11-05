# SORS MC Studio

라인스캔 SORS 광자 수송 Monte Carlo 시뮬레이터 (2.0, PySide6).
HSI Viewer 와 같은 graphite + teal 테마입니다.

## 실행

```
cd "SORS MC Studio"
python run.py
```

계산 엔진을 고정하려면 PowerShell 에서 `$env:MC_BACKEND="pmcx"; python run.py`
(PMCX 가 안 되면 오류를 냅니다. 기본값 auto 는 PMCX 가 안 될 때 native 로 넘어갑니다).

필요 패키지: PySide6, matplotlib, numpy, scipy, pmcx (GPU).

## 화면 구성

- **왼쪽 사이드바**: 무엇을 시뮬레이션할지. 매질 층, 광원, 검출기(최대 offset, 띠 폭), 광자 수.
  바꾸면 다시 실행해야 그래프에 반영되고, 그때 사이드바 아래에 안내가 뜹니다.
  맨 위 단면 그림은 입력한 층 두께와 검출 띠를 그대로 보여 줍니다.
- **그래프 위 헤더**: 어떻게 볼지. 검출 방향(+x 한쪽 / 양쪽 따로 / |x| 합침), 그룹 수, 범례.
  다시 실행하지 않고 바로 바뀝니다.
- **요약 줄**: 표면 검출, 바닥 투과, 흡수, 평균 전파 각도, 매질 두께, 검출 띠, 실행 시간.
- **그래프**: 바나나 경로, 전파 각도, 오프셋 프로파일, 깊이 분석, 실측 비교.
  각 그래프의 '그래프 데이터 CSV' 로 그린 값을 저장합니다.
- **아래 패널**: 지표 표 (그룹 평균 ± 표준편차, 표 전체 CSV 저장), 로그.
- **툴바 'MC 결과 저장'**: 전체 결과(.npz), 검출 광자(.csv), 반경 기준 오프셋 프로파일(이전 앱 형식), 지표 표.

## 폴더

- `sors_mc_studio/core/` : 계산 코드. 바탕화면 `Claude Skill/sors_MC_app/core` 에서 복사해 왔고, 이 폴더가 정본입니다.

라만 신호 패턴 계산은 별도 앱 `../Raman MC Studio` 로 분리했습니다.

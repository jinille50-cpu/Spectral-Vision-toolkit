# BACKLOG — Spectral Vision Toolkit (저장소 공통)

항목 형식: `- [ ] B-번호 (등록일) 내용 — 우선순위(상/중/하)`
번호는 지우지 않고 계속 늘린다.

## 진행 중
<!-- 한 번에 하나만 -->
- [ ] B-011 (2026-11-05) `HSI Viewr/`에 이미 공개된 CLAUDE.md, BACKLOG.md, PROGRESS.md 내용 점검 후 공개 여부 결정 — 중

## 대기
- [ ] B-001 (2026-11-05) `SegToolkit/docs/screenshot.png` 합성 데이터로 다시 찍고 README에 링크 복원 — 중
- [ ] B-002 (2026-11-05) `QuantToolkit/docs/screenshot.png` 합성 데이터로 다시 찍고 README에 링크 복원 — 중
- [ ] B-003 (2026-11-05) `yolo-toolkit/docs/screenshot.png` 합성 데이터로 다시 찍고 README에 링크 복원 — 중
- [ ] B-004 (2026-11-05) `Raman Data Processing App/screenshots/processing.png` 합성 데이터로 다시 찍기 — 중
- [ ] B-005 (2026-11-05) `Raman Data Processing App/screenshots/batch_save.png` 합성 데이터로 다시 찍기 — 중
- [ ] B-006 (2026-11-05) `spectrum-library-maker/docs/screenshot.png` 고분자 스펙트럼 출처 확인 후 유지 또는 제거 결정 (사용자 확인 대기, 그 전까지 건드리지 않음) — 상
- [ ] B-008 (2026-11-05) 최상위 README 정리: 깨진 링크, 빠진 앱, 실제와 다른 설명 — 중
- [ ] B-010 (2026-11-05) 이미 추적 중인 `__pycache__`/`.pyc` 38개 추적 해제 — 중
- [ ] B-012 (2026-11-05) `Raman MC Studio` 테스트(`test_pipeline.py`)가 `.gitignore`로 뺀 `make_paper_cases.py`를 불러옴: 공개본에서 처리 방법 결정 — 하

## 완료
<!-- 예: - [x] B-000 (2026-10-20 → 2026-10-24) 내용 -->
- [x] B-009 (2026-11-05 → 2026-11-05) 옛 커밋 주소 차단: GitHub 저장소를 지우고 새로 만들어 해결, 지원팀 요청 불필요 — 상 (확인: 로그인 없이 `5cebd33`, `62de127`의 커밋, 트리, 원본 파일 주소 요청 시 모두 404)
- [x] B-007 (2026-11-05 → 2026-11-05) `.gitignore` 추가 — 상 (확인: 지운 파일과 데이터 제외, 남길 이미지 유지를 `git check-ignore`로 확인, 푸시 후 `origin/main` 기록 검사)

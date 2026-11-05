# 공통 규칙 (C:\Users\User\Desktop\Python Code Develop)

이 폴더 아래 앱들은 각자 하위 폴더에 있다. 작업은 항상 **해당 앱 폴더 안에서만** 한다.

```
Python Code Develop/
├─ HSI Viewer/                 초분광 뷰어 + SAM 3 객체 분할, 클러스터링
├─ Raman Data App/             라만 데이터 처리 앱 v2 (코드 기반 화면)
├─ Raman Data Processing App/  라만 데이터 처리 앱 원본
├─ Raman MC Studio/            SORS 라만 신호 패턴 Monte Carlo
├─ SORS Align/                 SORS MAT 파일 파수 축 맞춤
├─ SORS Depth Labeler/         SORS 깊이 라벨링
├─ SORS MC Studio/             SORS 광자 Monte Carlo
├─ SORS Viewer/                HSI Viewer 번외편 (이미지 나눠 보기, 위키 연동 제외)
├─ md 기본형식/                새 앱 폴더용 md 양식
└─ sync_to_wiki.py             앱 정보를 옵시디언 위키 wiki/코드개발/로 옮기는 스크립트
```
앱별 설명, 화면, 진행 상황은 옵시디언 위키 [[코드개발 현황]]에 자동으로 정리된다.

각 앱 폴더에는 `CLAUDE.md`, `BACKLOG.md`, `PROGRESS.md`, `INBOX.md`가 있다.
앱 폴더를 새로 세팅할 때 `CLAUDE.md`, `BACKLOG.md`, `PROGRESS.md`, `INBOX.md` 4개를 만든다.
앱 화면을 `screenshot.png`로 앱 폴더에 저장해 둔다 (화면이 크게 바뀌면 새로 저장). 위키 메모에 이 그림이 들어간다.

## 작업 흐름

0. **시작할 때 (INBOX)**: 해당 앱 폴더의 `INBOX.md`에 항목이 있으면 `BACKLOG.md` "대기"로 옮기고 번호를 붙인 뒤 `INBOX.md`는 비운다. 옮긴 항목을 사용자에게 보여준다.
1. **시작할 때**: 해당 앱의 `BACKLOG.md`를 읽고, "진행 중" 항목이 있으면 그것부터 이어서 한다.
2. **새 요청이 오면**: 바로 구현하지 말고 먼저 `BACKLOG.md`의 "대기"에 항목으로 추가한 뒤, 지금 할지 물어본다. (단, 사용자가 "바로 해줘"라고 하면 추가 후 바로 진행)
3. **한 번에 한 항목만** 진행한다. 시작하면 "진행 중"으로 옮긴다.
4. **끝나면**:
   - `BACKLOG.md`에서 체크하고 "완료"로 옮긴다 (완료 날짜 기입).
   - `PROGRESS.md` 맨 위에 기록을 추가한다 (형식은 PROGRESS.md 참고).
   - 어떻게 확인했는지(실행, 테스트, 화면 확인) 한 줄로 남긴다.
5. 작업 중 새로 필요한 일이 생기면 직접 하지 말고 `BACKLOG.md` "대기"에 추가만 한다.

6. **위키 갱신**: `BACKLOG.md`, `PROGRESS.md`, `CLAUDE.md`, `screenshot.png` 중 하나라도 바꿨으면, 그 응답을 끝내기 전에 이 폴더(상위 폴더)에서 `python sync_to_wiki.py`를 실행해 옵시디언 위키 `wiki/코드개발/` 메모를 갱신한다. 항목을 대기에 추가만 한 경우도 포함한다.

## 코딩 원칙

- 요청한 것만 구현한다. 요청 없는 기능, 설정 옵션, 추상화는 만들지 않는다.
- 기존 코드는 필요한 곳만 고친다. 주변 코드 정리, 포맷 변경, 리팩터링은 하지 않는다.
- 기존 코드 스타일을 따른다.
- 내가 바꾼 것 때문에 쓸모없어진 import, 변수, 함수는 지운다. 원래 있던 죽은 코드는 지우지 말고 알려만 준다.
- 사용자는 MATLAB에 익숙하고 Python은 학습 중이다. 낯선 문법을 썼으면 짧은 주석으로 설명한다.

## 커밋

- 이 저장소의 커밋 메시지에는 `Co-Authored-By` 줄을 넣지 않는다.

## 하지 말 것

- 다른 앱 폴더의 파일을 수정하지 않는다.
- 파일 삭제, 데이터 폴더 이동은 반드시 먼저 묻는다.
- 원본 측정 데이터는 읽기만 한다.

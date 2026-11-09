# 가설 B 평가 계획

실행 전 고정. 결과를 본 뒤 기준·설정을 바꾸지 않는다. 동작 문제로 바꾸는 경우만 변경 기록에 남긴다.

## 1. 목표

학습한 E1·E2를 baseline(M0)과 같은 절차로 MMMU validation 900문항에 평가하고, 초기 발표(PPP 28–29쪽)에서 정한 기준으로 판정한다.

| 모델 | 무엇 | 출처 |
|---|---|---|
| M0 | 원본 Qwen3-VL-4B-Instruct | 기존 baseline 결과 (600/900, 재실행하지 않음) |
| E1 | 정리 풀이 + Final Answer 학습 | HF `hypB-e1-structured-3k`, 브랜치 `run-yoonseok` (`0de38ff6`) |
| E2 | Final Answer만 학습 | HF `hypB-e2-answer-3k`, 브랜치 `run-yoonseok` (`72e45c1b`) |

## 2. 평가 조건 (baseline과 동일, GitHub main `code/` 기준)

| 항목 | 값 |
|---|---|
| 모델 | 원본(revision `ebb281ec…`) + 각 adapter를 bf16으로 병합. 바뀌는 것은 모델 가중치뿐 |
| 데이터 | MMMU validation 30과목 × 30 = 900, revision `98e6ac0c…` |
| 추론 | vLLM 0.11.0, max_model_len 16384, max_new_tokens 8192 |
| 샘플링 | temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, seed 3407 |
| 이미지·프롬프트 | baseline과 같음 (min/max_pixels, 이미지 앞배치, 공식 프롬프트) |
| 채점 | `hybrid100_mc_qwen_ab_open_no_length_gate_v2` |
| Judge | `gpt-4.1-mini-2025-04-14`, temperature 0, seed 3407, max_tokens 8 |
| GPU | baseline과 같은 RTX 4090 우선 |

평가 코드·설정 파일은 수정하지 않는다. 모델 경로만 병합 모델로 지정한다.

## 3. 판정 기준 (PPP 28–29쪽)

| 구분 | 기준 |
|---|---|
| 1차 | baseline이 length로 잘린 128문항에서 net_gain(오답→정답 − 정답→오답) ≥ k0 + 4. v2(Judge 포함) 점수로 계산 |
| 안전 | 900문항 정확도 하락 ≤ 1.7%p, 그리고 정답→오답 수 ≤ 오답→정답 수 |
| 메커니즘 | 방향 전환 과다 비율, length 종료 비율이 M0보다 줄었는가 |

- k0: 같은 모델을 seed만 바꿔 돌렸을 때 생기는 net_gain(baseline_seed_robustness의 seed 42 / 1234 / 2026 실행, v2 Judge 점수: +10 / +8 / +5. 같은 seed 재실행(다른 호스트)은 +12 / +14). **k0 = 10 확정** (2026-11-01, 채점 전. 초기 발표의 정의대로 seed만 바꾼 재실행의 최댓값) → 1차 기준 net_gain ≥ 14. 같은 seed·다른 호스트 재실행(+12 / +14)은 판정에 쓰지 않고, 보고서에 참고로 "k0=14였다면 ≥18 통과 여부"를 함께 적는다.
- 모든 기준은 k0와 같은 v2(Judge 포함) 점수로 계산한다. Judge 없는 점수(MMMU 규칙 + 랜덤 폴백 제거)는 참고로만 보고한다.
- 방향 전환 과다: wait, perhaps, alternatively, actually, however, reconsider, re-examine, re-evaluate(대소문자 무시)가 응답에 21회 이상. M0 24.1% (217/900).
- E1·E2 각각 M0와 비교한다. E1 vs E2는 응답 형식 효과로 함께 보고한다.
- holdout loss는 판정에 쓰지 않는다 (holdout이 E1 형식이라 E2와 비교 불가).

## 4. 진행 순서

### 4-1. 준비 (비용 없음)
- k0 확정 (A와 합의). 기존 seed 실행으로 k0 값(+10 / +8 / +5)을 같은 v2 점수로 재현 확인
- 평가용 브랜치 작업 위치 확정: `exp/hypB-data`의 `assignment/experiments/hypB_eval/yoonseok/`
- 128문항 목록, 방향 전환 마커 집계 방식을 M0 결과로 먼저 재현해 고정 (M0 24.1%, length 128 재현 확인)

### 4-2. 생성 (GPU, 확인 후)
- GPU 대여, baseline과 같은 환경 구성
- E1 병합 → 900문항 생성 → E2 병합 → 900문항 생성
- 각 실행에서 raw 검증(900 고유 ID, 30×30, 오류 없음), 생성 시간·토큰·VRAM 기록
- raw를 로컬로 받고 해시 기록 후 파드 종료

### 4-3. 채점 (Judge API, 확인 후)
- E1·E2 각각 새 결과 폴더에서 준비(API 없음) → Judge 대상 수·예상 비용 확인
- 소량 확인 후 전체 Judge 실행
- 미완료 문항이 있으면 정확도는 null로 두고 재개

### 4-4. 집계·판정
- 정확도(전체·유형·과목·분야), 처리 경로(자동/Judge/충돌/추출 실패)
- 128문항 net_gain, 900문항 오답→정답 / 정답→오답
- length 종료율, 방향 전환 과다율, 반복 후보율, 출력 토큰
- 3절 기준으로 E1·E2 판정

## 5. 보관 위치

| 무엇 | 어디에 |
|---|---|
| raw 응답, 채점 결과 | 로컬 새 폴더 (해시 기록). GitHub에는 요약만 |
| 집계·판정 보고서, 실행 기록 | GitHub `exp/hypB-data`의 `assignment/experiments/hypB_eval/yoonseok/` |
| 평가에 쓴 모델 | 판정 후 HF 각 저장소 main에 반영하고 `v1.0` 태그 |

## 6. 지키는 것

- validation 결과로 학습 설정·데이터·체크포인트를 다시 고르지 않는다.
- MMMU test·MMMU-Pro는 최종 모델을 정한 뒤 한 번만 평가한다. 이번 단계에서 하지 않는다.
- 채점 변경으로 생긴 차이를 학습 효과로 설명하지 않는다.
- 기존 baseline raw와 팀원 결과는 수정하지 않는다.
- GPU·Judge API 사용 전에 확인한다. 예상: GPU 약 3시간(환경 + 병합 2회 + 생성 2회), Judge는 baseline 수준이면 모델당 1달러 미만.

## 변경 기록

- 2026-11-01: k0 = 10 확정 (E1·E2 생성 후, Judge 채점 전). 14도 검토했으나 초기 발표의 정의(seed 변경 재실행)를 따름.
- 2026-11-01: 생성 환경을 baseline과 같은 이미지 기본 Python으로 변경 (새 venv에서는 torch 메타데이터가 달라 게이트 실패). 병합만 venv 사용.
- 2026-11-01: Judge 실행 방식을 에이전트가 키 입력 창을 띄우고 사용자가 키만 입력하는 방식으로 변경. 키는 그 프로세스 환경에만 두고 저장하지 않음.

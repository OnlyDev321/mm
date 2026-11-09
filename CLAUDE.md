# CLAUDE.md — MMDL Team4 프로젝트 컨텍스트

이 파일은 Claude Code가 MMDL 저장소에서 작업할 때 자동으로 읽는 프로젝트 메모다.
기준일: 2026-10-02. 결정이 바뀌면 이 파일을 먼저 갱신한다.

## 1. 프로젝트 목표

- 과목: 숭실대 2026 Fall Multimodal Deep Learning (박승현 교수), Team4
- 팀원: 이하경, 채윤석, 트란트룽하우, 홍성준
- 목표: `Qwen/Qwen3-VL-4B-Instruct`를 **파인튜닝**해서 MMMU / MMMU-Pro 성능을 개선하고,
  baseline과 **같은 평가 절차**로 비교한다.
- 최종 평가: MMMU test + MMMU-Pro(표준 10지선다). 개발·검증은 MMMU validation.
- **해결책은 반드시 파인튜닝(파라미터 업데이트)이어야 한다.** 프롬프트·추론 방식·파서 변경은
  진단 용도로만 쓰고 성과로 보고하지 않는다.
- **MMMU / MMMU-Pro의 어떤 split도 학습 데이터에 넣지 않는다.** 외부 데이터는 중복·유사 문항 검사 필수.

## 2. 고정 조건 (바꾸면 모든 비교군을 다시 평가해야 함)

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16 |
| 데이터 | `MMMU/MMMU` validation 30과목 × 30 = 900, revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` |
| 추론 | vLLM 0.11.0, max_model_len 16384, max_new_tokens 8192 |
| 샘플링 | temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5, seed 3407 (engine + 요청별) — Qwen3-VL README Instruct 재현 설정 |
| 이미지 | min_pixels 1,003,520 / max_pixels 4,014,080. 모든 이미지를 텍스트 앞에 `image_N` 순서로 배치, 본문 `<image N>` 태그 유지 |
| 프롬프트 | Qwen3-VL 공식 `evaluation/mmmu/run_mmmu.py` 구조 (MC: Question/Options/"Please select the correct answer from the options above.", open: Question만) |
| 채점 | `hybrid100_mc_qwen_ab_open_no_length_gate_v2` |
| Judge | `gpt-4.1-mini-2025-04-14`, temperature 0, top_p 1, seed 3407, max_tokens 8 |

채점 정책 요약:
- 객관식: Qwen 규칙 + `Final Answer` 보완(유효 표기가 모두 같고 마지막 표기 뒤 100자 이내). 충돌·미추출만 Judge.
- 주관식: Qwen 방식 A=참조답 / B=Other Answers.
- length 종료라는 이유만으로 Judge에 보내지 않는다 (게이트 제거, 사용자 결정 — 재논의하지 않음).
- Judge `Z`·추출 실패 = 오답. 무작위 fallback 없음. API 실패는 pending, 미완료면 accuracy=null.

## 3. 확정 baseline (M0)

- **600/900 = 66.67%** (공식 참조 67.4, Δ −0.73%p). 객관식 560/847, 주관식 40/53.
- 정상 종료 554/772 (71.76%), length 종료 46/128 (35.94%). 자동 555 / Judge 345, 추출 실패 36.
- 반복 후보 73/900: 정확도 32.88%, 생성 토큰 32.00%, 추출 실패 20/36, 이 중 67건이 length 종료.
- Tech & Engineering: 문항 23%인데 출력 토큰 49%, 정확도 59.52%로 분야 최저.
- 약한 과목: Music 7/30, Diagnostics 10/30, Mechanical_Engineering 12/30, Agriculture 14/30.
- 오답 60개 표본(오답 300건에서 과목별 2개) 1차 분류: 지식 25, 이미지 해석 17, 계산·추론 8,
  반복·미완결 4, 불확실 5, 평가기 후보 1. 가중 추정치이며 300건 전수 라벨이 아니다.
- 최신 raw SHA-256: `ef23f0c49d9b1ae6c62b9625fbd52cce474a0c035c1a644cfa737e0967daa313`
- 제출 보고서: `reports/mmmu_baseline.md` (루트 사본과 혼동 주의)

해석 주의:
- 채점 변경으로 생긴 점수 변화를 모델 학습 효과로 설명하지 않는다.
- 같은 seed여도 vLLM 출력은 비트 단위로 재현되지 않는다 (8192 실행 length 수 108/114/128).
- 이전 8192 raw의 608/900, TSV·seed42 결과는 과거 기록이며 최신 결과와 섞지 않는다.

## 4. 실험 계획 초안 — **보류, 미확정** (2026-10-02 사용자 지시)

아래 M0~M3 구조는 확정안이 아니다. 오답 300문항 전수 검토(`reports/failure_analysis/failure_audit_300_20261002.md`)를
반영해 방향을 다시 정리하는 중이며, 사용자 확정 전까지 이 계획을 전제로 작업을 진행하지 않는다.

공통 대조군 M1을 공유하는 3개 학습:

| 모델 | 학습 | 비교 |
|---|---|---|
| M0 | 없음 | 원본 |
| M1 | 공통 자료 풀에서 무작위 N개, 풀이 + `Final Answer` | 공통 대조군 |
| M2 | M1과 같은 문항·순서, seed 3407로 고정한 50%만 단답 | M2 vs M1: 응답 형식 효과 |
| M3 | 같은 자료 풀에서 이미지 해석 기준으로 선정한 N개, 풀이 + `Final Answer` | M3 vs M1: 데이터 선정 효과 |

- 데이터 후보: `OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking`
  (`qwen3vl_235b_thinking_response` 풀이, `answer` 정답). 정답·풀이는 user 입력에 넣지 않는다.
- 학습: Unsloth + BF16 LoRA SFT, vision encoder 고정·언어 부분 LoRA부터 점검.
  M1·M2·M3는 같은 원본에서 독립 시작. 20 update 점검 가중치를 이어 쓰지 않는다.
- 학습 입력 형식(이미지 앞배치, 프롬프트 문구, chat template)을 평가와 맞춘다.
- 평가: 모델 snapshot만 바꾸고 2절 조건 그대로 900문항.

단계:
1. CPU: MMFineReason 50건 검토, 우리 tokenizer·processor 기준 길이 분포·초과 비율,
   M3 이미지 해석 기준으로 분류 가능한지, M3 후보 수가 N 이상인지.
2. GPU: 환경 복구 확인 → 입력(해상도) 진단 → 20 update 학습·병합·vLLM 재로딩, VRAM·step 시간 측정.
   여기서 N·시퀀스·학습 설정 확정.
3. M1·M2·M3 학습 → 각 900문항 평가.
후속(조건부): 시각 근거 구조화 학습(레이블 확보 가능 시), LoRA 위치 비교(데이터·메모리·예산 확인 후).

판단 규칙:
- 해상도 진단 결과로 vision LoRA 진행을 자동 결정하지 않는다.
- M1이 교사(Thinking) 풀이 때문에 M0보다 길어져 length가 늘었는지 먼저 확인한 뒤 M2를 해석한다.
- 모든 모델에서 처리 경로(자동/Judge/충돌/추출 실패)를 집계한다. 같은 Final Answer 형식이어도 경로는 달라질 수 있다.
- 결과를 보고 설정·대상·비율을 바꾸지 않는다. 튜닝은 별도 개발 자료에서만.
- 지표: 정확도, 오답→정답 / 정답→오답, 반복 후보율, length 종료율, 추출 실패율, 입력/출력 토큰, 시간·비용.
- 시각 문항 집합은 학습 전에 고정한다 (후보: MMMU `img_type` 메타데이터 기반 정의).

## 5. 일정·예산·자원

- 일정(사용자 확인, 2026-10-02): **계획 발표 6주차(10/5~10/9 주)**, **중간 발표 10주차(11/2~11/6 주)**, **최종 제출 14주차(11/30~12/4 주)**.
  주차 날짜는 학기 시작 9/1 가정이며 정확한 날짜는 팀 공지로 확정.
- 최종 평가(MMMU test·MMMU-Pro)는 validation으로 최종 모델을 확정한 뒤 **한 번만** 수행한다.
  개발 중 방향 선정·설정 조정·모델 선택에 쓰지 않는다 (사용자 지시 — 치팅 방지).

- 팀 120,000원 / 개인 30,000원. 기존 지출 차감 후 잔액 확인 필요.
- 이전 RunPod Pod·디스크는 2026-09-27 백업 후 삭제됨. GPU 재확보 필요.
- 학습 메모리·시간은 아직 실측 전. 비용은 2단계 실측 후 확정.

## 6. 작업 규칙

- 한국어로, 결과·근거·다음 행동을 짧게.
- **비용이 드는 작업(GPU 대여, 유료 API 호출)은 실행 전에 사용자 확인.**
- raw 응답과 팀원 작업은 수정·삭제하지 않는다. 새 결과는 새 경로에 저장하고 해시를 기록.
- 불필요한 도구 설치를 피한다. 설치가 필요하면 먼저 설명한다.
- 비공개로 정한 자료(주관식 오판정 검토·분리형 사전 실험, `failure_review/open_answer/`)는 commit/push하지 않는다.
- 커밋 규칙: `<type>(<scope>): <subject>` (data, exp, train, eval, model, docs, fix, refactor, chore).

## 7. 참고 자료

- 과제·강의 파일은 MMDL 폴더 안에 있다. 경로가 다르면 파일명으로 검색해서 찾는다.
  (2026-10-02 확인: 02·03 PDF, `assignment_guidance.md`, `SUBMISSION_TEMPLATE.md`는 `C:\Users\brain\Downloads\`에 있음)
  - `assignment_guidance.md` — baseline 과제 안내 (고정 조건·채점 기준)
  - `SUBMISSION_TEMPLATE.md` — 보고서 제출 템플릿
  - `01_MMDL_Course_Introduction.pdf` — 강의 소개, 성적 비중, 팀 프로젝트 개요
  - `02_MMDL_Team_Project_Introduction.pdf` — 프로젝트 흐름, GitHub 구조, 발표 요구사항
  - `03_MMDL_VLM_Introduction.pdf` — 평가 방식, 파인튜닝 정의(파라미터 업데이트 필수), 금지 사항.
    p16 데이터: 공개·외부 데이터, 자체 합성 데이터, 제3자 모델 생성·라벨링 허용 / p17 DPO 등 선호 목적함수 언급 /
    p18 전체·부분 파인튜닝, 시각 인코더·LLM·projector, PEFT(LoRA·QLoRA·adapter, 새 파라미터 추가 학습) 허용 /
    p19 파인튜닝 아님: 프롬프트 엔지니어링·few-shot·RAG, 앙상블·test-time scaling, 출력 파싱·정규화
- 강의 PDF는 교수님 자료이므로 공개 레포에 커밋하지 않는다 (`.gitignore` 처리).
- 실험 근거: `assignment/experiments/` 아래 `submission`, `raw_evaluation`, `repetition`, `failure_review`, `scoring_lab`
- 노션 실험 계획: "채윤석 — 이미지 해석·응답 효율 파인튜닝 실험 계획"
- 관련 연구: MMFineReason (arXiv 2601.21821, 4B MMMU 67.4→SFT 69.3→RL 69.6),
  TON (NeurIPS 2025, thought dropout)

# 작업 지시: D1 평가 (4090 — 생성 → v2 채점 준비)

## 먼저 읽을 것

- 평가 절차 원본: `work/eval_E3_20261101/` (README.md, scripts/, reference/, reports/) — **E3 평가와 같은 절차를 그대로 따른다. 바뀌는 것은 adapter뿐**
- 평가 조건: `work/eval_B_20261101/PLAN.md` 2절 (E3 사본: `work/eval_E3_20261101/reference/PLAN_eval_B.md`)
- D1 학습 결과: `work/train_RL_20261106/reports/D1_STAGE3_TRAIN_REPORT.md`, adapter `work/train_RL_20261106/adapter/d1_full/`
- 판정 기준: `work/train_RL_20261106/PLAN_D1.md` 6절
- 프로젝트 규칙: MMDL 루트 `CLAUDE.md`

## 작업 위치

- 개인 실험. **GitHub·HF에 아무것도 올리지 않는다**
- 로컬: `work/eval_D1_20261107/` 새로 만든다. `eval_E3_20261101/`의 scripts·source·reference를 복사해 쓰고, E3 폴더는 수정하지 않는다
- 파드: RTX 4090 1장 (사용자가 빌리고 접속 정보를 준다), 이미지 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

## 1단계: 생성 (GPU)

1. 입력 준비: E3와 같은 `source_bundle.tar.gz`(생성 소스 9fe57b11…, 평가 기준 main 016b3cf5…) + D1 adapter 압축. adapter 파일 SHA-256이 학습 보고서 값과 같은지 먼저 대조
2. 환경: E3와 같게. 생성은 이미지 기본 Python(Torch 2.8.0+cu128) + vLLM 0.11.0, 병합만 system-site-packages venv(PEFT 0.17.1). 그 밖의 패키지는 먼저 묻는다
3. 병합: M0 rev `ebb281ec…`에 D1 adapter를 bf16 병합. 병합 가중치 SHA-256을 A100 학습 후 병합 기록과 대조해 보고
4. 3문항 smoke → 900문항 본 생성. 조건은 E3와 같다 (max_model_len 16384, max_new_tokens 8192, 샘플링·seed·이미지·프롬프트 고정). 바꾸는 것은 모델 경로뿐
5. 검증: 900 고유 ID, 30과목×30, 오류 0, M0와 입력·프롬프트 토큰 일치
6. raw·기록을 로컬 `work/eval_D1_20261107/runs/D1/`, `records/`로 scp, SHA-256 대조
7. 보고 `reports/GENERATION_REPORT.md`: 생성 시간, length 종료 수(사고 과정 미종료 구분 포함), 출력 토큰 합계·평균·중앙값, M0·E3와 비교

## 2단계: 채점 준비 (CPU, 로컬)

- E3와 같은 v2 채점(`hybrid100_mc_qwen_ab_open_no_length_gate_v2`)으로 자동 채점
- 자동 / Judge 대상(규칙 미추출, 충돌) 건수와 보수적 Judge 예산 보고 `reports/SCORING_PREPARATION_REPORT.md`
- **Judge(유료 API)는 호출하지 않는다.** 사용자 승인 후 별도 진행
- 여기서 멈추고 보고

## 지킬 것

- 평가 코드·config·채점 기준·Judge 설정 수정 금지. E3 폴더·M0 raw·학습 결과 수정 금지
- 결과를 보고 재생성·재시도하지 않는다 (오류로 생성 실패한 경우만 보고 후 사용자 결정)
- MMMU test·Pro 평가 금지
- GitHub·HF 업로드 없음. HF 토큰·API 키 출력·저장 금지
- 파드 Stop/Terminate 금지 (사용자가 함)
- 오래 걸리는 작업은 진행 상황을 짧게 알린다

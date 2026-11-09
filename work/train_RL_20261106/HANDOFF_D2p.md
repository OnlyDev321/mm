# 작업 지시: D2' (2회차 DPO) — D1 샘플링 → 쌍 구성 → 학습

## 먼저 읽을 것

- 계획: `work/train_RL_20261106/PLAN_D2p.md` (이번 범위 전체)
- 재사용할 이전 작업 (형식·스크립트 원본):
  - R1 3단계 샘플링: `reports/STAGE3_SAMPLING_REPORT.md`, `scripts/sample_m0.py`, `scripts/build_groups.py`, `scripts/reward_v2.py`
  - D1 쌍·학습: `HANDOFF_D1.md`, `reports/D1_STAGE1_REPORT.md`, `reports/D1_STAGE3_TRAIN_REPORT.md`, `scripts/build_dpo_pairs.py`, `scripts/dpo_train.py`, `scripts/dpo_config.json`
  - D1 병합: `work/eval_D1_20261107/` (4090에서 D1 병합한 절차·해시)
- 프로젝트 규칙: MMDL 루트 `CLAUDE.md`

## 데이터 받기 (HF, 비공개, 팀 org 권한 필요)

- D1 adapter·학습 기록: `MMDL-team4/hypB-d1-dpo` (model) 브랜치 `run-yoonseok`
- 1,500문항 풀·B/C 이미지·M0 샘플링·D1 쌍·보상/쌍 스크립트: `MMDL-team4/hypB-rl-data` (dataset) 브랜치 `run-yoonseok`
- A 이미지: `MMDL-team4/hypB-train-data` v1.1 (커밋 `1833bac3`)
- 로컬 원본이 없으면 위에서 받아 원래 경로(`work/train_RL_20261106/...`)에 맞춰 놓고 각 저장소의 `SHA256SUMS.txt`로 대조한다

## 작업 위치

- 개인 실험. **GitHub에는 올리지 않는다. HF 업로드는 사용자 확인 후 비공개로만**
- 출력: `work/train_RL_20261106/` 아래 `runs/d2p_k4/`, `data/d2p_*`, `adapter/d2p_full/`, `reports/D2P_*`
- 기존 raw·r1_pool·r1_train·D1 쌍·reward_v2·평가 코드·D1 adapter는 수정하지 않는다. 기존 스크립트를 바꿔야 하면 새 이름으로 복사해서 쓴다

## 1단계: D1 샘플링 (RTX 4090, 사용자가 파드를 빌린 뒤)

1. 환경은 R1 3단계와 같게 (이미지 기본 Python Torch 2.8.0+cu128 + vLLM 0.11.0, 병합만 system-site-packages venv PEFT 0.17.1). 그 밖의 패키지는 먼저 묻는다
2. D1 병합: M0 rev `ebb281ec…` + `adapter/d1_full/` (SHA-256 `2d4bcce4…` 대조) → bf16 병합. 병합 샤드 해시가 A100·eval_D1 기록과 같은지 대조
3. `sample_m0.py`를 `scripts/sample_policy.py`로 복사해 **모델 경로 지정 옵션만** 추가 (그 밖의 동작·기본값은 그대로, 원본은 수정 금지). 원본과 diff를 보고서에 첨부
4. CPU dry-run: 프롬프트·이미지·요청 seed가 R1 3단계 기록과 같은지 (5문항 이상)
5. 1,500문항 × K=4 생성, `--reward-v2`. 조건은 R1 3단계와 같다 (temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, max_new_tokens 8192, max_model_len 16384). 바꾸는 것은 모델 경로뿐
6. `build_groups.py`(기준 그대로)로 표적/유지/제외 → 학습 세트 = 섞인 문항 전부 → `runs/d2p_k4/d2p_train.jsonl`
7. raw·그룹·학습 세트·실행 기록을 로컬 `runs/d2p_k4/`로 scp, SHA-256 대조 (`SHA256SUMS.json`)

## 2단계: 쌍 구성 (CPU, 로컬)

1. `build_dpo_pairs.py`를 D1과 같은 규칙으로 실행 (입력만 D2' 결과로). 필요하면 입력 경로 옵션만 추가한 복사본 사용 → `data/d2p_pairs_swift.jsonl`
2. 검증은 D1 1단계와 같다: 쌍 수·문항 수·그룹·분야·rejected 종류 분포, chosen/rejected 토큰 길이, reward_v2 재계산(chosen 1, rejected 0), ms-swift 전처리 토큰 일치 5쌍 이상
3. **쌍 수 < 300이면 학습 준비를 하지 않고 멈추고 보고** (PLAN_D2p 3절 사전 규칙)
4. max_length: 최장 ≤ 10,240이면 10,240 유지. 넘으면 최장 이상 12,288 이하 값을 제안하고 2단계 보고에 "메모리 파일럿 필요"로 표시. 12,288 초과는 멈추고 보고
5. 학습 config `scripts/d2p_config.json`: `dpo_config.json` 복사 후 **시작 모델 경로(D1 병합 가중치)와 데이터 경로, (필요시) max_length만** 변경. 기준 모델은 별도 지정 없이 새 adapter를 끈 시작 모델(D1)
6. 비교표: M0 샘플링(R1 3단계) vs D1 샘플링 — 학습 풀 정답 수 0~4 분포, length 종료 수, 평균 출력 토큰, 그룹 수
7. **멈추고 보고** (`reports/D2P_STAGE1_REPORT.md`, 1·2단계를 합쳐서)

## 3단계: 학습 (A100/H100 80GB, 사용자 확인 후)

- 환경·업로드 방식은 D1 본학습과 같게. D1 병합을 이 파드에서 다시 하고 해시 대조
- 2단계에서 "메모리 파일럿 필요"면 먼저 20 step (최장 쌍 포함) → 보고 후 사용자 확인
- 아니면 바로 본학습 2 epoch. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- step별 기록은 D1과 같은 항목. 학습 대상이 새 LoRA 252모듈뿐이고 기준 모델이 D1인지 확인
- 마지막 체크포인트 → D1 병합본 위에 bf16 병합 → vLLM 0.11.0 로딩 확인 (생성 안 함)
- adapter·기록을 `adapter/d2p_full/`, `reports/d2p_full/`로 scp 후 해시 대조
- OOM·NaN은 설정 바꾸지 말고 멈추고 보고
- **멈추고 보고** (`reports/D2P_STAGE3_TRAIN_REPORT.md`). 평가는 4090에서 따로 함

## 지킬 것

- MMMU / MMMU-Pro는 학습에 쓰지 않는다
- PLAN_D2p 고정값(풀, 샘플링 조건, 그룹 기준, 쌍 규칙, 300 규칙, 학습 설정, 마지막 체크포인트)을 바꾸지 않는다. 바꿔야 하면 멈추고 보고
- GitHub·HF 업로드 없음. HF 토큰·API 키 출력·저장 금지
- 파드 Stop/Terminate 금지 (사용자가 함)
- 오래 걸리는 작업은 진행 상황을 짧게 알린다

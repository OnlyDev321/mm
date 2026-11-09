# 작업 지시: D1 (DPO) — 쌍 구성 → GPU 파일럿

## 먼저 읽을 것

- 계획: `work/train_RL_20261106/PLAN_D1.md` (이번 범위 전체). 데이터 원천은 `PLAN.md` 3·4절
- 이전 결과: `reports/STAGE3_SAMPLING_REPORT.md`(전체 샘플링·학습 세트), `reports/STAGE2B_REWARD_REPORT.md`(reward_v2)
- E3 SFT 파일럿 기록(형식 참고): `work/train_E3_20261101/reports/STAGE3_REPORT.md`, E1 `train.py`·`train_config.json`
- 프로젝트 규칙: MMDL 루트 `CLAUDE.md`

## 전제

- R1 3단계 전체 샘플링이 끝나 `runs/full_k4/`(raw·그룹·`r1_train.jsonl`)이 로컬에 있고 해시가 기록돼 있어야 한다. 없으면 시작하지 않는다

## 작업 위치

- 개인 실험. **GitHub·HF에 아무것도 올리지 않는다**
- 출력: `work/train_RL_20261106/` 아래 `scripts/`, `data/`, `reports/`, `adapter/`
- 샘플링 raw·r1_train·reward_v2·평가 코드는 수정하지 않는다

## 1단계: 쌍 구성 (CPU, 로컬)

1. `scripts/build_dpo_pairs.py`: `r1_train.jsonl`의 각 문항에서 PLAN_D1 3절 규칙대로 쌍을 만든다
   - chosen: 정상 종료 + 보상 1 / rejected: 보상 0, 우선순위 잘림 → 정상 종료 오답 → 추출 실패
   - 문항당 최대 2쌍, 중복 없이, `random.Random(3407)`
   - 응답 텍스트는 raw 그대로 (잘린 응답도 그대로)
2. ms-swift DPO 입력 형식으로 변환 `data/d1_pairs_swift.jsonl` (messages + images + rejected_response). 프롬프트·이미지는 샘플링 때 입력과 바이트 단위 같게
3. 검증
   - 쌍 수, 문항 수, 그룹별(표적/유지)·분야별 분포, rejected 종류 분포(잘림/오답/추출 실패)
   - chosen·rejected 출력 토큰 길이 분포 (평균·중앙값·최대), 쌍 최장 시퀀스(프롬프트 + 응답, 이미지 포함)
   - 모든 chosen의 reward_v2 재계산 = 1, 모든 rejected = 0
   - 5쌍 이상에 대해 ms-swift 전처리 토큰 수(프롬프트·chosen·rejected)와 독립 계산 일치, 프롬프트가 `sample_m0.py` 경로와 같은 토큰 수
4. `max_length` 제안: 최장 시퀀스 이상 12,288 이하에서 값을 제안 (초과하는 쌍이 있으면 멈추고 보고)
5. 학습 스크립트 `scripts/dpo_train.py` + `scripts/dpo_config.json`: E1 `train.py`·`train_config.json`을 복사해 `rlhf_type dpo`, PLAN_D1 4절 값으로. 나머지(LoRA 범위·seed·마스킹·template)는 E1과 같게
6. **멈추고 보고** (`reports/D1_STAGE1_REPORT.md`)

## 2단계: GPU 파일럿 (사용자가 A100/H100 80GB 파드를 빌린 뒤)

- 이미지 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, `/workspace/train_D1/`, `HF_HOME=/workspace/hf_cache`
- 환경: 새 venv `--system-site-packages`(기본 Torch 2.8.0+cu128 재사용). E3 학습과 같은 버전 목록(ms-swift 3.12.3, transformers 4.57.1, peft 0.17.1, trl, accelerate 1.11.0, qwen-vl-utils 0.0.14) + 병합 확인용 vLLM 0.11.0. **그 밖의 패키지·버전은 먼저 묻는다**. Torch 재설치 금지
- 데이터: 쌍 jsonl·스크립트는 scp, A 이미지는 HF v1.1(커밋 1833bac3), B/C 이미지는 scp. 모두 SHA-256 대조
- 파일럿 20 step (원본 M0에서 시작, 최장 쌍이 들어간 step을 반드시 포함)
  - step별: loss, reward margin, chosen/rejected logp(정책·기준), reward accuracy, grad norm, VRAM(allocated·reserved·nvidia-smi), step 시간
  - 학습 대상이 LM LoRA 252모듈뿐, visual 0, 기준 모델이 adapter 끈 M0인지
  - 실제 ms-swift 인자 전체 기록
- checkpoint-20 병합 → vLLM 0.11.0 로딩 (평가 config 그대로, 생성 평가 안 함)
- adapter·기록을 로컬 `adapter/d1_pilot_checkpoint-20/`, `reports/d1_pilot/`로 scp, SHA-256 대조
- 본학습 1·2 epoch 시간·비용 견적
- OOM·오류는 설정 바꾸지 말고 멈추고 보고
- **멈추고 보고** (`reports/D1_STAGE2_PILOT_REPORT.md`). 본학습은 사용자 확인 후

## 지킬 것

- MMMU / MMMU-Pro는 학습에 쓰지 않는다
- PLAN_D1 고정값(쌍 규칙, LoRA, 기준 모델, 마지막 체크포인트)과 사용자가 확정한 β·rpo_alpha·lr·epoch를 바꾸지 않는다. 바꿔야 하면 멈추고 보고
- 샘플링 raw·r1_train·reward_v2·평가 코드·E1 원본 스크립트 수정 금지 (복사해서 사용)
- GitHub·HF 업로드 없음. HF 토큰 출력·저장 금지
- 파드 Stop/Terminate 금지 (사용자가 함)
- 오래 걸리는 작업은 진행 상황을 짧게 알린다

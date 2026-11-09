# D1 HF 업로드 준비 보고

2026-11-07. 지침 `HANDOFF_D1_hf_upload.md` 1단계. **업로드는 하지 않았다.** 스테이징: `hf_staging/` (원본 수정·이동 없음).

## 저장소별 구성

| 저장소 (private, 브랜치 `run-yoonseok`) | 파일 수 | 크기 | 목록 |
|---|---:|---:|---|
| `MMDL-team4/hypB-d1-dpo` (model) | 19 | 128MB | `hf_staging/hypB-d1-dpo/SHA256SUMS.txt` |
| `MMDL-team4/hypB-rl-data` (dataset) | 422 | 61MB | `hf_staging/hypB-rl-data/SHA256SUMS.txt` |

### hypB-d1-dpo
- adapter: `adapter_model.safetensors`, `adapter_config.json`, `additional_config.json`, `args.json`, `trainer_state.json` (`training_args.bin`은 pickle이라 제외, 같은 내용은 `args.json`)
- `dpo_config.json`, `dpo_train.py`
- `logs/`: step_metrics.csv, logging.jsonl, train.log, console.log, args.json, merge.json, vllm_load.json, nvidia_smi.txt, gpu_memory.csv, timing.log, input_verification.json
- `README.md` (모델 카드)

### hypB-rl-data
- `data/`: r1_pool.jsonl, images_B/ (140), images_B_manifest.csv, d1_pairs_swift.jsonl, d1_pairs_index.jsonl
- `sampling_m0_k4/`: raw.jsonl, groups.json, r1_train.jsonl, manifest.json, SHA256SUMS.json, records/
- `scripts/`: sample_m0.py, reward.py, reward_v2.py, frozen_v2/, build_groups.py, build_dpo_pairs.py
- `README.md` (데이터 카드). A 이미지는 미포함(`hypB-train-data` v1.1 참조)

## 해시 대조

| 파일 | SHA-256 | 기준 기록 |
|---|---|---|
| adapter_model.safetensors | `2d4bcce4…6132e8` | D1 학습 보고서와 일치 |
| d1_pairs_swift.jsonl | `d6ae9fa2…4abbca` | D1 학습 보고서와 일치 |
| raw.jsonl / groups.json / r1_train.jsonl | `8ba0b1ed…` / `327f7464…` / `a3f94290…` | `runs/full_k4/SHA256SUMS.json`과 일치 |

## 검사

- 키·토큰 패턴(hf_, sk-, api_key 값): 없음. `frozen_v2/vendor/qwen_eval_utils.py`의 환경변수 이름 읽기만 걸림 (값 없음, 원본 Qwen 코드)
- 로컬 사용자 경로(`C:/Users`, AppData, 이메일), IP 주소: 없음. 파드 경로 `/workspace`, `/root`는 로그에 남아 있음 (민감 정보 아님)
- MMMU / MMMU-Pro: 문항·응답·평가 raw·Judge 기록·오염 검사 상세(STAGE1B) 미포함. "MMMU" 문자열은 `frozen_v2` 채점 코드와 공개 라이선스 파일에만 있음
- 라이선스: MMFR Apache-2.0 (개별 원천 데이터셋은 각자 라이선스 — 데이터 카드에 명시). 재배포 금지 조항 없음. 비공개 저장소
- `__pycache__` 제거

## 다음 (사용자 확인 후)

1. 사용자가 HF 로그인 (`MMDL-team4` 쓰기 권한 계정)
2. 저장소 2개를 private으로 생성(없으면) → private 확인 → `run-yoonseok` 브랜치 생성 → 업로드 → 원격 파일·SHA-256 대조 → private 재확인

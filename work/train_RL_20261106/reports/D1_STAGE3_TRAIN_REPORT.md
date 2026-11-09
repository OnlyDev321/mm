# D1 3단계 본학습 보고

**2 epoch, 90 optimizer step 완료.** 마지막 `checkpoint-90`을 bf16으로 병합했고 vLLM 0.11.0 로딩을 확인했다. 생성 요청은 0건이다. adapter와 실행 기록의 파드·로컬 SHA-256이 일치했다.

## 환경과 시작 상태

- 파일럿과 같은 NVIDIA A100 80GB PCIe 1장(81,920 MiB)에서 기존 학습·평가 venv를 사용했다. 학습 환경은 Torch 2.8.0+cu128, ms-swift 3.12.3, transformers 4.57.1, trl 0.22.2다.
- 캐시의 원본 M0 `ebb281ec70b05090aa6165b016eac8ec08e71b17`에서 시작했다. `adapters=[]`, `ref_adapters=[]`, `ref_model=null`, `resume_from_checkpoint=null`을 실제 인자에서 확인했다. 기준 점수는 어댑터를 끈 M0를 사용했다. 학습 대상은 LM LoRA 252모듈(504개 텐서), 시각 0모듈이다.
- `dpo_config.json` SHA-256 `30f901b04abd4f49f5adbf59fe47c346f2190ebbd8c321cbe4ef85484e72a421`. β 0.1, rpo_alpha 1.0, lr 5e-6, `max_length=10240`, 2 epoch를 유지했다. `max_steps=-1`은 epoch 기준 실행을 뜻한다. 학습 프로세스에만 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 적용했다.
- 707쌍 JSONL SHA-256 `d6ae9fa2a4b391ed533964beadb1f1ed4d887359df7f7fd3aa88c5433d4abbca`와 이미지 548개 SHA-256을 시작 전에 다시 대조했다.

## 학습 실측

| 항목 | 실측 |
|---|---:|
| 학습 커맨드 전체 | 2시간 54분 27초 |
| Trainer 구간 | 2시간 52분 37초 |
| Step 시간 평균 / 중앙값 / 최대 | 112.91 / 114.55 / 161.71초 |
| PyTorch 최대 allocated | 46.86 GiB |
| PyTorch 최대 reserved | 58.71 GiB |
| `nvidia-smi` 최대 GPU 사용량 | 61,067 MiB |
| 첫 → 마지막 step loss | 0.9931 → 0.6278 |
| 첫 → 마지막 step reward margin | 0.0000 → 0.9163 |
| 마지막 step reward accuracy / grad norm | 66.67% / 23.1010 |

총 1,414쌍을 처리했다. 최장 9,708토큰 쌍은 step 20과 65에 들어갔다. 모든 기록이 유한했고 OOM은 없었다. `nvidia-smi`는 약 1초 간격의 GPU 전체 사용량이다. 마지막 step은 3쌍이므로 다른 배치와 직접 비교하기 어렵다.

| Epoch | 처리 쌍 | 평균 step loss | 평균 step margin | 평균 step reward accuracy |
|---:|---:|---:|---:|---:|
| 1 | 707 | 0.9343 | 0.2163 | 56.06% |
| 2 | 707 | 0.8713 | 0.4601 | 65.93% |

step별 loss, margin, accuracy, 정책·기준 chosen/rejected logp, grad norm, chosen NLL, 소요 시간과 세 가지 VRAM 수치는 `d1_full/step_metrics.csv`에 기록했다. 실제 ms-swift 인자는 `d1_full/args.json`에 있다.

## 마지막 체크포인트와 로컬 보존

- `checkpoint-90` adapter SHA-256 `2d4bcce4972b36b6ef49b0b73d46c9d72a3803f43632b6847ae6f4d65e0132e8`. 다른 체크포인트를 사용하지 않았다.
- 고정 평가 config SHA-256 `3946c9a93cd5ab1cc5a8010fc00473e8a71f1c589e0a5f664edb70a3097234dc`로 병합 모델이 vLLM 0.11.0에서 140.90초에 초기화됐다. 병합 모델은 파드에 보존했다.

| 병합 샤드 | SHA-256 |
|---|---|
| `model-00001-of-00002.safetensors` | `31d4561504484269e16e1cc00baef16d694cd44f25c9d282cfd45977caaab351` |
| `model-00002-of-00002.safetensors` | `1280742e74baab97037603e9bc2912626f2c85d6347d9762a1286566b4b7f431` |

결과 압축 SHA-256 `152e64cf048eaabbd7693cb1f3c83d2becb241a974d861e8e66593a6cdcf4ca1`와 압축 내 파일 25개의 개별 SHA-256이 파드·로컬에서 일치했다. adapter는 `adapter/d1_full/`, 기록은 `reports/d1_full/`에 있다. 평가 생성·GitHub/HF 업로드는 없었다. 파드는 실행 상태로 두었다.

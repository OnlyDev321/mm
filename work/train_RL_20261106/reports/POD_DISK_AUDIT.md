# R1 2단계 전 파드 디스크 조사

사용자 지정 파드를 SSH로 조사했다. **로컬 SHA-256 대조가 끝난 삭제 후보는 20개 경로, 약 0.79 GiB**다. 보관이 확인되지 않은 큰 항목은 보류했다. 이 보고서에서 승인받을 대상은 아래 D 번호의 정확한 경로다.

## 현재 상태

- GPU는 NVIDIA GeForce RTX 4090 한 장, 24,564 MiB, driver 595.91.07이다. 조사 시 NVIDIA compute process는 표시되지 않았다.
- `/`는 overlay이고, `/workspace`는 별도 FUSE 네트워크 마운트다. `df`의 `/workspace` 수치는 공유 파일시스템 표시값이며 이 파드만의 볼륨 쿼터를 확인한 값은 아니다.
- `du -x`로 집계한 `/workspace`의 점유량은 128,224,118,784바이트(**119.42 GiB**)다. 인증 관련 파일과 Git 메타데이터는 집계에서 제외했다.

| 경로 | 파일시스템 | df -h 크기 | 사용 | 가용 | 사용률 |
|---|---|---:|---:|---:|---:|
| `/` | overlay | 50G | 16M | 50G | 1% |
| `/workspace` | 네트워크 FUSE | 2.1P | 1.7P | 425T | 81% |

### 큰 폴더 순위

| 경로 | 점유량 | 내용 |
|---|---:|---|
| `/workspace/train_E3_20261101` | 28.82 GiB | E3 학습. 병합 모델 두 벌, 체크포인트, 데이터, venv와 기록 |
| `/workspace/hypB_eval_20261101_baseline` | 24.88 GiB | HypB E1·E2·smoke 생성과 병합 모델 세 벌 |
| `/workspace/hypB-eval-env` | 16.39 GiB | 이전 HypB 평가 venv |
| `/workspace/hf_cache` | 12.46 GiB | 원본 Qwen, MMMU 데이터, HypB adapter 캐시 |
| `/workspace/cache` | 11.51 GiB | pip·uv 캐시, vLLM 컴파일·ModelScope 캐시 |
| `/workspace/eval_E3_20261101` | 8.80 GiB | E3 평가. 병합 모델 한 벌, adapter, raw, 기록, merge venv |
| `/workspace/pip_cache` | 6.59 GiB | pip HTTP 다운로드 및 wheel 캐시 |
| `/workspace/tmp` | 5.90 GiB | pip unpack·build·install 임시 파일 |
| `/workspace/mmmu_cache` | 3.60 GiB | MMMU ModelScope 데이터 캐시 |
| `/workspace/MMDL` | 263.73 MiB | HypB 평가에 사용한 코드 checkout 및 실험 자료 |
| `/workspace/.cache` | 107.60 MiB | 대부분 pip 캐시 |
| `/workspace/vllm_cache` | 100.99 MiB | 대부분 torch compile 캐시 |
| `/workspace/triton_cache` | 3.98 MiB | Triton 커널 캐시 |
| `/workspace/hypB_eval_20261101` | 3.05 MiB | HypB 최초 환경 준비·실패 기록 |
| `/workspace/hypB_eval_20261101_resume` | 1.91 MiB | HypB 재개 작업 기록 |

## 보존 항목과 R1 환경

- **원본 모델 캐시**는 `/workspace/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct` (8.28 GiB)다. 정확한 `ebb281ec70b05090aa6165b016eac8ec08e71b17` snapshot에서 두 가중치 링크가 해석됐고, 두 파일 합계는 8,875,719,344바이트다. tokenizer와 processor 구성도 확인했다.
- **이미지 기본 Python**은 `/usr/bin/python3.12`와 `/usr/local/bin/python`에서 Python 3.12.3 / Torch 2.8.0+cu128로 확인됐다. 이 파드의 현재 기본 환경에는 vLLM, Transformers, tokenizers, qwen-vl-utils 배포 메타데이터가 없다.
- `/workspace/train_E3_20261101/.venv-eval`은 vLLM 0.11.0 / Torch 2.8.0+cu128 / Transformers 4.57.1 / tokenizers 0.22.2다. 현재 `pip check`는 tqdm, click, hf-xet, httpx2 의존성 누락을 보고했다.
- `/workspace/hypB-eval-env`는 vLLM 0.11.0 / Torch 메타데이터 2.8.0 / Transformers 4.57.1 / tokenizers 0.22.2다. `pip check`는 통과했지만 평가 config의 Torch 메타데이터 고정값은 2.8.0+cu128이다. 두 venv를 보존 대상으로 분류했다. 실제 vLLM 로딩은 별도 단계다.

## 삭제 승인 후보

원격의 현재 파일을 SHA-256으로 읽어 로컬 파일을 다시 해싱했다. 디렉터리 후보는 해당 트리의 모든 정규 파일이 일치하고 심볼릭 링크가 없는 경우만 올렸다. **176개 파일에서 불일치 0**이었다. 후보 경로의 부모 폴더는 이 표의 승인 범위에 포함되지 않는다. [파일별 백업 대조](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/POD_DISK_BACKUP_CHECK.json) · [원격 해시·패키지 확인 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/POD_DISK_EVIDENCE.txt)

| ID | 파드 경로 | MiB | 내용 | 로컬 백업과 대조 근거 | 삭제 시 영향 |
|---|---|---:|---|---|---|
| D01 | `/workspace/train_E3_20261101/artifacts/full_20261101/checkpoint-188` | 128.06 | E3 본학습 checkpoint-188 | 7/7 SHA 일치. [work/train_E3_20261101/adapter/full](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_E3_20261101/adapter/full) | 재병합·재사용 시 로컬 체크포인트를 전송해야 함 |
| D02 | `/workspace/train_E3_20261101/artifacts/pilot_20261101/checkpoint-20` | 128.03 | E3 파일럿 checkpoint-20 | 7/7 SHA 일치. [work/train_E3_20261101/adapter/pilot_checkpoint-20](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_E3_20261101/adapter/pilot_checkpoint-20) | 파일럿 재사용 시 로컬 체크포인트를 전송해야 함 |
| D03 | `/workspace/eval_E3_20261101/adapter/full` | 131.90 | E3 평가에 복사된 adapter·부속 기록 | 48/48 SHA 일치. [work/train_E3_20261101/adapter/full](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_E3_20261101/adapter/full) | 같은 adapter의 로컬 사본에서 다시 복사 |
| D04 | `/workspace/eval_E3_20261101/runs` | 33.23 | E3 본실행·smoke raw 및 생성 기록 | 40/40 SHA 일치. [work/eval_E3_20261101/runs](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_E3_20261101/runs) | 파드에서 결과를 바로 읽을 때 로컬 사본 복원 |
| D05 | `/workspace/eval_E3_20261101/records` | 5.73 | E3 평가 게이트·실행·환경 기록 | 32/32 SHA 일치. [work/eval_E3_20261101/records](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_E3_20261101/records) | 파드 기록 경로에 의존하는 후속 명령은 로컬 사본 복원 |
| D06 | `/workspace/hypB_eval_20261101_baseline/E1/generation` | 10.86 | HypB E1 raw·생성 부속 파일 | 10/10 SHA 일치. [work/eval_B_20261101/runs/E1/generation](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_B_20261101/runs/E1/generation) | E1 파드 경로를 쓰는 재검토 시 로컬 사본 복원 |
| D07 | `/workspace/hypB_eval_20261101_baseline/E2/generation` | 3.92 | HypB E2 raw·생성 부속 파일 | 10/10 SHA 일치. [work/eval_B_20261101/runs/E2/generation](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_B_20261101/runs/E2/generation) | E2 파드 경로를 쓰는 재검토 시 로컬 사본 복원 |
| D08 | `/workspace/hypB_eval_20261101_baseline/smoke_E1/generation` | 2.04 | HypB E1 smoke raw·생성 부속 파일 | 10/10 SHA 일치. [work/eval_B_20261101/runs/smoke_E1/generation](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_B_20261101/runs/smoke_E1/generation) | smoke 파드 경로를 쓰는 재검토 시 로컬 사본 복원 |
| D09 | `/workspace/train_E3_20261101/full_results.tar.gz` | 117.18 | E3 본학습 결과 압축본 | 1/1 SHA 일치. [work/train_E3_20261101/reports/stage4/full_results.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_E3_20261101/reports/stage4/full_results.tar.gz) | 파드에 압축본이 필요하면 로컬에서 복사 |
| D10 | `/workspace/train_E3_20261101/stage3_pilot_results.tar.gz` | 117.38 | E3 파일럿 결과 압축본 | 1/1 SHA 일치. [work/train_E3_20261101/reports/pod_stage3/stage3_pilot_results.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_E3_20261101/reports/pod_stage3/stage3_pilot_results.tar.gz) | 파드에 압축본이 필요하면 로컬에서 복사 |
| D11 | `/workspace/train_E3_20261101/stage3_stopped_reports.tar.gz` | 0.01 | E3 중단 기록 압축본 | 1/1 SHA 일치. [work/train_E3_20261101/reports/pod_stage3/stage3_stopped_reports.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_E3_20261101/reports/pod_stage3/stage3_stopped_reports.tar.gz) | 파드에 압축본이 필요하면 로컬에서 복사 |
| D12 | `/workspace/train_E3_20261101/stage3_transfer.tar.gz` | 7.59 | E3 파일럿 입력 전송 압축본 | 1/1 SHA 일치. [work/train_E3_20261101/stage3_transfer.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_E3_20261101/stage3_transfer.tar.gz) | 해당 입력 재사용 시 로컬 압축본 복원 |
| D13 | `/workspace/eval_E3_20261101/generation_results.tar.gz` | 6.99 | E3 평가 결과 압축본 | 1/1 SHA 일치. [work/eval_E3_20261101/reports/transfer/generation_results.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_E3_20261101/reports/transfer/generation_results.tar.gz) | 파드에 결과 압축본이 필요하면 로컬에서 복사 |
| D14 | `/workspace/eval_E3_20261101/input/E3_adapter_full.tar.gz` | 117.18 | E3 평가용 adapter 입력 압축본 | 1/1 SHA 일치. [work/eval_E3_20261101/input/E3_adapter_full.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_E3_20261101/input/E3_adapter_full.tar.gz) | 병합 작업 재실행 시 로컬 압축본 복원 |
| D15 | `/workspace/eval_E3_20261101/input/source_bundle.tar.gz` | 0.15 | E3 평가용 코드 입력 압축본 | 1/1 SHA 일치. [work/eval_E3_20261101/input/source_bundle.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_E3_20261101/input/source_bundle.tar.gz) | 동일 코드 재배치 시 로컬 압축본 복원 |
| D16 | `/workspace/hypB_eval_20261101_baseline/common_records_20261101.tar.gz` | 0.01 | HypB 공통 설치·환경 기록 압축본 | 1/1 SHA 일치. [work/eval_B_20261101/runs/baseline_environment_20261101/common_records_20261101.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_B_20261101/runs/baseline_environment_20261101/common_records_20261101.tar.gz) | 환경 기록을 파드에서 읽으려면 복원 |
| D17 | `/workspace/hypB_eval_20261101_baseline/smoke_records_20261101.tar.gz` | 0.07 | HypB E1 smoke 기록 압축본 | 1/1 SHA 일치. [work/eval_B_20261101/runs/smoke_E1/smoke_records_20261101.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_B_20261101/runs/smoke_E1/smoke_records_20261101.tar.gz) | smoke 기록을 파드에서 읽으려면 복원 |
| D18 | `/workspace/hypB_eval_20261101/setup_records_20261101.tar.gz` | 0.02 | HypB 초기 setup 기록 압축본 | 1/1 SHA 일치. [work/eval_B_20261101/runs/setup_20261101/setup_records_20261101.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_B_20261101/runs/setup_20261101/setup_records_20261101.tar.gz) | 초기 기록을 파드에서 읽으려면 복원 |
| D19 | `/workspace/hypB_eval_20261101_baseline/E1_records_20261101.tar.gz` | 2.09 | HypB E1 평가 기록 압축본 | 1/1 SHA 일치. [work/eval_B_20261101/runs/E1/E1_records_20261101.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_B_20261101/runs/E1/E1_records_20261101.tar.gz) | 평가 기록을 파드에서 읽으려면 복원 |
| D20 | `/workspace/hypB_eval_20261101_baseline/E2_records_20261101.tar.gz` | 0.21 | HypB E2 평가 기록 압축본 | 1/1 SHA 일치. [work/eval_B_20261101/runs/E2/E2_records_20261101.tar.gz](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_B_20261101/runs/E2/E2_records_20261101.tar.gz) | 평가 기록을 파드에서 읽으려면 복원 |

기존 기록의 앵커 해시도 현재 양쪽 파일과 일치했다. E3 본학습 adapter `0ad919a7…`와 결과 압축 `223ed9c4…`는 Stage 4의 47파일 전송 기록, E3 파일럿 adapter `dda3cea3…`와 압축 `44f598ae…`는 Stage 3의 90파일 전송 기록과 일치한다. E3 평가 raw `cf0aba5b…`와 압축 `95febc0a…`는 평가 71파일 전송 기록과 일치한다. [E3 본학습 전송 검증](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_E3_20261101/reports/STAGE4_TRANSFER_CHECK.json) · [E3 파일럿 전송 검증](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_E3_20261101/reports/STAGE3_TRANSFER_CHECK.json) · [E3 평가 전송 검증](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_E3_20261101/reports/TRANSFER_CHECK.json)

HypB E1 raw `a6e307b8…`와 압축 `acac4223…`, E2 raw `0026e5e6…`와 압축 `275ebb13…` 역시 파드·로컬·당시 검증 기록이 일치한다. [E1 전송 검증](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_B_20261101/runs/E1/transfer_verification.json) · [E2 전송 검증](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_B_20261101/runs/E2/transfer_verification.json)

## 보류 항목

병합 모델은 기존 보고서에 가중치 조각의 SHA-256이 남아 있지만, 해당 실험의 로컬 보관 폴더에서 병합 가중치 파일 자체를 확인하지 못했다. E3의 merge 기록 및 복구에 필요한 adapter가 있어도 병합 모델 백업과는 구분했다. [E3 병합 해시 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_E3_20261101/reports/GENERATION_REPORT.md) · [HypB 병합 해시 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/hypB-train-git/assignment/experiments/hypB_eval/yoonseok/GENERATION_REPORT_20261101.md)

| 경로 | 점유량 | 내용·로컬 근거 | 삭제 영향과 상태 |
|---|---:|---|---|
| `/workspace/train_E3_20261101/artifacts/merged_pilot`, `merged_full_20261101` | 각 8.28 GiB | E3 파일럿·본학습 병합 가중치. adapter만 로컬 대조 완료 | E3 모델 재사용 시 재병합. 보류 |
| `/workspace/eval_E3_20261101/models/E3` | 8.28 GiB | E3 평가 병합 모델. 기존 2개 shard 해시 기록, 로컬 가중치 미확인 | E3 평가 재실행 시 재병합. 보류 |
| `/workspace/hypB_eval_20261101_baseline/{E1,E2,smoke_E1}/merged` | 각 8.28 GiB | HypB 병합 가중치. raw·기록 백업에 가중치 미포함 | E1/E2 재실행 시 재병합. 보류 |
| `/workspace/hypB-eval-env`, `/workspace/train_E3_20261101/.venv-eval` | 16.39 + 6.01 GiB | vLLM 0.11.0 환경. 패키지 목록은 보관됐으나 venv 전체 백업 미확인 | 평가 환경 복구 비용. 보존 |
| `/workspace/train_E3_20261101/.venv-train*`, `/workspace/eval_E3_20261101/merge_env` | 약 5.43 + 0.20 GiB | E3 훈련, 중단 설치 시도, 병합 venv. 환경 디렉터리 백업 미확인 | 학습·병합에 환경 재설치 필요. 보류 |
| `/workspace/cache`, `/workspace/pip_cache`, `/workspace/.cache/pip` | 11.51 + 6.59 + 0.10 GiB | pip·uv 다운로드, vLLM·Triton 컴파일, ModelScope 캐시. 로컬 캐시 백업 미확인 | 재설치·재실행 시 재다운로드·컴파일. 보류 |
| `/workspace/tmp` | 5.90 GiB | 이전 pip unpack·build·install 임시 디렉터리. 로컬 백업 미확인 | 미완료 설치 진단 자료 손실 가능. 보류 |
| `/workspace/mmmu_cache`, `/workspace/hf_cache/hub/datasets--MMMU--MMMU` | 3.60 + 3.41 GiB | MMMU 데이터 캐시. 로컬 평가 raw는 데이터 캐시 사본과 다름 | 향후 평가 시 데이터 재수신. 보류 |
| `/workspace/hf_cache/hub/models--MMDL-team4--hypB-*`, `datasets--MMDL-team4--hypB-train-data`, `hub/blobs` | adapter 각 0.13 GiB, 데이터 7.97 MiB, 공용 blobs 0.46 GiB | HypB adapter·데이터·공용 blob 캐시. 트리 전체 로컬 대조 미완료 | 후속 로딩·재병합에 재수신 가능성. 보류 |
| `/workspace/MMDL`, `/workspace/hypB_eval_20261101_resume` 및 D표 밖의 실험 기록·데이터·소스 | MMDL 263.73 MiB, 재개 폴더 1.91 MiB 및 기타 | 일부 코드·자료는 로컬에 있으나 현재 원격 트리 전체의 일치 미확인 | 작업 사본과 실행 증거 손실 가능. 보류 |

6개 병합 모델을 합치면 약 **49.68 GiB**지만 보류 항목이므로 확보 계산에서 제외했다. pip·uv 및 임시 디렉터리도 같은 기준을 적용했다.

## R1 100문항 시험 용량 비교

| 항목 | 용량 근거 |
|---|---|
| M0 모델 | 원본 가중치 약 8.27 GiB가 현재 캐시에 있다. 필요한 새 모델 복사량은 0이다. |
| 이미지·입력 | 시험 100장 5,764,434바이트. 전체 1,500장까지 전송하면 82,569,758바이트다. 풀과 시험 목록 JSONL은 약 1.70 MB다. |
| 생성 결과 | K=4, 400응답, 최대 3,276,800 출력 토큰. 이전 E3 raw의 바이트/토큰 비율로 보수적으로 잡으면 약 27 MB이며, 기록과 여유를 포함해 0.1 GiB를 예산으로 잡는다. |
| 새 작업 여유 | 기존 모델을 재사용하는 시험에 **0.5 GiB** 작업 여유를 잡았다. 별도로 고정 평가 config의 `min_free_disk_gb: 20`은 운영상 비교 기준이다. |
| 승인 후보 전부 삭제 시 | **852,090,880바이트 (812.62 MiB, 0.79 GiB)**의 디렉터리·파일 점유량이 대상이다. 다른 작업의 사용량 변동과 공유 볼륨 회수 동작에 따라 `df` 표시는 달라질 수 있다. |

공유 파일시스템의 표시 가용량을 기준으로는 시험 데이터 용량 때문에 정리가 필요한 상태는 아니다. 시험 시작 전에 해결할 항목은 현재 기본 Python의 vLLM 환경이다. 위 용량은 모델·이미지·생성 기록 기준이며 신규 환경 설치 용량을 포함하지 않는다.

조사는 여기서 멈췄다. 삭제 승인 범위는 위 **D01–D20 중 사용자가 지정한 경로**다.

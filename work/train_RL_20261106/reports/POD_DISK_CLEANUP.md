# R1 2단계 전 파드 디스크 정리

승인된 **27개 경로**의 삭제를 완료했다. `/workspace`의 `du` 점유량은 **119.42 → 63.03 GiB**, **56.39 GiB 감소**했다. 이 작업은 기존 디스크 정리 단계에서 종료했다. [사전 조사와 승인 후보](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/POD_DISK_AUDIT.md)

## 직전 확인과 삭제 목록

D01–D20, 사용자가 지정한 병합 모델 6개, `/workspace/tmp`를 정확한 절대 경로로 검사했다. 27개 모두 존재했고 `du -sx -B1` 점유량이 사전 조사와 일치했다. 각 경로는 삭제 직전에 크기·유형·해석 경로와 하위 마운트를 다시 확인했다. D01–D20은 기존 기록의 176개 파일을 로컬과 파드에서 다시 해싱해 대조한 뒤 삭제했다. 실행 결과는 27개 모두 종료 코드 0과 경로 부재로 확인됐다.

| ID | 승인 및 삭제 경로 | 삭제 직전 bytes | 결과 |
|---|---|---:|---|
| D01 | `/workspace/train_E3_20261101/artifacts/full_20261101/checkpoint-188` | 134,285,312 | 삭제 및 경로 부재 확인 |
| D02 | `/workspace/train_E3_20261101/artifacts/pilot_20261101/checkpoint-20` | 134,247,936 | 삭제 및 경로 부재 확인 |
| D03 | `/workspace/eval_E3_20261101/adapter/full` | 138,306,560 | 삭제 및 경로 부재 확인 |
| D04 | `/workspace/eval_E3_20261101/runs` | 34,840,064 | 삭제 및 경로 부재 확인 |
| D05 | `/workspace/eval_E3_20261101/records` | 6,004,224 | 삭제 및 경로 부재 확인 |
| D06 | `/workspace/hypB_eval_20261101_baseline/E1/generation` | 11,385,856 | 삭제 및 경로 부재 확인 |
| D07 | `/workspace/hypB_eval_20261101_baseline/E2/generation` | 4,113,408 | 삭제 및 경로 부재 확인 |
| D08 | `/workspace/hypB_eval_20261101_baseline/smoke_E1/generation` | 2,136,064 | 삭제 및 경로 부재 확인 |
| D09 | `/workspace/train_E3_20261101/full_results.tar.gz` | 122,869,760 | 삭제 및 경로 부재 확인 |
| D10 | `/workspace/train_E3_20261101/stage3_pilot_results.tar.gz` | 123,076,608 | 삭제 및 경로 부재 확인 |
| D11 | `/workspace/train_E3_20261101/stage3_stopped_reports.tar.gz` | 6,656 | 삭제 및 경로 부재 확인 |
| D12 | `/workspace/train_E3_20261101/stage3_transfer.tar.gz` | 7,958,016 | 삭제 및 경로 부재 확인 |
| D13 | `/workspace/eval_E3_20261101/generation_results.tar.gz` | 7,327,232 | 삭제 및 경로 부재 확인 |
| D14 | `/workspace/eval_E3_20261101/input/E3_adapter_full.tar.gz` | 122,869,760 | 삭제 및 경로 부재 확인 |
| D15 | `/workspace/eval_E3_20261101/input/source_bundle.tar.gz` | 152,064 | 삭제 및 경로 부재 확인 |
| D16 | `/workspace/hypB_eval_20261101_baseline/common_records_20261101.tar.gz` | 14,848 | 삭제 및 경로 부재 확인 |
| D17 | `/workspace/hypB_eval_20261101_baseline/smoke_records_20261101.tar.gz` | 70,656 | 삭제 및 경로 부재 확인 |
| D18 | `/workspace/hypB_eval_20261101/setup_records_20261101.tar.gz` | 16,384 | 삭제 및 경로 부재 확인 |
| D19 | `/workspace/hypB_eval_20261101_baseline/E1_records_20261101.tar.gz` | 2,187,776 | 삭제 및 경로 부재 확인 |
| D20 | `/workspace/hypB_eval_20261101_baseline/E2_records_20261101.tar.gz` | 221,696 | 삭제 및 경로 부재 확인 |
| M01 | `/workspace/train_E3_20261101/artifacts/merged_pilot` | 8,890,288,640 | 삭제 및 경로 부재 확인 |
| M02 | `/workspace/train_E3_20261101/artifacts/merged_full_20261101` | 8,890,288,640 | 삭제 및 경로 부재 확인 |
| M03 | `/workspace/eval_E3_20261101/models/E3` | 8,890,288,640 | 삭제 및 경로 부재 확인 |
| M04 | `/workspace/hypB_eval_20261101_baseline/E1/merged` | 8,890,288,640 | 삭제 및 경로 부재 확인 |
| M05 | `/workspace/hypB_eval_20261101_baseline/E2/merged` | 8,890,288,640 | 삭제 및 경로 부재 확인 |
| M06 | `/workspace/hypB_eval_20261101_baseline/smoke_E1/merged` | 8,890,288,640 | 삭제 및 경로 부재 확인 |
| T01 | `/workspace/tmp` | 6,334,660,096 | 삭제 및 경로 부재 확인 |

- D01–D20의 삭제 직전 점유량 합계는 852,090,880바이트다. 로컬 백업 위치와 파일별 SHA-256이 대조 기록에 있다. [백업 대조 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/POD_DISK_BACKUP_CHECK.json)
- 병합 모델 6개는 53,341,731,840바이트, `/workspace/tmp`는 6,334,660,096바이트다. 병합 모델 조각의 기존 해시는 각 실험 보고서에 있다.
- 경로별 점유량 합계는 **60,528,482,816바이트 (56.37 GiB)**다. 전체 디렉터리 전후 측정의 차이와는 15,838,208바이트 차이가 난다. 경로별 합산과 볼륨 트리 전체 재측정을 따로 기록했다.

## 삭제 후 RunPod 볼륨 점유량

`du -x -B1 --max-depth=1 /workspace`에 사전 조사와 같은 제외 조건을 적용했다. 아래 사용량은 이 파드에서 접근하는 `/workspace` 트리 기준이다.

| 경로 | 정리 전 bytes | 정리 후 bytes |
|---|---:|---:|
| **`/workspace` 전체** | **128,224,118,784** | **67,679,797,760** |
| `/workspace/hypB-eval-env` | 17,595,802,112 | 17,595,802,112 |
| `/workspace/hf_cache` | 13,376,876,032 | 13,376,876,032 |
| `/workspace/train_E3_20261101` | 30,948,700,160 | 12,644,656,128 |
| `/workspace/cache` | 12,356,889,088 | 12,356,889,088 |
| `/workspace/pip_cache` | 7,072,364,544 | 7,072,364,544 |
| `/workspace/tmp` | 6,334,660,096 | 삭제 |
| `/workspace/mmmu_cache` | 3,868,312,576 | 3,868,312,576 |
| `/workspace/MMDL` | 276,541,952 | 276,541,952 |
| `/workspace/eval_E3_20261101` | 9,448,291,328 | 240,478,720 |
| `/workspace/hypB_eval_20261101_baseline` | 26,714,580,992 | 16,800,256 |
| `/workspace/.cache` | 112,823,808 | 112,823,808 |

정리 직후 `df -h / /workspace` 결과:

```text
Filesystem                  Size  Used Avail Use% Mounted on
overlay                      50G   16M   50G   1% /
mfs#euro-2.runpod.net:9421  2.1P  1.7P  425T  81% /workspace
```

| 마운트 | 정리 전 `df -B1` 가용 bytes | 정리 후 `df -B1` 가용 bytes |
|---|---:|---:|
| `/` | 53,670,469,632 | 53,670,469,632 |
| `/workspace` | 466,352,339,681,280 | 466,361,784,729,600 |

`/workspace`의 `df`는 공유 네트워크 파일시스템의 사용량이며 다른 작업의 변동이 포함된다. 이 정리의 감소량은 위의 `/workspace` 전후 `du` 차이로 보고했다.

원본 모델을 담은 `/workspace/hf_cache`와 `/workspace/hypB-eval-env`는 전후 점유량이 같다. R1 2단계는 사용자 확인 대기 상태다.

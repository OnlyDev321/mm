# R1 2단계 M0 시험 샘플링

2026-11-06 KST. PLAN 3-2절의 고정 시험 100문항에서 **400응답을 생성하고 파드·로컬 해시 대조까지 완료**했다. 표적 25건, 유지 26건, 제외 49건이다. RTX 4090 한 장에서 생성 구간은 **17분 40초**, 엔진 초기화·종료를 포함한 실행은 **19분 21초**였다.

## 1. 설치와 입력 검증

같은 파드의 이미지 기본 Python 3.12.3을 사용했다. 가설 B 평가에서 사용한 main `requirements.txt` SHA-256은 `0562e7266d8eac3fe4b3f0cab3906a748762a3164b1a0d4ddee23d5ee995a04b`이고, `run_generate.sh` 설치 블록을 바이트 그대로 사용했다. 첫 호출은 PEP 668의 externally managed 환경에서 거부됐고, 원래 블록의 `--break-system-packages` 재호출로 성공했다. 기존 pip 캐시의 사본과 기존 uv 캐시에서 찾은 wheel 사본을 새 작업 폴더에 두고 사용했다.

Torch는 사전 pip 설치 계획에서 제외됐다. 설치 전후 메타데이터·런타임이 모두 **2.8.0+cu128**, CUDA는 12.8이었다. Torch RECORD SHA-256과 모듈·코어 파일 경로 및 수정 시각도 같았다. pip 설치 기록에 Torch 제거가 없었고, `pip check`는 **No broken requirements found.**를 반환했다. [설치 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/records/01_install.log) · [Torch 설치 전](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/records/torch_before.json) · [Torch 설치 후](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/records/torch_after.json)

| 패키지 | 설치 후 버전 |
|---|---|
| torch | `2.8.0+cu128` |
| vllm | `0.11.0` |
| transformers | `4.57.1` |
| tokenizers | `0.22.2` |
| qwen-vl-utils | `0.0.14` |
| huggingface-hub | `0.35.3` |
| datasets | `4.0.0` |
| safetensors | `0.6.2` |
| pillow | `11.3.0` |
| numpy | `2.2.6` |
| xformers | `0.0.32.post1` |

[전체 패키지 버전](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/records/packages.json)

SCP로 코드·풀·고정 시험 목록·B/C 이미지 140개를 포함한 **163개 파일**을 전송하고 각각 SHA-256을 확인했다. 입력 압축 SHA-256은 `765aeaa239617c5dc5a70622525ee6fbac79475b98e8bfdc61ee4296a982c52f`이다. A 88개 이미지는 HF `MMDL-team4/hypB-train-data` v1.1 태그가 `1833bac3be047aa8bd146492436323bb7c14092a`를 가리키는지 확인했다. 같은 커밋의 기존 캐시에서 88개를 찾아 복사했고, 각 크기와 SHA-256이 image_manifest에 일치했다. A 이미지 합계는 5,584,611바이트다. [SCP 입력 검증](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/records/input_verification.json) · [A 이미지 검증](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/records/A_image_verification.json)

## 2. 생성 결과

M0 `Qwen/Qwen3-VL-4B-Instruct` revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`의 기존 가중치 캐시를 재사용했다. 평가 config SHA-256 `3946c9a93cd5ab1cc5a8010fc00473e8a71f1c589e0a5f664edb70a3097234dc`로 bf16, context 16,384, max_new_tokens 8,192, temperature 0.7, top_p 0.8, top_k 20, repetition penalty 1.0, presence penalty 1.5를 적용했다. 엔진 seed 3407과 고정 시험의 400개 요청 seed를 대조했다. 100개 ID 각각에 회차 0–3이 하나씩 있고 프롬프트 토큰 수가 고정 풀과 일치한다. [실행 manifest](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/manifest.json) · [생성 실행 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/records/sampling.log)

| 항목 | 실측 |
|---|---:|
| GPU | RTX 4090 1장, 24,564 MiB |
| 문항 / 응답 | 100 / 400 |
| 생성 구간 | 1,060.4초 (17분 40초) |
| 엔진 초기화·종료 포함 | 1161.2초 (19분 21초) |
| 출력 토큰 합계 | 782,961 |
| 출력 토큰 평균 / 중앙값 | 1957.40 / 695 |
| 출력 토큰 최소 / 최대 | 5 / 8,192 |
| length 종료 | **35/400 (8.75%)** |
| 최대 GPU 사용 메모리 | **22,086 MiB (21.57 GiB)** |
| nvidia-smi 표본 수 | 1,124 |

VRAM은 약 1초 간격의 GPU 전체 사용량 실측 최대치다. 생성 구간은 엔진 로딩 이후 샘플러의 입력 구성·생성·raw 기록을 포함한다. 모델과 엔진 초기화·종료의 추가 시간은 약 100.8초였다. [GPU 메모리 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/records/gpu_memory.csv)

| 전체 샘플링 | 응답 수 | 동일 생성 속도 외삽 | 초기화·종료 1회 포함 |
|---|---:|---:|---:|
| 1,500 × K=4 | 6,000 | 4시간 25분 6초 | **4시간 26분 47초** |
| 1,500 × K=8 | 12,000 | 8시간 50분 12초 | **8시간 51분 53초** |

시험의 평균 출력 길이와 처리량이 전체에서 유지된다는 가정이다. 설치, 파일 전송, 이미지 준비 시간은 별도다.

## 3. 보상과 그룹 집계

| 보상 이유 | 응답 수 | 비율 |
|---|---:|---:|
| `correct` | 197 | 49.25% |
| `wrong` | 50 | 12.50% |
| `extract_fail` | 118 | 29.50% |
| `truncated` | 35 | 8.75% |
| 합계 | 400 | 100.00% |

문항당 정답 수 0/1/2/3/4의 분포는 **24 / 22 / 12 / 17 / 25건**이다. 문항당 length 수 0/1/2/3/4는 **81 / 9 / 6 / 2 / 2건**이다. 정답 197개와 잘림 35개는 문항별 합계와도 일치했다.

기존 `build_groups.py`를 사용했다. 혼합 보상 그룹에서 잘림 또는 마커 21회 이상이 한 번이라도 있으면 표적, 나머지 혼합 보상은 유지다. 제외 49건은 네 응답 모두 오답 24건과 모두 정답 25건이다. 전부 정답이면서 마커 기준에도 걸리는 사례는 0건이었다. 로컬에서 raw의 보상, 마커 수와 그룹을 다시 계산해 파드 결과와 일치함을 확인했다. [build_groups.py 결과](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/groups.json)

| 분야 | 표적 | 유지 | 제외 | 합계 |
|---|---:|---:|---:|---:|
| Art & Design | 1 | 0 | 3 | 4 |
| Business | 5 | 5 | 8 | 18 |
| Health & Medicine | 6 | 6 | 10 | 22 |
| Humanities & Social Science | 1 | 3 | 9 | 13 |
| Science | 11 | 4 | 7 | 22 |
| Tech & Engineering | 1 | 8 | 12 | 21 |
| **전체** | **25** | **26** | **49** | **100** |

| 출처 | 표적 | 유지 | 제외 | 합계 |
|---|---:|---:|---:|---:|
| A | 25 | 22 | 41 | 88 |
| B | 0 | 4 | 8 | 12 |

이 시험의 전체 비율을 1,500건에 그대로 적용하면 **표적 약 375건, 유지 약 390건, 제외 약 735건**이다. 시험은 분야 층화 표본이며 C 출처는 표본에 없었다.

### B 순서도 12문항

기존 순서도 검출 기록으로 B 시험 12건 모두를 확인했다. **표적 0건, 유지 4건, 제외 8건**이다. 제외 8건은 모두 4/4 정답이었다. B의 48응답에서 length는 0건이고 마커 최댓값은 5였다. [기존 B 순서도 검출 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_cs_flowchart.json)

| row | 정답 수 | length 수 | 마커 최대 | 그룹 |
|---|---:|---:|---:|---|
| 10900 | 3 | 0 | 0 | 유지 |
| 33892 | 4 | 0 | 2 | 제외 |
| 72 | 4 | 0 | 0 | 제외 |
| 11247 | 4 | 0 | 5 | 제외 |
| 34486 | 4 | 0 | 0 | 제외 |
| 12267 | 3 | 0 | 0 | 유지 |
| 16106 | 4 | 0 | 0 | 제외 |
| 18469 | 3 | 0 | 0 | 유지 |
| 16602 | 3 | 0 | 0 | 유지 |
| 7687 | 4 | 0 | 0 | 제외 |
| 27370 | 4 | 0 | 5 | 제외 |
| 32710 | 4 | 0 | 0 | 제외 |

## 4. 로컬 전송과 검증

파드 결과 압축을 `work/train_RL_20261106/runs/trial_100/`에 SCP로 받았다. 원격·로컬 압축 SHA-256은 `c797c8c8b0e034b18213feb77272520cd9df8067471508c2fe76052d5a31b036`로 같고 압축 내 17개 파일의 개별 해시가 일치했다. 원격·로컬 raw SHA-256은 모두 `d3a92b5359e5d998069482fe69915b47d415567a269bb06788dfd48cba8b81f5`이다. [raw.jsonl](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/raw.jsonl) · [전송 검증](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/transfer_verification.json)

R1 2단계 시험과 로컬 검증까지 마쳤다. 전체 샘플링은 사용자 확인 대기 상태다.

## 부록. 시험 100문항 개별 결과

표 순서는 고정된 시험 목록과 같다. `마커 최대`는 각 문항의 네 응답 중 최대 방향 전환 마커 수다. [문항별 CSV](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/question_results.csv) · [집계 JSON](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/runs/trial_100/trial_statistics.json)

| 순서 | MMFR row | 출처 | 과목 | 정답 수 | length 수 | 마커 최대 | 그룹 |
|---:|---|---|---|---:|---:|---:|---|
| 1 | 577055 | A | Music | 4 | 0 | 0 | 제외 |
| 2 | 940096 | A | History | 4 | 0 | 0 | 제외 |
| 3 | 1639138 | A | Basic_Medical_Science | 3 | 0 | 8 | 유지 |
| 4 | 768447 | A | Economics | 1 | 1 | 105 | 표적 |
| 5 | 1589281 | A | Basic_Medical_Science | 1 | 0 | 73 | 표적 |
| 6 | 1606522 | A | Clinical_Medicine | 0 | 0 | 1 | 제외 |
| 7 | 1721870 | A | Economics | 0 | 2 | 133 | 제외 |
| 8 | 763905 | A | Economics | 0 | 1 | 99 | 제외 |
| 9 | 1054292 | A | Computer_Science | 2 | 2 | 91 | 표적 |
| 10 | 10900 | B | Computer_Science | 3 | 0 | 0 | 유지 |
| 11 | 684295 | A | Economics | 1 | 0 | 0 | 유지 |
| 12 | 160799 | A | History | 0 | 0 | 0 | 제외 |
| 13 | 1213301 | A | Physics | 3 | 0 | 14 | 유지 |
| 14 | 33892 | B | Computer_Science | 4 | 0 | 2 | 제외 |
| 15 | 769105 | A | Math | 2 | 0 | 24 | 표적 |
| 16 | 608806 | A | Economics | 3 | 0 | 67 | 표적 |
| 17 | 1012609 | A | Math | 1 | 0 | 83 | 표적 |
| 18 | 497692 | A | History | 0 | 0 | 0 | 제외 |
| 19 | 80834 | A | Basic_Medical_Science | 1 | 0 | 48 | 표적 |
| 20 | 522395 | A | Physics | 1 | 0 | 23 | 표적 |
| 21 | 1580114 | A | History | 2 | 0 | 0 | 유지 |
| 22 | 57041 | A | Math | 2 | 0 | 28 | 표적 |
| 23 | 1597410 | A | Economics | 0 | 1 | 174 | 제외 |
| 24 | 288752 | A | Music | 4 | 0 | 0 | 제외 |
| 25 | 869601 | A | Math | 1 | 1 | 42 | 표적 |
| 26 | 150552 | A | Economics | 0 | 0 | 2 | 제외 |
| 27 | 909594 | A | Clinical_Medicine | 1 | 0 | 3 | 유지 |
| 28 | 1532848 | A | Clinical_Medicine | 0 | 4 | 106 | 제외 |
| 29 | 481458 | A | History | 0 | 0 | 0 | 제외 |
| 30 | 1182614 | A | Economics | 1 | 0 | 8 | 유지 |
| 31 | 102164 | A | Computer_Science | 3 | 0 | 0 | 유지 |
| 32 | 1417741 | A | Math | 2 | 1 | 33 | 표적 |
| 33 | 72 | B | Computer_Science | 4 | 0 | 0 | 제외 |
| 34 | 443010 | A | Economics | 4 | 0 | 0 | 제외 |
| 35 | 344397 | A | Diagnostics_and_Laboratory_Medicine | 1 | 0 | 0 | 유지 |
| 36 | 742575 | A | Public_Health | 4 | 0 | 0 | 제외 |
| 37 | 65614 | A | Diagnostics_and_Laboratory_Medicine | 2 | 0 | 65 | 표적 |
| 38 | 473628 | A | Computer_Science | 4 | 0 | 0 | 제외 |
| 39 | 360248 | A | Economics | 1 | 0 | 0 | 유지 |
| 40 | 57570 | A | Clinical_Medicine | 0 | 4 | 142 | 제외 |
| 41 | 41052 | A | Chemistry | 2 | 1 | 132 | 표적 |
| 42 | 67810 | A | Basic_Medical_Science | 0 | 0 | 38 | 제외 |
| 43 | 78559 | A | History | 0 | 0 | 20 | 제외 |
| 44 | 426465 | A | Math | 3 | 0 | 0 | 유지 |
| 45 | 629323 | A | Math | 1 | 0 | 40 | 표적 |
| 46 | 475657 | A | Finance | 0 | 0 | 0 | 제외 |
| 47 | 774997 | A | History | 2 | 0 | 1 | 유지 |
| 48 | 856852 | A | Computer_Science | 3 | 0 | 0 | 유지 |
| 49 | 502992 | A | Basic_Medical_Science | 1 | 0 | 58 | 표적 |
| 50 | 232801 | A | Computer_Science | 4 | 0 | 0 | 제외 |
| 51 | 144512 | A | Economics | 0 | 0 | 102 | 제외 |
| 52 | 44289 | A | History | 1 | 0 | 32 | 표적 |
| 53 | 11247 | B | Computer_Science | 4 | 0 | 5 | 제외 |
| 54 | 1145992 | A | Economics | 3 | 0 | 77 | 표적 |
| 55 | 823720 | A | Economics | 1 | 0 | 32 | 표적 |
| 56 | 345525 | A | Math | 1 | 1 | 24 | 표적 |
| 57 | 72742 | A | Computer_Science | 4 | 0 | 8 | 제외 |
| 58 | 1364550 | A | Basic_Medical_Science | 4 | 0 | 0 | 제외 |
| 59 | 34486 | B | Computer_Science | 4 | 0 | 0 | 제외 |
| 60 | 839453 | A | History | 3 | 0 | 0 | 유지 |
| 61 | 659720 | A | Design | 4 | 0 | 0 | 제외 |
| 62 | 1043321 | A | Computer_Science | 4 | 0 | 0 | 제외 |
| 63 | 12267 | B | Computer_Science | 3 | 0 | 0 | 유지 |
| 64 | 16106 | B | Computer_Science | 4 | 0 | 0 | 제외 |
| 65 | 417333 | A | Math | 2 | 0 | 9 | 유지 |
| 66 | 1718702 | A | Math | 0 | 0 | 73 | 제외 |
| 67 | 1387799 | A | History | 0 | 2 | 335 | 제외 |
| 68 | 401081 | A | Math | 0 | 0 | 2 | 제외 |
| 69 | 43568 | A | Physics | 0 | 2 | 133 | 제외 |
| 70 | 294586 | A | Basic_Medical_Science | 4 | 0 | 3 | 제외 |
| 71 | 556415 | A | Economics | 1 | 0 | 4 | 유지 |
| 72 | 484440 | A | Economics | 3 | 0 | 0 | 유지 |
| 73 | 1379570 | A | Basic_Medical_Science | 4 | 0 | 12 | 제외 |
| 74 | 485655 | A | Computer_Science | 2 | 0 | 6 | 유지 |
| 75 | 801661 | A | Pharmacy | 4 | 0 | 1 | 제외 |
| 76 | 18469 | B | Computer_Science | 3 | 0 | 0 | 유지 |
| 77 | 16602 | B | Computer_Science | 3 | 0 | 0 | 유지 |
| 78 | 197926 | A | Math | 3 | 0 | 36 | 표적 |
| 79 | 1648474 | A | Physics | 4 | 0 | 2 | 제외 |
| 80 | 1700762 | A | Biology | 0 | 2 | 114 | 제외 |
| 81 | 376944 | A | Computer_Science | 3 | 0 | 2 | 유지 |
| 82 | 398695 | A | Physics | 0 | 0 | 98 | 제외 |
| 83 | 1584288 | A | Clinical_Medicine | 2 | 2 | 110 | 표적 |
| 84 | 616081 | A | Basic_Medical_Science | 0 | 1 | 130 | 제외 |
| 85 | 129734 | A | Music | 2 | 1 | 195 | 표적 |
| 86 | 1667385 | A | Clinical_Medicine | 1 | 3 | 121 | 표적 |
| 87 | 955732 | A | Economics | 4 | 0 | 0 | 제외 |
| 88 | 7687 | B | Computer_Science | 4 | 0 | 0 | 제외 |
| 89 | 27370 | B | Computer_Science | 4 | 0 | 5 | 제외 |
| 90 | 666776 | A | History | 0 | 0 | 0 | 제외 |
| 91 | 32710 | B | Computer_Science | 4 | 0 | 0 | 제외 |
| 92 | 1511334 | A | Math | 1 | 0 | 5 | 유지 |
| 93 | 140407 | A | Economics | 3 | 0 | 122 | 표적 |
| 94 | 1775294 | A | Math | 1 | 3 | 111 | 표적 |
| 95 | 182814 | A | Clinical_Medicine | 1 | 0 | 19 | 유지 |
| 96 | 1311317 | A | Clinical_Medicine | 3 | 0 | 11 | 유지 |
| 97 | 1602307 | A | Diagnostics_and_Laboratory_Medicine | 1 | 0 | 0 | 유지 |
| 98 | 1349827 | A | History | 4 | 0 | 2 | 제외 |
| 99 | 119907 | A | History | 0 | 0 | 3 | 제외 |
| 100 | 143434 | A | Biology | 0 | 0 | 70 | 제외 |

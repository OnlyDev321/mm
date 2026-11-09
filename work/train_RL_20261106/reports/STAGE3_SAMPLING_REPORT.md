# R1 3단계 M0 전체 샘플링

1,500문항 × K=4, **6,000응답**을 새 run에서 생성했다. 파드와 로컬에서 해시를 대조하고 기존 그룹 기준으로 **표적 322 / 유지 226 / 제외 952**를 확인했다. 혼합 보상 문항 **548건**을 seed 3407로 섞어 학습 세트를 만들었다. 표적:유지 실측 비율은 **322:226**다.

## 입력과 환경

- M0 `Qwen/Qwen3-VL-4B-Instruct` `ebb281ec70b05090aa6165b016eac8ec08e71b17`의 기존 모델 캐시를 사용했다. 데이터 A는 v1.1 `1833bac3be047aa8bd146492436323bb7c14092a`의 manifest와 대조했고, B/C는 이전에 전송한 이미지와 대조했다. 검증한 이미지 수는 A 1,360개, B 139개, C 1개다. 기존 이미지 트리 228개, 고정 캐시 1,272개, 새 다운로드 0개다.
- 평가 config SHA-256 `3946c9a93cd5ab1cc5a8010fc00473e8a71f1c589e0a5f664edb70a3097234dc`를 사용했다. bf16, presence penalty 1.5, max_new_tokens 8,192, temperature 0.7, top_p 0.8, top_k 20이다. 보강 보상 SHA-256은 `ba71a13013bdd3244d972d3d2d45edf4bf22abdccc8eb80163d28f7e64263b82`다. 1,500문항의 요청 시드와 프롬프트 토큰을 고정 풀과 대조했다.

```text
2026-11-06T07:39:42+00:00
Python 3.12.3
vLLM 0.11.0 Torch metadata 2.8.0+cu128 Torch runtime 2.8.0+cu128 CUDA 12.8
name, memory.total [MiB], memory.used [MiB], driver_version
NVIDIA GeForce RTX 4090, 24564 MiB, 1 MiB, 595.91.07
No broken requirements found.
```

## 생성과 보상

| 항목 | 실측 |
|---|---:|
| 생성 구간(입력 구성·생성·raw 기록) | 3시간 22분 34초 (12,153.9초) |
| 엔진 초기화·종료 포함 | 3시간 24분 4초 |
| 출력 토큰 합계 | 9,847,792 |
| 출력 토큰 평균 / 중앙값 | 1,641.30 / 640.0 |
| length 종료 | 357/6,000 (5.95%) |
| 최대 GPU 사용 메모리 | 22,671 MiB (22.14 GiB) |
| nvidia-smi 1초 표본 | 12,170 |

VRAM은 `nvidia-smi`의 GPU 전체 사용량 표본 최대치다.

| 보상 이유 | 응답 수 |
|---|---:|
| `correct` | 3,136 |
| `wrong` | 785 |
| `fallback_correct` | 1,229 |
| `fallback_wrong` | 426 |
| `extract_fail` | 67 |
| `truncated` | 357 |

| 문항당 정답 수 | 문항 수 |
|---:|---:|
| 0 | 162 |
| 1 | 140 |
| 2 | 159 |
| 3 | 249 |
| 4 | 790 |

## 그룹과 학습 세트

### 분야별

| 분야 | 표적 | 유지 | 제외 | 학습 세트 | 전체 |
|---|---:|---:|---:|---:|---:|
| Art & Design | 2 | 1 | 58 | 3 | 61 |
| Business | 93 | 37 | 144 | 130 | 274 |
| Health & Medicine | 71 | 66 | 190 | 137 | 327 |
| Humanities & Social Science | 16 | 46 | 136 | 62 | 198 |
| Science | 116 | 42 | 164 | 158 | 322 |
| Tech & Engineering | 24 | 34 | 260 | 58 | 318 |

### 출처별

| 출처 | 표적 | 유지 | 제외 | 학습 세트 | 전체 |
|---|---:|---:|---:|---:|---:|
| A | 318 | 219 | 823 | 537 | 1360 |
| B | 4 | 7 | 128 | 11 | 139 |
| C | 0 | 0 | 1 | 0 | 1 |

### 과목별

| 분야 / 과목 | 표적 | 유지 | 제외 | 학습 세트 | 전체 |
|---|---:|---:|---:|---:|---:|
| Art & Design / Art | 0 | 0 | 7 | 0 | 7 |
| Art & Design / Art_Theory | 1 | 1 | 3 | 2 | 5 |
| Art & Design / Design | 0 | 0 | 25 | 0 | 25 |
| Art & Design / Music | 1 | 0 | 23 | 1 | 24 |
| Business / Accounting | 2 | 0 | 6 | 2 | 8 |
| Business / Economics | 88 | 36 | 126 | 124 | 250 |
| Business / Finance | 1 | 0 | 3 | 1 | 4 |
| Business / Manage | 1 | 0 | 4 | 1 | 5 |
| Business / Marketing | 1 | 1 | 5 | 2 | 7 |
| Health & Medicine / Basic_Medical_Science | 40 | 23 | 70 | 63 | 133 |
| Health & Medicine / Clinical_Medicine | 13 | 3 | 53 | 16 | 69 |
| Health & Medicine / Diagnostics_and_Laboratory_Medicine | 10 | 33 | 41 | 43 | 84 |
| Health & Medicine / Pharmacy | 3 | 4 | 17 | 7 | 24 |
| Health & Medicine / Public_Health | 5 | 3 | 9 | 8 | 17 |
| Humanities & Social Science / History | 15 | 46 | 130 | 61 | 191 |
| Humanities & Social Science / Literature | 0 | 0 | 4 | 0 | 4 |
| Humanities & Social Science / Psychology | 0 | 0 | 1 | 0 | 1 |
| Humanities & Social Science / Sociology | 1 | 0 | 1 | 1 | 2 |
| Science / Biology | 10 | 9 | 25 | 19 | 44 |
| Science / Chemistry | 7 | 0 | 9 | 7 | 16 |
| Science / Geography | 9 | 5 | 11 | 14 | 25 |
| Science / Math | 74 | 22 | 96 | 96 | 192 |
| Science / Physics | 16 | 6 | 23 | 22 | 45 |
| Tech & Engineering / Agriculture | 0 | 1 | 1 | 1 | 2 |
| Tech & Engineering / Architecture_and_Engineering | 0 | 0 | 2 | 0 | 2 |
| Tech & Engineering / Computer_Science | 17 | 30 | 247 | 47 | 294 |
| Tech & Engineering / Electronics | 4 | 3 | 2 | 7 | 9 |
| Tech & Engineering / Energy_and_Power | 1 | 0 | 4 | 1 | 5 |
| Tech & Engineering / Materials | 0 | 0 | 1 | 0 | 1 |
| Tech & Engineering / Mechanical_Engineering | 2 | 0 | 3 | 2 | 5 |

## 시험 100문항과 일치 정도

시험 재채점과 전체 run에서 **79/100문항**의 그룹이 같았다. 시험 응답 400건을 행·회차로 짝지어 요청 시드와 프롬프트 토큰이 모두 같은지 확인했다. 응답 원문은 36/400건, 이진 보상은 342/400건이 일치했다.

| 시험 재채점 → 전체 | 표적 | 유지 | 제외 |
|---|---:|---:|---:|
| 표적 | 16 | 1 | 9 |
| 유지 | 1 | 3 | 2 |
| 제외 | 6 | 2 | 60 |

그룹이 달라진 풀 행 ID는 `768447`, `1589281`, `1721870`, `1012609`, `57041`, `869601`, `1417741`, `344397`, `65614`, `67810`, `856852`, `502992`, `144512`, `1145992`, `417333`, `1718702`, `401081`, `1700762`, `398695`, `1584288`, `1511334`이다.

## 로컬 전송

파드·로컬 압축 SHA-256 `77adb927203a69578590d76a0949242f923fd4adf85a480efe29fa2920bbf95c` 및 압축 내 파일 14개의 개별 SHA-256이 일치했다. 결과는 `runs/full_k4/`에 있다.

| 파일 | SHA-256 |
|---|---|
| `raw.jsonl` | `8ba0b1edfa9b2bb5c58fad9b96b2f320bd5b35789466aa3b4ded6d98d33a2d2e` |
| `manifest.json` | `3fdfba071d91c53a8c96c06eab4683aed7065d070c3dfc4ce92f4d395124ff7a` |
| `groups.json` | `327f74646280b68edaa389200af5643b22cc8e86b021af756db953572a9a6434` |
| `r1_train.jsonl` | `a3f94290caf2d2bbc0b0b3beea26bd83e0b994b78b572ea00691e8df400a1017` |

GRPO 파일럿은 사용자 확인 대기 상태다. 파드는 실행 상태로 남겼다.

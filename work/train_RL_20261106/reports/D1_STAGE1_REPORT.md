# D1 1단계 DPO 쌍 구성

2026-11-06 KST. CPU에서 **548문항의 707쌍**을 구성하고 검증했다. 최장 시퀀스는 **9,708토큰**이며 12,288 초과는 0쌍이다. `max_length=10240`으로 학습 config를 작성했다. GPU 파일럿은 실행하지 않았다.

## 입력 대조와 쌍 구성

`runs/full_k4/SHA256SUMS.json`과 입력 세 파일의 SHA-256을 작업 전에 대조했고, 검증 후에도 같은 해시를 확인했다.

| 입력 | SHA-256 |
|---|---|
| `raw.jsonl` | `8ba0b1edfa9b2bb5c58fad9b96b2f320bd5b35789466aa3b4ded6d98d33a2d2e` |
| `groups.json` | `327f74646280b68edaa389200af5643b22cc8e86b021af756db953572a9a6434` |
| `r1_train.jsonl` | `a3f94290caf2d2bbc0b0b3beea26bd83e0b994b78b572ea00691e8df400a1017` |

학습 세트 순서로 처리했다. 문항마다 회차 순으로 후보를 만들고 동일 보상 안의 원문 중복을 제거한 뒤 `random.Random(3407)`로 섞었다. rejected는 잘림 → 정상 종료 오답 → 추출 실패로 안정 정렬하고, 같은 종류 안에서는 섞은 순서를 사용했다. chosen과 rejected를 각각 한 번씩만 사용해 문항당 최대 2쌍을 만들었다. 389문항에서 1쌍, 159문항에서 2쌍이 나왔다. 원문이 중복된 chosen 후보가 한 문항에 있었지만, 그 문항은 rejected가 하나여서 쌍 수는 그대로였다.

| 그룹 | 문항 | 쌍 | rejected 오답 | rejected 잘림 | rejected 추출 실패 |
|---|---:|---:|---:|---:|---:|
| 표적 | 322 | 427 | 225 | 190 | 12 |
| 유지 | 226 | 280 | 250 | 0 | 30 |
| **전체** | **548** | **707** | **475** | **190** | **42** |

예상 쌍 수와 rejected 분포가 모두 일치했다. 오답 475건의 구성은 `wrong` 315건과 `fallback_wrong` 160건이다. 고정 `reward_v2`로 다시 계산한 결과 chosen **707/707건이 1**, rejected **707/707건이 0**이었고, 보상 이유도 raw 기록과 일치했다.

| 분야 | 표적 쌍 | 유지 쌍 | 전체 |
|---|---:|---:|---:|
| Art & Design | 2 | 1 | 3 |
| Business | 124 | 49 | 173 |
| Health & Medicine | 91 | 80 | 171 |
| Humanities & Social Science | 19 | 58 | 77 |
| Science | 159 | 51 | 210 |
| Tech & Engineering | 32 | 41 | 73 |

## ms-swift 형식과 CPU 토큰 검증

`d1_pairs_swift.jsonl`은 `messages`, `images`, `rejected_response`를 사용한다. 프롬프트와 이미지 경로는 Stage 3의 `sample_m0.example_for`로 다시 만든 값과 전 쌍에서 일치한다. 로컬 이미지 548개의 크기와 SHA-256을 샘플링 풀의 기록과 대조했다. chosen·rejected 문자열은 raw와 문자 단위로 일치한다. 잘린 rejected에는 `rejected_add_eos=false`를 두고, 학습 config에서 이 열을 유지한다. 정상 종료 응답은 E1과 같은 기본 접미사를 사용한다.

| 토큰 기준 | chosen 평균 / 중앙값 / 최대 | rejected 평균 / 중앙값 / 최대 |
|---|---:|---:|
| vLLM raw 출력 토큰 | 1,997.70 / 1,369 / 8,146 | 3,501.21 / 1,883 / 8,192 |
| 학습 응답 토큰 독립 계산 | 1,998.60 / 1,370 / 8,147 | 3,501.58 / 1,884 / 8,192 |

전 쌍의 길이는 고정 풀에 기록된 이미지 포함 프롬프트 토큰 수에 별도로 인코딩한 응답 토큰 수를 더했다. 로컬 ms-swift 3.12.3의 JSONL 로더는 707쌍을 읽었다. 아래 7쌍에서는 `sample_m0.prompt_input`으로 프롬프트를 재구성한 뒤 DPO 전처리와 비교했다. chosen·rejected 모두 전체 토큰 ID와 label 마스크가 일치했고, 프롬프트·이미지 토큰은 모두 마스킹됐다. 표본의 잘린 rejected 3건은 8,192토큰을 그대로 사용했다.

| 쌍 인덱스(0부터) | 풀 행 | 프롬프트 | 그중 이미지 | chosen | rejected | 긴 쪽 전체 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `1295429` | 1,086 | 1,020 | 975 | 500 | 2,061 |
| 1 | `1295429` | 1,086 | 1,020 | 652 | 413 | 1,738 |
| 2 | `1541116` | 1,154 | 1,012 | 5,456 | 8,192 | 9,346 |
| 11 | `208154` | 1,207 | 1,088 | 472 | 500 | 1,707 |
| 12 | `140407` | 1,089 | 1,000 | 1,658 | 8,192 | 9,281 |
| 21 | `160659` | 1,129 | 1,040 | 321 | 430 | 1,559 |
| **114** | **`156933`** | **1,516** | **989** | **1,794** | **8,192** | **9,708** |

`max_length=10240`은 실측 최장값보다 532토큰 여유가 있고 상한 12,288 이하다. 초과 입력은 `truncation_strategy=raise`로 실패 처리한다.

## 학습 스크립트와 설정

- E1의 설정에서 모델 revision, bf16·SDPA, LoRA 대상 정규식, r16·α32·dropout 0.05, 이미지 크기, template, 마스킹, batch 1 × 16, seed 3407을 가져왔다. DPO의 기본 dropout 비활성화를 해제해 E1의 LoRA dropout을 유지한다.
- `rlhf_type=dpo`, sigmoid, **β 0.1 / rpo_alpha 1.0 / lr 5e-6 / 2 epoch**를 적용했다. 분할과 중간 평가는 껐다. 초기 adapter·resume·별도 기준 모델은 비어 있으며, DPO의 기준 경로는 LoRA를 끈 M0를 사용한다.
- 파일럿 순서를 CPU에서 검증했다. 최장 쌍은 optimizer step **20**에 배치되고, 파일럿은 step 20에 저장하도록 설정했다. 본학습은 epoch 저장과 마지막 체크포인트 한 개 보존 설정이다.
- 기존 CPU 환경에서 ms-swift가 생성하는 DPO 인자에 β·RPO·lr·epoch·길이·dropout·저장 설정이 전달되는지 확인했다. 모델 가중치·trainer를 로드하지 않았다. Windows에서 첫 JSONL 로더 호출이 기본 캐시 경로 쓰기 권한에 걸려 검증 캐시를 작업 폴더 아래로 지정한 뒤 통과했다. 패키지 설치·GPU·유료 API·GitHub/HF 업로드는 없었다.

## 결과 파일 SHA-256

| 파일 | SHA-256 |
|---|---|
| `data/d1_pairs_swift.jsonl` | `d6ae9fa2a4b391ed533964beadb1f1ed4d887359df7f7fd3aa88c5433d4abbca` |
| `data/d1_pairs_index.jsonl` | `2de82d0968838795254097d419082907e79e4a57044eaaff39f46d903cca63f5` |
| `scripts/build_dpo_pairs.py` | `34d7f7e11b215ebac909ea60b8d3737ac3f0d79bb84cf6e6da5bfacd0607230b` |
| `scripts/check_dpo_pairs.py` | `629b8f2042764ddeff6d173ad325cbfc702bb1d05281dcfab4d2f2fcf5707207` |
| `scripts/dpo_train.py` | `68fe6cae4e93d06ce8457a0694eae4e7ead9ce8e7534094fc034ea58ff9323fb` |
| `scripts/dpo_config.json` | `30f901b04abd4f49f5adbf59fe47c346f2190ebbd8c321cbe4ef85484e72a421` |
| `reports/d1_pair_validation.json` | `763f7cb86a6d9fe5f1e28b989981996e22f670d9d078737ea67275aff283e09b` |

1단계에서 멈췄다. GPU 파일럿은 사용자 승인 후 진행한다.

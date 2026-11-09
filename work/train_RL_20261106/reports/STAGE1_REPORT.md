# R1 1단계 CPU 작업 보고

2026-11-06 KST. HANDOFF의 1~6을 수행하고 7단계에서 멈췄다. **학습 3,000건 중 답 형식 후보는 2,333건, 현재 고정 추출기로 2단계 시험에 연결할 수 있는 객관식은 1,574건이다.** 숫자와 짧은 답 759건은 원래 v2 추출 동작을 진단했고 보상 정책 결정이 필요하다. 실행 범위는 로컬 CPU 및 공개 문서·소스 조사였다.

## 1. 답 형식 분류

| 형식 | 학습 | holdout | 학습 간이 분류 대비 |
|---|---:|---:|---:|
| mc_letter | 1,343 | 121 | +74 |
| mc_value | 231 | 31 | −5 |
| numeric | 587 | 81 | −67 |
| short_rule | 172 | 9 | −480 |
| excluded | 667 | 57 | +478 |
| 합계 | 3,000 | 299 | 0 |

출력은 원래 handoff 순서로 3,299행을 기록했다. `index`는 전체 순서, `swift_index`는 split 내부 0-based 순서다. `row/source/id/role`과 이미지 manifest의 경로·순서를 교차 확인했다. 답 형식 후보는 학습 2,333 / holdout 242, 그중 객관식은 학습 1,574 / holdout 152다. [answer_types.jsonl](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/data/answer_types.jsonl)

- 객관식은 E1 Swift 프롬프트 끝의 연속 보기 블록과 고정 안내문으로 식별했다. 유효한 단일 문자와 `Option B`, 소문자 등 명시적 라벨은 `mc_letter`다. 보기 텍스트는 기존 E1의 `content_key`로 비교해 정확히 한 보기에 대응할 때 `mc_value`로 두었다. 라벨이 함께 적힌 경우 값의 대응과 라벨이 일치해야 한다. E1 마지막 `Final Answer`의 마침표와 닫는 인용부호는 확인할 때 표기 구분자로 처리했다. [기존 E1 변환기](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_data_B_20261023/final_v5/stage47_prepare.py)
- 주관식 숫자는 부호·소수·지수 표기를 포함한 단일 숫자 표면형만 분류했다. 단위, 쉼표가 들어간 값, %, 분수, 식·구간·복수 값은 보류 집계인 `excluded`에 넣었다. `short_rule`은 yes/no 및 두 글자 이상 영문 단일 단어 후보로 한정했다. 간이 분류의 '기타 짧은 답'에는 분수·여러 답이 포함되어 있어 직접 대응하지 않는다.
- 문자 또는 값으로 대응된 객관식 1,726건은 E1 마지막 답 라벨과 **전부 일치**했다. 별도 목록 48건은 학습 복수 문자 29건, 대응 불가 또는 표기 차이 학습 16건·holdout 3건이다. 예를 들어 row 153949는 `-105,000` / 보기 `-$105,000`, row 113504는 `12` / 보기 `(12)`, row 745619는 `2` / 보기 `2 cm`다. E1의 선택 라벨은 남기되 통화·단위·괄호의 동치 기준을 도입하기 전에는 보상용 정답으로 채택하지 않았다. [객관식 교차 확인 목록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/e1_mc_crosscheck.json)

| split | 분야 | mc_letter | mc_value | numeric | short_rule | excluded | 합계 |
|---|---|---:|---:|---:|---:|---:|---:|
| train | Art & Design | 56 | 4 | 7 | 12 | 36 | 115 |
| train | Business | 444 | 43 | 149 | 21 | 139 | 796 |
| train | Health & Medicine | 281 | 47 | 109 | 57 | 53 | 547 |
| train | Humanities & Social Science | 198 | 0 | 7 | 5 | 19 | 229 |
| train | Science | 246 | 76 | 199 | 10 | 293 | 824 |
| train | Tech & Engineering | 118 | 61 | 116 | 67 | 127 | 489 |
| holdout | Art & Design | 0 | 0 | 0 | 0 | 0 | 0 |
| holdout | Business | 55 | 3 | 27 | 2 | 22 | 109 |
| holdout | Health & Medicine | 23 | 3 | 11 | 5 | 6 | 48 |
| holdout | Humanities & Social Science | 0 | 0 | 0 | 0 | 0 | 0 |
| holdout | Science | 16 | 6 | 22 | 1 | 22 | 67 |
| holdout | Tech & Engineering | 27 | 19 | 21 | 1 | 7 | 75 |

## 2. 보상 함수와 숫자 비교 근거

`reward(text, finish_reason, item)`은 정상 종료에서 추출 정답이면 `1/correct`, 다른 답이면 `0/wrong`, 답을 추출하지 못하면 `0/extract_fail`을 반환한다. `length`는 추출 결과에 관계없이 `0/truncated`다. `extract()`는 M0 검증용으로 잘림 게이트 전 추출도 노출한다. [reward.py](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/scripts/reward.py)

평가 정책은 `hybrid100_mc_qwen_ab_open_no_length_gate_v2`다. 원본은 `work/team-evaluator-no-length/assignment/experiments/raw_evaluation/evaluate.py`이며 길이 게이트를 제거한 변경 커밋은 `7c663738e7acafafeb5022da4aabea65d7c76688`, 읽은 체크아웃 HEAD는 `be93bce5b7c114f2fe7776822e4eaa2c5ccb798c`다. 사본의 `derive_item`을 호출해 Qwen 규칙과 객관식 Final Answer 보완·충돌을 그대로 적용한다. v2 manifest가 기록한 5개 코드 해시와 원본·사본을 모두 대조했다. 평가기 SHA-256은 `0ed761277d1c947cf16ab71aba8cdb74629f208a81f96e47d63c688fca8481e7`이다. [평가기 원본](C:/Users/brain/Documents/ChatGPT/MMDL/work/team-evaluator-no-length/assignment/experiments/raw_evaluation/evaluate.py) · [M0 v2 manifest](C:/Users/brain/Documents/ChatGPT/MMDL/work/team-evaluator-no-length/assignment/experiments/raw_evaluation/outputs/release-v2-latest/manifest.json)

숫자와 짧은 답은 v2의 A=참조답 / B=Other Answers 경로를 **진단용**으로 재현했다. 이 추출기는 응답 전체에서 부분 문자열도 찾는다. 합성 사례에서 참조 `31`에 `Final Answer: 131`을 1점, 참조 `yes`에 앞서 yes를 언급한 뒤 `Final Answer: no`라고 끝낸 응답을 1점으로 처리했다. 현 단계 샘플러의 실행 분기는 열린 답 형식을 명시적으로 보류한다.

| 사용자 결정 항목 | 제안과 근거 |
|---|---|
| 숫자 추출 위치 | 승인 후 명시된 최종 답에서 단일 스칼라를 검증하는 경로를 비교한다. 위 사례처럼 부분 문자열을 수치로 취급하면 0/1 보상 오류가 생긴다. 여러 최종값·충돌·범위는 추출 실패 후보로 분리한다. |
| 허용 오차 | Decimal로 `31`과 `31.0`, `1e3`과 `1000`의 정확 동치를 먼저 비교할 수 있다. 근사 허용이 필요한 문항에는 명시된 반올림 자릿수 또는 절대·상대 오차의 근거가 필요하다. 작은 값과 0 부근은 상대 오차만으로 판정하기 어렵다. 오차값은 미정이다. |
| 쉼표·단위·% | 쉼표는 엄격한 3자리 묶음일 때만 수천 구분자 후보로 처리한다. `1,000`과 목록 `1, 0`은 구별한다. 단위의 생략·변환은 문항이 요구하는 단위와 대조하고, `50%`와 `0.5`의 동치는 비율 의미가 확인된 경우만 후보로 둔다. |
| 짧은 답 | 승인 후 최종 답의 완전 일치와 정해진 대소문자 처리 범위를 비교한다. 동의어·부분 일치 도입 여부는 holdout에서 확인할 수 있다. |

## 3. 보상 검증

M0 raw SHA-256 `ef23f0c49d9b1ae6c62b9625fbd52cce474a0c035c1a644cfa737e0967daa313`와 v2 manifest, 문항 ID 900개, 개별 응답 해시, route를 대조했다. MMMU raw는 이 대조에만 사용했다.

| v2 경로·항목 | 결과 |
|---|---:|
| 자동 555건의 추출 답 일치 | **555/555 (100%)** |
| 자동 555건의 길이 게이트 전 정오 일치 | **555/555 (100%)** |
| 자동 중 정상 종료 보상 정오 일치 | **537/537 (100%)** |
| 자동 전체의 R1 보상 정오 일치 | **543/555 (97.84%)** |
| 자동 중 length | 18건. v2 정답 12건이 R1 잘림 0점으로 바뀌고 기존 오답 6건은 0점 |
| Judge 경로 345건의 R1 결과 | 추출 실패 235 / 잘림 110 / 정답 0 / 추출된 오답 0 |
| Judge 경로의 잘림 게이트 전 추출 | 실패 345. 규칙 추출 실패 경로 341, 추출 규칙 충돌 4 |

불일치 12건은 모두 잘림 게이트에서 발생했다. 전체 ID와 원인, 게이트 전후 값을 보존했다. 이 차이는 지정한 R1 보상과 기존 v2 채점 조건의 차이다. 합성 테스트는 실제 학습 행의 답 형식을 사용해 6/6/6/6/5건, 총 **29/29**가 통과했다. 숫자·짧은 답의 부분 문자열 문제는 별도 진단 2건으로 재현했다. [문항별 불일치와 합성 테스트 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/reward_check.json)

## 4. pass_rate 출처와 분포

논문 §3.3에 따르면 난도 측정기는 **Qwen3-VL-4B-Thinking**이며 문항당 독립 응답 **4개**의 정답률을 계산한다. 데이터 카드의 `qwen3vl_4b_pass_rate` 설명과 로컬 parquet 수집 코드의 `pass_rate` 별칭이 연결된다. 이 값은 Thinking 모델의 난도 지표이고 R1용 Instruct M0의 성공률·잘림·방향 전환 측정을 대신할 수 없다. [로컬 열 매핑](C:/Users/brain/Documents/ChatGPT/MMDL/work/mmfr18m_check/stage1.py:70) · [MMFR 데이터 카드](https://huggingface.co/datasets/OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking#data-schema) · [MMFR 논문 §3.3](https://arxiv.org/html/2601.21821v1#S3.SS3)

0, 0.25, 0.5, 0.75, 1의 학습 건수는 초안과 일치한다. 원본에는 여기에 `0.6666666666666667`이 2건 추가된다. 두 건은 Science의 row 1064545와 586909다. 실제 시도 분모는 개별 기록으로 확인되지 않아 2/3회라고 확정할 수 없다. [분포 JSON](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/pass_rate_distribution.json)

| split | 형식 또는 분야 | 0 | 0.25 | 0.5 | 0.666… | 0.75 | 1 | 합계 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | 전체 | 570 | 529 | 529 | 2 | 911 | 459 | 3000 |
| train | excluded | 139 | 116 | 130 | 1 | 206 | 75 | 667 |
| train | mc_letter | 247 | 244 | 220 | 1 | 367 | 264 | 1343 |
| train | mc_value | 35 | 42 | 52 | 0 | 88 | 14 | 231 |
| train | numeric | 127 | 103 | 104 | 0 | 191 | 62 | 587 |
| train | short_rule | 22 | 24 | 23 | 0 | 59 | 44 | 172 |
| train | Art & Design | 11 | 4 | 5 | 0 | 7 | 88 | 115 |
| train | Business | 212 | 200 | 153 | 0 | 221 | 10 | 796 |
| train | Health & Medicine | 78 | 79 | 67 | 0 | 123 | 200 | 547 |
| train | Humanities & Social Science | 34 | 14 | 19 | 0 | 24 | 138 | 229 |
| train | Science | 123 | 153 | 182 | 2 | 364 | 0 | 824 |
| train | Tech & Engineering | 112 | 79 | 103 | 0 | 172 | 23 | 489 |
| holdout | 전체 | 29 | 28 | 39 | 0 | 43 | 160 | 299 |
| holdout | excluded | 4 | 6 | 6 | 0 | 12 | 29 | 57 |
| holdout | mc_letter | 9 | 12 | 15 | 0 | 12 | 73 | 121 |
| holdout | mc_value | 4 | 1 | 8 | 0 | 12 | 6 | 31 |
| holdout | numeric | 11 | 8 | 10 | 0 | 7 | 45 | 81 |
| holdout | short_rule | 1 | 1 | 0 | 0 | 0 | 7 | 9 |
| holdout | Business | 3 | 0 | 5 | 0 | 2 | 99 | 109 |
| holdout | Health & Medicine | 0 | 0 | 0 | 0 | 0 | 48 | 48 |
| holdout | Science | 10 | 12 | 15 | 0 | 19 | 11 | 67 |
| holdout | Tech & Engineering | 16 | 16 | 19 | 0 | 22 | 2 | 75 |

holdout의 Art & Design 및 Humanities & Social Science는 0건이다.

## 5. M0 샘플링 준비와 CPU dry-run

`sample_m0.py`는 train 후보를 E1 Swift 순서로 읽고 첫 `--limit 100`을 시험할 수 있다. 프롬프트는 E1의 user 메시지와 이미지 앞배치를 재사용한다. 평가 `code/config.yaml` 사본 SHA-256은 `3946c9a93cd5ab1cc5a8010fc00473e8a71f1c589e0a5f664edb70a3097234dc`, 원본 체크아웃은 `83564713473b535e12a78ff881e8fa8e0733e713`이다. vLLM 0.11.0, bf16 M0 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, context 16384, 출력 한도 8192, temperature 0.7 / top_p 0.8 / top_k 20 / repetition 1.0 / presence 1.5, pixels 1,003,520–4,014,080을 읽는다. [샘플러](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/scripts/sample_m0.py) · [평가 설정 사본](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/scripts/frozen_generation/config.yaml)

엔진 seed 3407, 요청 seed는 `3407 + swift_index × 65536 + repeat`이며 repeat는 0부터 시작한다. K·재개 위치가 달라져도 같은 문항·회차의 seed가 같다. 3,000×8의 24,000개 seed가 고유함을 CPU에서 확인했다. 실제 엔진 스케줄과 입력 묶음에 따라 출력 재현성은 달라질 수 있다. raw에는 문항·회차, 응답 원문, 종료 사유, 토큰 수, 보상·사유, 마커 수를 기록하도록 했다. manifest의 입력·설정·코드 해시와 회차를 확인해 완료 요청을 건너뛴다. 중간에 끊긴 마지막 행은 해시 표식을 append하여 봉인하고 기존 바이트를 보존하는 경로를 합성 파일로 확인했다. 강제 종료 후 잠금이 남았으면 이전 프로세스가 종료됐음을 확인한 뒤 잠금을 해제해야 한다.

CPU dry-run은 **100/100** 프롬프트가 E1 원본과 메시지·이미지 순서·pixels까지 일치했고, tokenizer/processor 계산도 동일 입력으로 측정한 기존 ms-swift 참조와 일치했다. 형식별 51/16/25/8건, 프롬프트 최소 1,020 / 중앙값 1,106.5 / 최대 1,317토큰이다. 기존 참조에 있는 학습 후보 전체 2,333건은 1,009–3,607토큰으로 평가 프롬프트 한도 8,064 이내다. 모델 없는 환경에서 Torch 2.8.0, Transformers 4.57.1, tokenizers 0.22.2, qwen-vl-utils 0.0.14를 재사용했다. [dry-run 100행 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/sampling_dry_run.json) · [ms-swift 토큰 참조](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_E3_20261101/reports/token_lengths_E3.csv)

`build_groups.py`는 K회가 모두 저장된 문항만 집계한다. 혼합 보상 중 1회 이상 정답이며 length 또는 마커 ≥21이 한 번이라도 있으면 표적, 나머지 혼합 보상은 유지다. 초안의 두 조건이 겹칠 수 있는 전부 정답+방향 전환 과다의 경우, 그룹 상대 학습 신호가 없다는 제외 기준을 먼저 적용하고 겹침 플래그를 남긴다. 이 우선순위는 보고용 구현 해석이다. 7:3은 명시 인자로 시험할 수 있고 seed 3407로 중복 없이 두 풀을 섞는다. 비율을 정확히 맞추기 위해 작은 풀로 가능한 비율 단위까지만 뽑는다. 합성 6개 분기와 재현 가능한 7:3을 확인했다. 표적 300건 미만이면 플래그만 기록한다. 실제 표적·유지 수와 비율은 미확정이다. [그룹 빌더](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/scripts/build_groups.py)

| 2단계 범위 | K=4 예시 | K=8 예시 |
|---|---:|---:|
| 100문항 시험 | 400 | 800 |
| 현재 객관식 1,574문항 | 6,296 | 12,592 |
| 열린 답까지 승인 후 최대 후보 2,333문항 | 9,332 | 18,664 |

100문항 시험의 GPU 후보는 이전 동일 계열 bf16 생성 환경인 **RTX 4090 24GB 한 장**이다. E1/E3 생성 기록은 각각 약 694/593 출력 토큰/초였다. 평균 2,000토큰으로 가정하면 400응답은 19–23분, 800응답은 38–45분이다. 초기화·입력 처리 여유를 더하면 대략 **25–35분 / 45–60분**이며, 가중치 다운로드 시간은 별도다. 출력이 모두 8192토큰에 가까우면 400응답만 79–92분 수준으로 늘 수 있다. 이는 다른 출력 분포에서 얻은 외삽이며 실제 M0 throughput은 시험에서 측정해야 한다. [E1 시간 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_B_20261101/runs/E1/generation/run_metadata.json) · [E3 시간·VRAM 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/eval_E3_20261101/reports/GENERATION_REPORT.md)

## 6. ms-swift 3.12.3 GRPO 조사

로컬 읽기 전용 소스의 태그는 v3.12.3, 커밋은 `d2644e01001e91c5666f969377d7680a2509c14d`다. 이 버전은 Qwen3-VL-4B-Instruct 등록, GRPO의 colocate engine, LoRA 학습·가중치 동기화와 사용자 정의 ORM 경로를 갖는다. 태그 예제는 Qwen2.5-VL의 LoRA+colocate를 보여 주며, 정확히 R1의 4B·단일 GPU·8192 조합은 GPU 파일럿으로 검증해야 한다. [릴리스](https://github.com/modelscope/ms-swift/releases/tag/v3.12.3) · [Qwen3-VL 등록](https://github.com/modelscope/ms-swift/blob/d2644e01001e91c5666f969377d7680a2509c14d/swift/llm/model/model/qwen.py#L1105) · [멀티모달 LoRA 예제](https://github.com/modelscope/ms-swift/blob/d2644e01001e91c5666f969377d7680a2509c14d/examples/train/grpo/internal/vllm_lora_qwenvl72b.sh)

| 항목 | 코드에서 확인한 경로와 R1 확인 사항 |
|---|---|
| 기본 인자 | `--rlhf_type grpo --train_type lora --use_vllm true --vllm_mode colocate`. `--num_generations`는 학습 rollout의 G이며 2단계 사전 샘플링 K와 별개다. `--max_completion_length`, `--vllm_max_model_len`, `--temperature`, `--top_p`, `--top_k`, `--repetition_penalty`, `--learning_rate`, `--beta`를 구분한다. |
| 배치·KL | rollout 배치는 G로 나누어져야 한다. 16문항×8이면 128 completion의 생성 배치이며 `--generation_batch_size` 또는 `--steps_per_generation`과 per-device batch·프로세스 수를 조합한다. optimizer step 및 gradient accumulation과 구분해 확인한다. PEFT 모델의 reference 경로는 adapter를 끌 수 있다. 시작 모델 및 reference 구성을 확정한 뒤 KL을 확인한다. |
| 사용자 보상 | 외부 plugin에서 `swift.plugin.ORM` 구현을 `orms["r1_binary"]`에 등록하고 `--external_plugins … --reward_funcs r1_binary`로 연결할 수 있다. 문항의 형식·정답·보기는 입력의 별도 열로 유지하고, trainer가 넘기는 `completions`와 배치 `finish_reason`을 R1 reward에 전달한다. rollout 소스가 `finish_reason`·`is_truncated`를 기록한다. 잘림을 0점 비교 대상으로 유지하려면 `overlong_filter`의 학습 마스크를 확인해야 한다. |
| 멀티모달·VRAM | 언어 LoRA만 적용하고 ViT·aligner 고정 여부를 확인한다. `--vllm_enable_lora`는 adapter 전송 최적화이며, 꺼진 경로도 LoRA 학습 가중치를 동기화한다. 3.12.3은 ViT LoRA를 vLLM에 적용할 때 경고한다. colocate는 학습·rollout이 VRAM을 공유하므로 utilization, sleep/offload, 모델 길이와 배치의 실측이 필요하다. |
| 버전 | 태그의 설치 참고 스크립트에 vLLM `>=0.5.1,<0.11.1`, Transformers `<4.58`, TRL `<0.25`가 적혀 있어 기존 vLLM 0.11.0/Transformers 4.57.1/TRL 0.22.2가 범위에 든다. 이 범위는 설치 참고이고 전체 조합의 성공 보장은 아니다. rollout의 processed logprobs 분기는 vLLM ≥0.10.2를 사용한다. |
| 평가 생성과의 차이 | **stock 3.12.3 GRPO는 `presence_penalty=1.5`를 rollout 요청으로 전달하지 않는다.** `RequestConfig`에 필드가 있고 vLLM adapter도 읽지만 trainer 초기화가 넘기지 않아 기본값 0.0이 된다. 엔진 seed 역시 rank 기반 경로다. 정확한 rollout 조건을 요구하면 승인된 별도 trainer 연결에서 전달 여부를 확인하거나, 해당 인자를 실제로 전달하는 릴리스를 확인해야 한다. 이번 조사에서는 그 조건을 만족하는 업그레이드 버전을 확정하지 않았다. 2단계 독립 샘플러는 고정 vLLM 경로로 1.5를 직접 전달한다. |

소스 근거는 [GRPO trainer의 보상 배치](https://github.com/modelscope/ms-swift/blob/d2644e01001e91c5666f969377d7680a2509c14d/swift/trainers/rlhf_trainer/grpo_trainer.py#L328), [rollout 요청·colocate](https://github.com/modelscope/ms-swift/blob/d2644e01001e91c5666f969377d7680a2509c14d/swift/trainers/rlhf_trainer/rollout_mixin.py#L101), [finish_reason 전달](https://github.com/modelscope/ms-swift/blob/d2644e01001e91c5666f969377d7680a2509c14d/swift/trainers/rlhf_trainer/rollout_mixin.py#L1133), [RequestConfig 기본값](https://github.com/modelscope/ms-swift/blob/d2644e01001e91c5666f969377d7680a2509c14d/swift/llm/infer/protocol.py#L63), [vLLM 전달](https://github.com/modelscope/ms-swift/blob/d2644e01001e91c5666f969377d7680a2509c14d/swift/llm/infer/infer_engine/vllm_engine.py#L439), [설치 참고 범위](https://github.com/modelscope/ms-swift/blob/d2644e01001e91c5666f969377d7680a2509c14d/requirements/install_all.sh)다. 보상 plugin 구현·설치·rollout 실행은 이번 조사 범위 밖이다.

## 7. 멈춤과 결정 필요 항목

숫자 오차·정규화와 짧은 답의 최종 답 검증, K=4/8, 7:3 조합과 표적 부족 시 추가 풀, 시작 모델, lr, β, 팀 공식 여부는 사용자 결정 사항이다. 학습 rollout의 `presence_penalty` 전달 경로도 파일럿 전에 확인해야 한다. 100문항 GPU 시험은 별도 단계다. 이 단계의 결과는 로컬에만 저장했다.

CPU의 합성 재개 검사 중 Windows가 Python 임시 디렉터리 접근을 거부했다. 접근 가능한 별도 합성 파일로 재개 검사를 다시 실행해 통과했고 해당 파일은 지웠다. 접근이 거부된 임시 디렉터리 `tmpw2zfz6zq`는 권한 문제로 정리하지 못했다.

### 생성 파일과 SHA-256

경로는 R1 폴더 기준이다. 보고서 자체의 해시는 별도 `STAGE1_REPORT.sha256`에 기록한다.

| 파일 | SHA-256 |
|---|---|
| `data/answer_types.jsonl` | `c4f03528d3f788a5508ddc99e60c92f741002c6f3cb475d9db1cd823c6e6c166` |
| `reports/e1_mc_crosscheck.json` | `60f8cd3e9c721137cc3b94874adfae1036a4cc5c9f5a92c3a8792b012269e9d5` |
| `reports/pass_rate_distribution.json` | `e8372f6496fc7553ec08f69c579dd8e0f7a59c99b86139da2e5ba426b1cb2bbb` |
| `reports/reward_check.json` | `2bfffbced291315675a9068f4a30ebc3f89fbb466407fd55a43cf0924d31f63d` |
| `reports/sampling_dry_run.json` | `1dbb6c76bbb1c6bf0f7baa0462c5f08b7d058da27b075e13cc12515e67056de7` |
| `scripts/build_groups.py` | `1af71c428677c83445448fb94414616a0e85ed916aad1ac8f8687175b638ec8c` |
| `scripts/classify_answers.py` | `6f4e24f8f4a739e8037732118ae53816ee0974270bf69607446233a238d2d262` |
| `scripts/frozen_generation/common.py` | `1a6f6c05c8a16f7ef39d1800752f3185d4e856c21e1cb7cecece3b37cff2dc14` |
| `scripts/frozen_generation/config.yaml` | `3946c9a93cd5ab1cc5a8010fc00473e8a71f1c589e0a5f664edb70a3097234dc` |
| `scripts/frozen_generation/generate.py` | `efa9fcd8466b3e268dcc5634c493fa37acc1bb0c6459ef23004625d5ca627497` |
| `scripts/frozen_generation/inputs.py` | `d2bc62cd80a840215484da90e8062ebc92c5175e9a7d5b48956c9806c4055c3b` |
| `scripts/frozen_generation/vllm_engine.py` | `979e99079c7a8d31dc969fef16530e3cc8bc47588e1c38447edee92c2b759301` |
| `scripts/frozen_v2/evaluate.py` | `0ed761277d1c947cf16ab71aba8cdb74629f208a81f96e47d63c688fca8481e7` |
| `scripts/frozen_v2/frozen/parsers.py` | `df41e5e5c99eabeda4d0a1cb1ce557e2f3f9c13416fd9481aaba0388b0a303bc` |
| `scripts/frozen_v2/frozen/THIRD_PARTY.md` | `e3a3f3a1efdedae21a20fba05bda15b12677c7781cc13cd740f16af57fa66f45` |
| `scripts/frozen_v2/frozen/vendor/MMMU-LICENSE` | `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6` |
| `scripts/frozen_v2/frozen/vendor/mmmu_eval.py` | `4ab1574dec79f0ba4835c03a255d8e2b860377d1ea8f795b53a8d1869f69e0f4` |
| `scripts/frozen_v2/frozen/vendor/Qwen-LICENSE` | `47b2b7375318c574ef44d0def1e28d296e8bac5940d24d2694f5d99e7053b780` |
| `scripts/frozen_v2/frozen/vendor/qwen_eval_utils.py` | `9967335315d0f5905b4f1da7052e0cabdb6f2abd94c166d3a5b2533ac46cad2d` |
| `scripts/frozen_v2/frozen/vendor/VLMEvalKit-LICENSE` | `accb7415e6082edd0ee2918738fd53579782e8fbeb983028f2bdf6795442290a` |
| `scripts/frozen_v2/frozen/vendor/vlmevalkit_matching.py` | `025cce503c644b51793c3118551d32d1ea67673365cf04164afbef3ceff012a7` |
| `scripts/reward.py` | `b697783b47b2d6bc520c11e119cf0c1ff6f1f02094f6670784b0c2a56756345e` |
| `scripts/sample_m0.py` | `9104cc786411e75e4a1e7843195b91b950ea5c4bb3ed9ff24d3cccc1d214d178` |
| `scripts/verify_reward.py` | `06b05cf15b5a6e65085c078ea0cbca59304d72cf8f783f04ee189e88022ad29b` |

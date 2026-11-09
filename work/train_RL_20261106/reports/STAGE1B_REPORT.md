# R1 1-2단계 학습 풀 구성 보고

2026-11-06 KST. **A 1,360 + B 139 + C 1 = 1,500건**을 확정했다. 근접쌍 4개가 나온 B 후보 3건을 제외했다. 시험용 100건을 고정하고 CPU dry-run 100/100을 통과한 뒤 7번에서 멈췄다.

## 1. A 상한 적용

학습 객관식 1,574건 중 Economics 463건을 MMFR `row` 정수 오름차순으로 정렬하고 `random.Random(3407).sample(..., 250)`로 남겼다. 선택된 Economics와 나머지 과목을 기존 handoff 순서로 기록했다. **A 상한 후 1,361건**, B 자리는 **139건**이다. Business는 274건 중 Economics 250건(91.24%)이다. C 상한 계산에 쓰는 A의 Art & Design은 60건이다.

| 분야 | 건수 | 과목별 건수 |
|---|---:|---|
| Art & Design | 60 | Art 7, Art_Theory 5, Design 24, Music 24 |
| Business | 274 | Accounting 8, Economics 250, Finance 4, Manage 5, Marketing 7 |
| Health & Medicine | 328 | Basic_Medical_Science 134, Clinical_Medicine 69, Diagnostics_and_Laboratory_Medicine 84, Pharmacy 24, Public_Health 17 |
| Humanities & Social Science | 198 | History 191, Literature 4, Psychology 1, Sociology 2 |
| Science | 322 | Biology 44, Chemistry 16, Geography 25, Math 192, Physics 45 |
| Tech & Engineering | 179 | Agriculture 0, Architecture_and_Engineering 1, Computer_Science 161, Electronics 9, Energy_and_Power 2, Materials 1, Mechanical_Engineering 5 |

제외한 Economics 213건의 row 목록과 전 과목 집계를 보관했다. [상한 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_a_cap.json)

## 2. B·C 객관식 후보 추출

지정 CSV를 split·분야·과목으로 걸렀고, 로컬 SQLite는 `mode=ro`와 `query_only=ON`으로 열어 `source + id + split`을 대조했다. 제외 사유는 먼저 해당한 항목 하나로 집계했다. C에서는 기존 train·holdout 3,299건과 동일한 source+id를 먼저 제외했다. 기존 E1 변환기의 `split_options`와 `content_key`, R1 1단계의 `classify`를 사용했다.

| 풀 | 과목 | 대상 | 기존 ID | pass_rate 0 | 불일치 라벨 | 비객관식 | 보기 모호 | 정답 대응 모호 | 객관식 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | Agriculture | 15 | 0 | 5 | 0 | 8 | 0 | 0 | 2 |
| B | Architecture_and_Engineering | 19 | 0 | 10 | 0 | 8 | 0 | 0 | 1 |
| B | Computer_Science | 1378 | 0 | 591 | 0 | 370 | 10 | 25 | 382 |
| B | Electronics | 26 | 0 | 23 | 0 | 2 | 0 | 1 | 0 |
| B | Energy_and_Power | 12 | 0 | 7 | 0 | 2 | 0 | 0 | 3 |
| B | Materials | 18 | 0 | 16 | 0 | 2 | 0 | 0 | 0 |
| B | Mechanical_Engineering | 29 | 0 | 23 | 0 | 6 | 0 | 0 | 0 |
| B | 합계 | 1497 | 0 | 675 | 0 | 398 | 10 | 26 | 388 |
| C | Art | 129 | 21 | 11 | 0 | 83 | 0 | 0 | 14 |
| C | Art_Theory | 117 | 38 | 5 | 0 | 70 | 0 | 1 | 3 |
| C | Design | 104 | 10 | 8 | 0 | 79 | 1 | 0 | 6 |
| C | Music | 250 | 80 | 22 | 0 | 115 | 0 | 0 | 33 |
| C | 합계 | 600 | 149 | 46 | 0 | 347 | 1 | 1 | 56 |

B의 비CS 공학 객관식 후보는 총 6건으로 Agriculture 2, Architecture and Engineering 1, Energy and Power 3이다. Computer Science는 객관식 382건 중 동일 보기 중복 1건(row 8357)이 구조 검사에서 탈락해 이미지 검사 대상 381건이다. 비CS를 모두 통과한다고 가정해도 B 139자리 중 최소 133자리는 CS가 필요하다. 최종 선정과 CS 순서도 실행 집계는 4절에 기록했다. [행별 추출 결과](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_prescreen.jsonl)

## 2-2. C 문항 내용 직접 판정

객관식 56건의 문항·선택지를 전부 읽어 rubric_v2의 과목 기준을 적용했다. **Design 1건**이 통과했다. row 435237은 시계에 적용된 캐릭터 그래픽 로고 자체의 식별이다. 텍스트 기반 판정이며 이미지 검사 대상이다.

55건은 일반 장면·문자 식별 20, 일반 수학·계수 23, 경제 3, 그래프 이론 3, 통계·순서도 2, 퍼즐 2, 해부학 1, 텍스트만으로 예술 여부 불명 1이다. 음악 수업에 늦게 출발한 사람을 찾는 시계 문항과 물감 통·붓 개수를 비교하는 문항은 일반 수학에 해당했다. 모든 row의 판단 근거를 기록했다. [직접 검토 목록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_c_content_review.json) · [rubric_v2](C:/Users/brain/Documents/ChatGPT/MMDL/work/mmfr18m_relabel/rubric_v2.md)

## 3. 이미지 수신과 검사

사전 산정 1,102,724,846바이트가 1 GB를 넘어 일시 정지했다. 사용자 승인 후 고정 revision `7c93f724791d01246f5425e1182b14ec8d013847`의 승인된 259개 row group에서 이미지 열 범위를 받았다. 응답이 HTTP 206이며 요청한 Content-Range와 일치할 때만 본문을 읽었다. 성공한 범위의 수신 본문은 예상과 정확히 일치했다. 헤더는 집계에서 제외했다. [전송 계획](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_image_transfer_plan.json) · [전송 영수증](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_image_receipt.json)

| 풀 | 이미지 대상 | row group | 예상 bytes | 실수신 bytes |
|---|---:|---:|---:|---:|
| B | 387 | 258 | 1,099,704,755 | 1,099,704,755 |
| C | 1 | 1 | 3,020,091 | 3,020,091 |
| 합계 | 388 | 259 | 1,102,724,846 | 1,102,724,846 |

388개 대상 이미지를 로컬에 6,984,062바이트로 저장했다. 압축된 parquet 범위에는 같은 row group의 다른 행도 들어 있어 전송량과 저장량이 다르다. 선정 B·C 이미지 140개의 합계는 2,789,601바이트이며 파드 전송용 manifest에 이 140개를 기록했다. [선정 이미지 manifest](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/data/images_B_manifest.csv)

| 검사 결과 | B | C |
|---|---:|---:|
| 정답·선택지 구조 탈락 (수신 전) | 1 | 0 |
| 이미지 없음·디코딩 실패·단색 | 0 | 0 |
| 프롬프트·이미지 태그 불일치 또는 8,064토큰 초과 | 0 | 0 |
| 기존 3,299건과 중복 | 12 | 0 |
| 오염 근접 후보 제외 | 3 | 0 |
| B·C 후보 사이 정확 중복 | 12 | 0 |
| 검사 통과 | **360** | **1** |

기존 3,299건은 source+id, 정규화 텍스트(문항 전체 및 stem·보기), 이미지 SHA-256으로 대조했다. 중복 사유는 한 행에 겹칠 수 있다. 기존 풀과 겹친 12건 중 이미지 해시 일치 9, 텍스트 일치 6이었다. 새 후보 사이에서도 정확 중복을 제거했다. 깨끗한 대표를 고를 때 B의 비CS를 먼저, 같은 우선순위에서는 정수 row 순으로 대조했다. [행별 검사 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_checks.jsonl)

오염 검사는 기존 answer-free 캐시 SHA-256을 대조한 후 읽기 전용으로 실행했다. 기존 풀과 정확 중복을 제외한 후보 376개를 MMMU dev/validation/test 11,550건과 MMMU-Pro standard10 1,730건에 대조했다. 13-gram 일치 0, exact 5-gram Jaccard ≥0.50은 0(최대 0.1053)이었고, 64-bit 이미지 pHash 거리 ≤10 근접쌍은 4개였다. 5-gram은 모든 양의 겹침을 exact 계산해 기존 LSH 뒤의 exact guard와 같은 컷오프를 적용했다. 아래 후보 3건을 모두 풀에서 제외했다. [근접쌍 목록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_contamination_pairs.json)

| MMFR row | 참조 | 참조 ID | 참조 이미지 | pHash 거리 |
|---|---|---|---|---:|
| 7918 | MMMU/MMMU_Pro | `validation_Design_1` | image_1 | 10 |
| 17956 | MMMU/MMMU | `test_Architecture_and_Engineering_466` | image_1 | 10 |
| 17468 | MMMU/MMMU | `test_Chemistry_19` | image_3 | 10 |
| 7918 | MMMU/MMMU | `validation_Design_1` | image_1 | 10 |

## 4. B·C 선정과 A 축소

B 통과 360건 중 **비CS 6건을 전부 먼저** 넣었다. Agriculture 2, Architecture and Engineering 1, Energy and Power 3건이다. 남은 133자리는 CS 통과 354건을 정수 row 순으로 정렬해 `random.Random(3407)`로 뽑았다. CS 선택분의 **132/133 (99.25%)**를 이전 텍스트 실행 검출기가 순서도 실행 문항으로 표시했다. 검출 범위에는 정방향 출력 계산, 역추적, 조건 상자 채우기와 같은 절차 문항이 포함된다. 선정은 기존 규칙대로 완료했고 이 비율은 보고용이다. [CS 행별 실행 검출 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_cs_flowchart.json)

C의 Design 1건(row 435237)이 모든 검사를 통과했다. A의 Art 60건과 합해 Art & Design은 61건이다. 당시 Health 328, Science 322, Business 274에서 가장 큰 Health를 골라 pass_rate 1.0 우선, 정수 row 순을 seed 3407로 섞은 순서에서 Basic Medical Science row 1545924 한 건을 뺐다. 제거는 Health 1, Science 0, Business 0이며 **pass_rate 1.0 제거 비율은 1/1 (100%)**다. A의 Tech 179건은 유지됐다. Business는 Economics 250/274 (91.24%)다. [선정 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_selection.json)

## 5. 최종 풀

A는 기존 E1 Swift user prompt·이미지 순서를 사용했다. B·C는 E1 변환기의 `split_options`, `_prompt_text`, `build_messages`를 사용해 이미지 번호와 평가 프롬프트를 맞췄다. 전체 1,500건의 row 및 source+id가 고유하고, 이미지 경로·SHA-256과 프롬프트 토큰 한도를 검증했다. A 1,360건의 원본 split은 MMFR SQLite에서 모두 SFT로 확인했다. 형식은 mc_letter 1,220 / mc_value 280이다. [최종 학습 풀](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/data/r1_pool.jsonl)

| 분야 | A | B | C | 풀 1,500 | MMMU validation 900 | 시험 100 |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 60 | 0 | 1 | 61 (4.07%) | 120 (13.33%) | 4 |
| Business | 274 | 0 | 0 | 274 (18.27%) | 150 (16.67%) | 18 |
| Health & Medicine | 327 | 0 | 0 | 327 (21.80%) | 150 (16.67%) | 22 |
| Humanities & Social Science | 198 | 0 | 0 | 198 (13.20%) | 120 (13.33%) | 13 |
| Science | 322 | 0 | 0 | 322 (21.47%) | 150 (16.67%) | 22 |
| Tech & Engineering | 179 | 139 | 0 | 318 (21.20%) | 210 (23.33%) | 21 |
| 합계 | 1,360 | 139 | 1 | 1,500 (100.00%) | 900 (100.00%) | 100 |

| 분야 | 과목 | A | B | C | 합계 |
|---|---|---:|---:|---:|---:|
| Art & Design | Art | 7 | 0 | 0 | 7 |
| Art & Design | Art_Theory | 5 | 0 | 0 | 5 |
| Art & Design | Design | 24 | 0 | 1 | 25 |
| Art & Design | Music | 24 | 0 | 0 | 24 |
| Business | Accounting | 8 | 0 | 0 | 8 |
| Business | Economics | 250 | 0 | 0 | 250 |
| Business | Finance | 4 | 0 | 0 | 4 |
| Business | Manage | 5 | 0 | 0 | 5 |
| Business | Marketing | 7 | 0 | 0 | 7 |
| Health & Medicine | Basic_Medical_Science | 133 | 0 | 0 | 133 |
| Health & Medicine | Clinical_Medicine | 69 | 0 | 0 | 69 |
| Health & Medicine | Diagnostics_and_Laboratory_Medicine | 84 | 0 | 0 | 84 |
| Health & Medicine | Pharmacy | 24 | 0 | 0 | 24 |
| Health & Medicine | Public_Health | 17 | 0 | 0 | 17 |
| Humanities & Social Science | History | 191 | 0 | 0 | 191 |
| Humanities & Social Science | Literature | 4 | 0 | 0 | 4 |
| Humanities & Social Science | Psychology | 1 | 0 | 0 | 1 |
| Humanities & Social Science | Sociology | 2 | 0 | 0 | 2 |
| Science | Biology | 44 | 0 | 0 | 44 |
| Science | Chemistry | 16 | 0 | 0 | 16 |
| Science | Geography | 25 | 0 | 0 | 25 |
| Science | Math | 192 | 0 | 0 | 192 |
| Science | Physics | 45 | 0 | 0 | 45 |
| Tech & Engineering | Agriculture | 0 | 2 | 0 | 2 |
| Tech & Engineering | Architecture_and_Engineering | 1 | 1 | 0 | 2 |
| Tech & Engineering | Computer_Science | 161 | 133 | 0 | 294 |
| Tech & Engineering | Electronics | 9 | 0 | 0 | 9 |
| Tech & Engineering | Energy_and_Power | 2 | 3 | 0 | 5 |
| Tech & Engineering | Materials | 1 | 0 | 0 | 1 |
| Tech & Engineering | Mechanical_Engineering | 5 | 0 | 0 | 5 |

| 출처 | 0 | 0.25 | 0.5 | 0.666… | 0.75 | 1.0 | 합계 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 230 | 221 | 234 | 1 | 397 | 277 | 1360 |
| B | 0 | 6 | 8 | 0 | 14 | 111 | 139 |
| C | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| 합계 | 230 | 227 | 242 | 1 | 411 | 389 | 1500 |

B에서 5문항, C에서 1문항을 사용한 합성 테스트는 정답 6/6, 오답 6/6, 잘림 6/6으로 **18/18 통과**했다. 기존 보상 함수를 재사용했다. [보상 검사](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_reward_check.json)

## 6. 시험 100건 고정과 CPU dry-run

풀의 분야 비율을 최대잔여법으로 100자리에 배정했다. 잔여율 동률은 분야명 순이다. 분야명·정수 row 순으로 후보를 준비한 뒤 하나의 `Random(3407)`으로 분야 내 추출과 최종 셔플을 했다. 분야별 배정은 5절 표의 시험 열과 같으며 출처는 A 88 / B 12 / C 0이다. 단일 C 행은 층화 무작위 표본에 뽑히지 않았다. [고정 시험 100건](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/data/trial_100.jsonl)

엔진 seed는 3407이다. 요청 seed는 1단계의 `3407 + 문항 인덱스 × 65536 + repeat` 공식을 유지했다. A의 인덱스는 기존 E1 Swift 인덱스이고, B·C는 A 영역과 겹치지 않도록 `3000 + MMFR 정수 row`를 쓴다. trial 100건 × K=4의 요청 seed 400개가 고유하다. 실행 시 고정된 trial 및 pool 해시를 기록하고, 그룹 빌더도 이 새 manifest를 검증한다. B 합성 raw로 그룹 빌더 CLI를 확인했다.

CPU dry-run **100/100**에서 경로·이미지 해시·프롬프트 토큰이 일치했다. A 88건은 E1 Swift의 user·이미지 순서와 기존 토큰 참조를 대조했다. B 12건은 tokenizer ID와 이미지 격자의 pad 확장을 독립 계산해 전체 prompt ID까지 일치했다. 시험 프롬프트는 최소 1,058 / 중앙값 1,135 / 최대 1,432토큰, 풀 전체는 1,033 / 1,128 / 3,267토큰이다. [dry-run 기록](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_sampling_dry_run.json)

다음 100문항 × K=4 시험은 **400응답**, RTX 4090 24 GB 한 장에서 **약 25–35분**으로 추정한다. 이전 E1/E3 생성에서 관측한 약 694/593 출력 토큰/초와 평균 2,000 출력 토큰 가정을 사용했다. 가중치 다운로드는 별도이며 실제 M0 출력 길이에 따라 달라진다.

## 7. 보관과 정지점

작업은 로컬 CPU에서 완료했다. GPU 시험은 별도 단계이며, 전체 샘플링 K, 시작 모델, 학습 세트 비율, lr, β 및 팀 공식 여부는 결정 대상이다.

### 수정한 1단계 스크립트

| 파일 | 전 SHA-256 | 후 SHA-256 |
|---|---|---|
| `scripts/sample_m0.py` | `9104cc786411e75e4a1e7843195b91b950ea5c4bb3ed9ff24d3cccc1d214d178` | `475a4ddbd097743e718761a8edeeb741c11969893cc26019eae16e6dc26adc36` |
| `scripts/build_groups.py` | `1af71c428677c83445448fb94414616a0e85ed916aad1ac8f8687175b638ec8c` | `4d4074d857efc3cf31baa242173ef7eeba3ed9adba904ae871c15f0dac985f50` |

### 파일과 SHA-256

아래 경로는 R1 폴더 기준이다. 이미지 388개와 row group 영수증 259개를 포함한 전체 672개 파일은 체크섬 목록에 기록했다. 목록의 SHA-256은 `a55bd00ba1a60f39021dcbfe8aeaaf14406ef8c42a52bd43c0ce4f85b5204f90`다. 보고서 자체의 해시는 별도 `STAGE1B_REPORT.sha256`에 있다. [전체 체크섬](C:/Users/brain/Documents/ChatGPT/MMDL/work/train_RL_20261106/reports/stage1b_artifacts.sha256)

| 파일 | SHA-256 |
|---|---|
| `data/images_B_manifest.csv` | `ccaf9ddf741e888f8f2843fee8d4ad18834e338f0bbb323fbd85b073984c689a` |
| `data/r1_pool.jsonl` | `81b0485c14e9859df3e78dd69bf123ae80e187b7a61b117aa2d3359a6a757c96` |
| `data/stage1b/a_capped.jsonl` | `d20882873c41c85742f0812f5454211d8de3e5a97fb763321bb2de1e504b4d33` |
| `data/stage1b/b_mc_candidates.jsonl` | `e115ae5afb5a5774e41094fc7d7c1f10e04c6f23fa302e7a18c87dade1ce3562` |
| `data/stage1b/c_mc_candidates.jsonl` | `e2e2b4a1554d0289bab82c9395b43946c43d1a6870d4180755d914fd44d1c541` |
| `data/stage1b/image_candidates.jsonl` | `19f421db199de88b1c2ee73ddc3ba3df988cbbc2c1eea52c2c5ce982fca770d7` |
| `data/trial_100.jsonl` | `651d813f6fb75d534c3ef8a24640233e1f76d38bbe2537af70cca47805497e5f` |
| `reports/stage1b_a_cap.json` | `2eadc75fb1759be168adc929967aef1a4036eabf0b19af6f8fe163964b6977bc` |
| `reports/stage1b_c_content_review.json` | `afc1b694dda9c415832e6e70e69f4b29fc196e9ae85180fd1e684d0a192154cb` |
| `reports/stage1b_check_summary.json` | `b572d5b81b9c49acf4c365e0897274f5f51b75dce209281980c59d123ba52008` |
| `reports/stage1b_checks.jsonl` | `12b2ebd4f9b0bed08c5de1989d7c0657b2bdcb5beb6bc544df6dd0b7690c3b4a` |
| `reports/stage1b_contamination_pairs.json` | `44f32de5782f6aae495d86be164e1522861a0bf02b725db79016d756fc92a504` |
| `reports/stage1b_cs_flowchart.json` | `5b33fced707c0c5d06560c398b75189d75211e56a25c7447751b0dfde785bed2` |
| `reports/stage1b_image_receipt.json` | `dda7ad5003d2f2863c1ff28f3cc3c6da484c2508f1143919df0971d29c11ca56` |
| `reports/stage1b_image_transfer_plan.json` | `5af5f899226972570496a02c8b86ebe157280387155639a92841181b47b94ce1` |
| `reports/stage1b_prescreen.jsonl` | `2ecc14fb39a5b14dbc289bdadd4f87c9db3d42fd9fea5cc4293075ee5a07f56e` |
| `reports/stage1b_reward_check.json` | `116a5e965ecf6311f70c52f600e6d704586eee03e8a61daef3cafcb98cf0f6c7` |
| `reports/stage1b_sampling_dry_run.json` | `2bf5e92e220d42830987e4c1fbada83b97387e8b695656f32697c22ae026e8f3` |
| `reports/stage1b_selection.json` | `5ed43b074eb151755e36eeca2476252605042e5a68bbd3844a7da826cbe1161e` |
| `reports/stage1b_structure_exclusions.json` | `0dda170d5563af002b8cc4b2a61331c16418e8b757e01fc7871dbd1186cfdb49` |
| `scripts/build_groups.py` | `4d4074d857efc3cf31baa242173ef7eeba3ed9adba904ae871c15f0dac985f50` |
| `scripts/fetch_stage1b_images.py` | `dc3c068ebaf9473bf77ce15e09004bd20456afd7d71e6d08bba85578343568c9` |
| `scripts/finalize_pool_stage1b.py` | `654f69e70fdacd33facb409923647c3705666ef5739d96f08a8e9ab51862c466` |
| `scripts/prepare_pool_stage1b.py` | `8d6dcfadb50a5125fe313f043062c5176f192eb04b27eecbee7f98c7bbf8f87b` |
| `scripts/sample_m0.py` | `475a4ddbd097743e718761a8edeeb741c11969893cc26019eae16e6dc26adc36` |

# raw.jsonl 평가 및 8192·2048 비교

저장된 추론 결과를 동일한 규칙과 Judge 설정으로 평가한다. 모델 재추론은 하지 않는다. `prepare`, `summarize`, `compare`는 오프라인이며, `run`만 OpenAI API를 호출한다. 실행자는 본인 API 키를 사용한다.

이미 완료된 최신 900문항의 과제 보고서용 결과·실험 근거와 **API 없는 재현 명령**은 [제출 결과 묶음](../../assignment/experiments/submission/README.md)에 있다.

2026-09-23부터 `hybrid100_mc_qwen_ab_open_no_length_gate_v2`를 사용한다. 추론의 `finish_reason=length`만으로 Judge에 보내는 게이트를 제거했다. 기존 v1 결과 폴더는 그대로 보관하고 아래처럼 새 폴더에서 `prepare`한다. v1 결과의 재집계는 당시 코드(commit `518c12c916cdd194712e5db0340688b48042d62d`)로 수행한다. v2에서 v1 결과를 재개하거나 manifest를 수동 수정하지 않는다.

저장소 루트에서 Python 3.10 이상으로 실행한다.

```bash
python -m pip install -r code/scoring/requirements.txt
```

## 1. 입력 검증 및 요청 준비 — API 호출 없음

```bash
python code/scoring/evaluate.py prepare --config code/config.yaml --raw assignment/runs/draft/raw.jsonl --out code/scoring/outputs/8192_no_length_v2
python code/scoring/evaluate.py prepare --config code/config.yaml --raw assignment/runs/run_max_new_tokens2048/raw.jsonl --out code/scoring/outputs/2048_no_length_v2
```

900개 고유 ID, 30과목 × 30문항, 객관식 847개·주관식 53개, 생성 상태와 토큰 개수를 검증한다. 누락·오류 입력은 유효 문항만 골라 점수를 내지 않고 중단한다. 입력은 결과 폴더의 `input/raw.jsonl`에 그대로 복사하고 해시를 기록한다. 원본 `runs/*/raw.jsonl`은 수정하지 않는다.

Judge 대상 수는 각 입력에서 다시 계산한다. 8192 실행의 대상 수를 2048 실행에 고정하지 않는다. `code/config.yaml`의 `scoring` 설정을 사용하며, 기존 `budget.max_new_tokens`를 현재 raw의 생성 한도로 덮어씌우지 않는다. 입력 검증은 이 과제의 고정 MMMU validation 900문항과 현재 24필드 raw 형식을 대상으로 한다.

## 2. 본인 키로 Judge 실행

```bash
python code/scoring/evaluate.py run --out code/scoring/outputs/8192_no_length_v2 --ask-key
python code/scoring/evaluate.py run --out code/scoring/outputs/2048_no_length_v2 --ask-key
```

터미널의 숨김 입력에 API 키를 붙여 넣는다. 숨김 입력을 지원하지 않는 환경에서는 입력을 받지 않고 중단하므로 대화형 터미널을 사용한다. 키는 프로세스 메모리에서만 사용하며 파일에 저장하지 않는다. 이미 `OPENAI_API_KEY` 환경 변수가 있으면 `--ask-key`를 생략할 수 있다. 키를 코드·JSON·채팅·Git에 넣지 않는다.

처음 소량만 확인하려면 `run`에 `--limit 3`을 추가한다. 이후 같은 명령에서 `--limit`을 빼면 저장된 응답을 재사용하고 남은 문항을 진행한다. 완료된 `Z`도 재호출하지 않는다. 네트워크 오류나 호출 도중 중단으로 처리 여부가 불명확한 요청은 자동 재시도하지 않고 중단한다. 이러한 실행의 잠금·호출 기록을 지워 강제로 재개하지 말고 먼저 기록을 확인한다.

## 3. 오프라인 재집계 및 비교

```bash
python code/scoring/evaluate.py summarize --out code/scoring/outputs/8192_no_length_v2
python code/scoring/evaluate.py summarize --out code/scoring/outputs/2048_no_length_v2
python code/scoring/evaluate.py compare --left code/scoring/outputs/8192_no_length_v2 --right code/scoring/outputs/2048_no_length_v2 --out code/scoring/outputs/comparison_no_length_v2
```

전체·문항 유형·종료 사유·과목별 점수, 추출 실패, Judge 사용 토큰과 추정 비용을 집계한다. API 미완료 문항이 있으면 최종 정확도는 `null`이며 오답으로 합치지 않는다. MMMU 규칙 점수는 API와 독립적으로 확인할 수 있다.

비교 전 ID별 질문·정답·선택지·이미지 인덱스와 채점 설정이 같은지 확인한다. raw에는 모델 revision과 생성 설정 전체가 없으므로, 이 비교만으로 점수 차이의 원인이 오직 토큰 한도라고 확정할 수는 없다. 원인 분석에는 추론팀의 실제 실행 메타데이터를 함께 확인한다.

## 적용 정책

- 객관식: 고정 Qwen 규칙 + 사용자 정의 `Final Answer` 보완. 마지막 답 표기 뒤의 텍스트가 100자 이내이고 유효 표기들이 일치할 때만 보완한다. 생성 길이를 100자로 제한하는 규칙이 아니다.
- 종료 사유와 관계없이 규칙 추출 성공은 자동 처리하고, 답 미추출·Qwen과 보완 규칙의 충돌만 Judge로 보낸다. `finish_reason`은 검증과 종료 사유별 통계에 계속 사용한다.
- 객관식 Judge 요청에는 gold를 별도 필드로 전달하지 않는다. 고정 질문 문맥과 선택지, 생성 응답 전체를 전달한다.
- 주관식: `A=참조답 원문`, `B=Other Answers`로 변환한 Qwen 방식이다. 참조답을 보는 동치 판정이며 `Final Answer` 문자 보완은 적용하지 않는다.
- Judge가 정상 종료(`stop`)했을 때만 응답에서 선택지를 추출한다. 완료된 `Z`, 유효 답 없음, 비정상 종료는 추출 실패·오답이다. API 오류/미응답은 미완료다.
- 별도 진단값은 **MMMU 규칙 + 랜덤 폴백 제거** 점수다. 여기에는 Judge와 length gate를 적용하지 않는다. 리스트 형태로 직렬화된 주관식 정답은 안전한 literal 변환으로 MMMU의 대체 정답 목록에 전달한다.

기존 실험과 비교하기 위해 Qwen 프롬프트를 보존했다. 그 프롬프트는 출력 설명에 A–D/Z를 열거하지만 실제 문항 선택지가 더 많으면 그 선택지들도 전달·허용한다. 이 도구를 Qwen/MMMU 공식 evaluator 전체의 완전 재현이라고 부르지 않는다. 높은 채점 점수만으로 Judge의 추출 정확도가 검증되는 것도 아니다. 고정 소스와 라이선스는 [frozen/THIRD_PARTY.md](frozen/THIRD_PARTY.md)에 있다.

설정이나 코드를 바꾸면 새 결과 폴더로 준비한다. 실행 결과는 `outputs/` 아래에 두면 Git에서 제외된다. 다른 실행자의 결과를 섞거나 기존 요청/응답을 수동 수정하지 않는다.

## 오프라인 검증

```bash
python -m unittest discover -s code/scoring/tests -v
```

9개 테스트가 통과했다. 입력·해시 검증, 종료 사유별 동일 규칙 처리, Judge 비정상 종료, 캐시 재개·중복 호출 방지, 미완료 비교를 검사한다. HTTP 요청은 모의 응답을 사용한다.

## 검증 결과

### 현재 v2 — 2026-09-23

고정 GPT-4.1-mini 응답을 재사용해 세 입력의 요청 payload·request hash·raw text hash와 900행 원본을 검증했다.

| 입력 | 자동: v1 → v2 | Judge: v1 → v2 | 하이브리드 정답: v1 → v2 | MMMU 규칙 단독 |
|---|---:|---:|---|---:|
| `runs/draft/raw.jsonl` (8192) | 565 → 582 | 335 → 318 | 609 → **608/900 (67.56%)** | 450/900 (50.00%) |
| `runs/run_max_new_tokens2048/raw.jsonl` | 491 → 523 | 409 → 377 | **미완료: v2 Judge 377건** | 390/900 (43.33%) |
| [최신 제출 raw](../../assignment/experiments/submission/input/raw.jsonl) (8192) | 537 → 555 | 363 → 345 | 594 → **600/900 (66.67%)** | 452/900 (50.22%) |

두 8192 입력은 서로 다른 실행이다. 최신 raw와 대응 Judge 캐시·재현 결과는 [제출 묶음](../../assignment/experiments/submission/README.md)에 있다. 완료된 입력은 반복 집계 결과가 바이트 단위로 일치했고, 2048은 최종 정확도와 정확도 차이를 `null`로 유지했다.

| 입력 | SHA-256 |
|---|---|
| 저장소 8192 | `003d09dabceae4167c9158c3b028b5c89671060d72182f4514a51dd0ad617c69` |
| 저장소 2048 | `6371cffb954125393eed87a508ea42b35045ac96b6932a31c9565c05cfb7cc81` |
| 최신 제출 8192 | `ef23f0c49d9b1ae6c62b9625fbd52cce474a0c035c1a644cfa737e0967daa313` |

### 과거 v1 비교 — 2026-09-21

당시 저장소 commit은 `aa11a7488e397dfb06699a9d6be71d7128420d88`이다. `runs/draft`와 `runs/run_max_new_tokens2048`은 각각 900문항(객관식 847·주관식 53)이며, ID별 질문·정답·선택지·이미지 정보·prompt token 수가 일치했다. stop/length 구성은 각각 **792/108**, **666/234**였다.

`runs/draft`의 Judge 대상 335건은 기존 요청 payload와 request hash가 일치했다. 동일 저장 응답을 v1으로 재집계한 결과는 다음과 같다.

| Judge snapshot | 정답 수 | 저장 usage 기반 비용 추정 |
|---|---:|---:|
| gpt-4.1-mini-2025-04-14 | 609/900 | $0.487914 |
| gpt-3.5-turbo-0125 | 577/900 | $0.6130075 |
| gpt-4o-mini-2024-07-18 | 581/900 | $0.18296355 |

세 실행의 `summary.json`, `final_results.jsonl`, `REPORT.md`는 반복 집계 시 바이트가 일치했다. 당시 자동 테스트는 8개를 통과했다.

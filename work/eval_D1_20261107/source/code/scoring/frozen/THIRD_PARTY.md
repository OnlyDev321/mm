# 고정한 공개 소스

1. `vendor/qwen_eval_utils.py`: [Qwen3-VL eval_utils.py](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/eval_utils.py), commit `96588727e44c78b25ba03ea03b8e12f7e64fd0da`. Apache-2.0, `vendor/Qwen-LICENSE`.
2. `vendor/mmmu_eval.py`: [MMMU eval_utils.py](https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py), commit `51ce7f3e829c16bb44bc5445782686b4c3508794`. Apache-2.0, `vendor/MMMU-LICENSE`.
3. `vendor/vlmevalkit_matching.py`: [VLMEvalKit matching_util.py](https://github.com/open-compass/VLMEvalKit/blob/f71d47360cb884eee041f280c8e8095b6b051251/vlmeval/utils/matching_util.py), commit `f71d47360cb884eee041f280c8e8095b6b051251`. 라이선스는 `vendor/VLMEvalKit-LICENSE`. 로컬 보관본은 toolkit logger import만 Python logging으로 대체했다. 함수 본문과 정규식은 해당 commit 원문과 대조했다.

원본 소스를 보관하고 `parsers.py`에서 필요한 함수의 AST만 읽어 실행한다. Qwen의 API wrapper, random fallback, MMMU의 전역 `random.seed(42)`는 로드하지 않는다. 따라서 생성 seed 3407 설정을 바꾸지 않는다.

Qwen: `can_infer_option`, `can_infer_text`, `can_infer`, `build_prompt` 함수 본문은 그대로 사용한다. choices를 복사하여 원본 선택지가 소문자로 변하는 부작용을 격리한다. 선택지의 Python None은 누락값이며 문자열 `"None"`은 유효한 선택지다.

MMMU: MC/open 추출 및 채점 함수 본문은 그대로 사용한다. MC의 `random.choice` 진입 시 예외를 잡아 추출 실패로 기록한다. `np.argmax`는 첫 최댓값의 인덱스를 반환하는 표준 라이브러리 함수로 대체한다. 임의 답은 생성하지 않는다. 명칭은 반드시 **MMMU 규칙 + 랜덤 폴백 제거**로 쓴다.

VLMEvalKit: `can_infer_option`, `can_infer_text`, `can_infer` 및 `_VERBOSE_ANSWER_RE`를 그대로 읽는다. `VERBOSE`는 unset으로 고정하며 실제 프로세스 환경 변수는 바꾸지 않는다. Qwen의 과거 fork와 동일한 함수라고 가정하지 않는다. 이 커밋에는 선택지 문자 위치, 긴 본문 제한, answer-is 정규식 등 차이가 있다. open에는 실험의 공통 A/B 변환을 적용한다. toolkit 전체 MMMU evaluator의 완전 재현이라고 부르지 않는다.

Final Answer 100자 규칙은 우리 과거 실험의 사용자 정의 규칙이다. 원본 정규식과 조건을 보존했다. Qwen/MMMU 공식 규칙이라고 부르지 않는다. 예컨대 `Final Answer: A or B`도 A로 채택할 수 있는 약점이 있어 검증 대상이다.

이 도구는 Qwen 공식 evaluator 전체 재현물이 아니다. API 중첩 재시도와 랜덤 폴백을 실행하지 않고, hybrid routing은 별도 실험 정책으로 적용한다. 현재 v2는 length gate를 적용하지 않는다. `parsers.py`의 `analyze`는 과거 게이트 비교용 함수를 보존한 것이며 현재 실행기는 호출하지 않는다. 결과에 “공식 채점 그대로”라고 쓰지 않는다.

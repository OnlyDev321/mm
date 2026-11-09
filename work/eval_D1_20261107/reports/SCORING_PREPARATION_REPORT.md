# D1 평가 2단계 v2 채점 준비

고정 v2로 자동 **584건**을 처리했다. Judge 대상 **316건**은 전부 pending이며 전체 정확도는 `null`이다. Judge 호출과 API 비용은 0이다.

## 입력과 처리 경로

- `source/code/scoring/evaluate.py prepare`를 네트워크 연결을 막은 로컬 CPU에서 실행했다. 정책 `hybrid100_mc_qwen_ab_open_no_length_gate_v2`, 원본·준비 사본 공통 raw SHA-256 `60ee87670752b663edacf7a42f24320f97c33198741a6b1465594939361ba23e`다.
- Judge 설정은 `gpt-4.1-mini-2025-04-14`, temperature 0, top_p 1, seed 3407, max_tokens 8이다. 코드·설정·라우팅은 E3와 동일하다.

| 집단 | 문항 | 자동 | Judge 대상 | 규칙 미추출 | 충돌 |
|---|---:|---:|---:|---:|---:|
| 전체 | 900 | 584 | 316 | 314 | 2 |
| stop | 799 | 569 | 230 | 228 | 2 |
| length 전체 | 101 | 15 | 86 | 86 | 0 |
| length · 종료 태그 있음 | 0 | 0 | 0 | 0 | 0 |
| length · 시작 태그만 있음 | 0 | 0 | 0 | 0 | 0 |
| length · 두 태그 없음 | 101 | 15 | 86 | 86 | 0 |

M0의 자동/Judge 대상은 555/345건, E3는 737/163건이었다. 규칙 미추출은 Judge로 보내는 사유다. 최종 추출 실패 수와 정확도는 Judge 완료 전까지 미확정이다.

## 보수적 Judge 예산

예상 호출 **316회**, 예상 비용 **$1.49**(계산값 $1.4889484)다. E3와 같은 계산으로 요청 JSON UTF-8 3,671,811바이트에 요청당 128토큰을 더한 입력 3,712,259토큰과 최대 출력 2,528토큰을 적용했다. 캐시 할인은 적용하지 않았다. 고정 단가와 일치하는 공개 가격표에서 해당 스냅샷은 Deprecated로 표시된다. [OpenAI GPT-4.1 mini 가격표](https://developers.openai.com/api/docs/models/gpt-4.1-mini)

준비 결과는 `scoring/D1/`, 경로·예산·파일 해시는 `reports/SCORING_PREPARATION_SUMMARY.json`에 보관했다. Judge와 최종 판정은 사용자 확인 후 별도로 진행한다.

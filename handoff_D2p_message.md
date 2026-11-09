# D2' (2회차 DPO) 실험 인계

첨부: `handoff_D2p_20261107.zip` (16.5MB, SHA-256 `69f1b6b30401f4cad93d0847e92b9baad284328ac884921a943393db742dccb9`)

## 한 줄 요약

D1(DPO)으로 처음 하방 방어에 성공했다. 다만 효과가 1차 기준에 조금 못 미쳐서, D1 모델로 같은 과정을 한 번 더 하는 D2'를 부탁한다.

## 지금까지

- 목표: Qwen3-VL-4B-Instruct(M0, MMMU val 600/900)를 파인튜닝해 length 종료(8,192토큰에서 잘림) 문항의 정답을 늘리는 것
- SFT 3번(E1·E2·E3)은 모두 정답→오답이 많아 실패했다
- **D1 (DPO)**: M0 자기 응답 중 "끝까지 가서 맞힌 응답"을 "잘리거나 틀린 응답"보다 선호하도록 학습. 707쌍
  - 602/900
  - 정답→오답 54 / 오답→정답 56
  - length 128 → 101
  - length 128문항 net_gain +10 (기준 ≥14 미달)
- **D2'**: D1으로 1,500문항 재샘플링 → 쌍 구성 → D1에서 이어서 DPO. 설정은 D1과 같다

## 받을 것

1. **첨부 압축 파일**: 문서, 보고서, 평가 스크립트, tokenizer 캐시
   - 압축을 풀면 `MMDL/...` 구조가 나온다. 스크립트가 이 상대 경로를 쓰니 구조를 그대로 둔다
2. **HF 비공개 저장소** (MMDL-team4 org 권한 필요, 없으면 알려 달라)
   - `MMDL-team4/hypB-d1-dpo` (model), 브랜치 `run-yoonseok`: D1 adapter
   - `MMDL-team4/hypB-rl-data` (dataset), 브랜치 `run-yoonseok`: 1,500문항 풀, 이미지, M0 샘플링, D1 쌍, 스크립트
   - `MMDL-team4/hypB-train-data` v1.1: A 이미지
   - 받는 방법과 둘 경로는 `HANDOFF_D2p.md`의 "데이터 받기"에 있다

## 본인이 준비할 것

- RunPod: RTX 4090 (샘플링·평가), A100 80GB (학습). 총 약 $9
- RunPod에 SSH 키 등록, HF 로그인(`hf auth login`)
- OpenAI API 키: Judge 채점용, 약 $0.5
- 키와 토큰은 본인 것을 쓰고 파일에 저장하거나 공유하지 않는다

## 진행 순서

1. `CLAUDE.md` → `work/train_RL_20261106/PLAN_D2p.md` → `HANDOFF_D2p.md` 순서로 읽는다
2. 4090을 빌리고 아래 메시지로 에이전트를 시작한다. 단계마다 보고를 받고 다음으로 넘어간다
   1. 샘플링과 쌍 구성 (4090)
   2. 학습 (A100)
   3. 평가 (4090, `HANDOFF_D1_eval.md`와 같은 절차. `work/eval_D1_20261107/`을 참고하되 adapter만 D2'로 바꾼다)

## 꼭 지킬 것

- GitHub에 올리지 않는다. HF 업로드는 비공개로만, 확인받고 한다
- MMMU / MMMU-Pro는 학습에 쓰지 않는다
- validation 결과를 보고 설정(lr, β 등)을 바꾸지 않는다
- 쌍이 300개 미만이면 학습하지 말고 멈춘다 (계획에 미리 정한 규칙)
- 평가는 반드시 4090에서 한다 (M0와 조건 맞추기)
- GPU와 Judge처럼 비용이 드는 단계는 실행 전에 확인한다

## 첫 에이전트 메시지

```
D2'(2회차 DPO) 1·2단계 진행해.
지침: work/train_RL_20261106/HANDOFF_D2p.md (1·2단계)
계획: work/train_RL_20261106/PLAN_D2p.md

파드: ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519  (RTX 4090)

- 로컬에 데이터가 없으면 지침의 "데이터 받기"대로 HF에서 받고 SHA256SUMS.txt로 대조
- 접속 후 nvidia-smi로 4090 확인. 다르면 멈추고 보고
- D1 병합(adapter 해시 2d4bcce4… 대조, 병합 샤드 해시 기존 기록과 대조)
- sample_m0.py는 복사본(sample_policy.py)에 모델 경로 옵션만 추가. 원본 수정 금지
- 1,500×4 샘플링, R1 3단계와 같은 조건. 바꾸는 건 모델뿐
- 로컬로 받아 해시 대조 → 쌍 구성·검증
- 쌍 300 미만이면 학습 준비 없이 멈추고 보고
- 2단계 끝에서 멈추고 보고 (reports/D2P_STAGE1_REPORT.md). 학습은 시작하지 마
- GitHub 업로드, 키·토큰 출력, 파드 Stop/Terminate 금지
```

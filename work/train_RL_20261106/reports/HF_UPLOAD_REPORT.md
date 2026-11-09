# D1 HF 업로드 보고

2026-11-07. 두 저장소 모두 **private**, 브랜치 `run-yoonseok` (main에는 `.gitattributes`만).

| 저장소 | 최종 커밋 | 원격 파일 | 대조 |
|---|---|---:|---|
| `MMDL-team4/hypB-d1-dpo` (model) | `5c651e5c452c45af37bfd02ca90f2279b7b27747` | 20 + .gitattributes | 누락 0 / 추가 0 / 불일치 0 |
| `MMDL-team4/hypB-rl-data` (dataset) | `5d7d3888201f20c5e83b959a503b5c14d6c3058d` | 423 + .gitattributes | 누락 0 / 추가 0 / 불일치 0 |

- 대조: LFS 파일은 SHA-256, 일반 파일은 git blob SHA-1을 로컬 `hf_staging/`과 비교
- 처음에 브랜치를 `run-yunseok`으로 올렸다가(데이터셋 첫 커밋 `304d2ba2…`) 사용자 정정으로 `run-yoonseok`을 그 위에서 만들고 README 브랜치 표기를 고친 뒤 `run-yunseok`을 삭제했다
- 업로드 후 private 재확인. GitHub 커밋·push 없음. 토큰 출력·저장 없음

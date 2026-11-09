# D2' — Sampling & Pair Construction

- GPU: NVIDIA GeForce RTX 4090
- D1 adapter + 2 merged shards: SHA-256 verified (`d2p_weights_verified.json`).
- Sampling: 1,500 questions × K=4; 6,000 unique responses validated against the pool.
- Correct: 4388/6000 (73.13%). Length: 259/6000 (4.32%).
- Seed 3407; temperature 0.7; top_p 0.8; top_k 20; presence_penalty 1.5; max_new_tokens 8192; context 16384.

| Correct / 4 | D1 | M0 baseline (task.md) |
| --- | ---: | ---: |
| 0/4 | 159 | 162 |
| 1/4 | 140 | 140 |
| 2/4 | 175 | 159 |
| 3/4 | 206 | 249 |
| 4/4 | 820 | 790 |

## Groups and pairs

- Target: 307; retain/keep: 214; excluded/discard: 979.
- Training questions: 521; all mixed reward questions. All-correct and all-wrong questions are excluded by the original classifier.
- DPO pairs: **696**, meets ≥300 rule. At most 2/question, no reused responses, original text verified.
- Pair construction verifies image sizes and SHA-256.

## Token bound and configuration

- Conservative per-branch bound: 9740 tokens, from recorded vLLM prompt + completion + 32 reserve (including image tokens).
- This corrects task.md's illustrative sum of serialized chosen messages plus rejected text, which is not a DPO branch length.
- Prepared `scripts/d2p_config.json` from the original adapter's D1 config, with D1 merged start model and `dataset` set to the pair file.
- max_length: 10240. A100 memory pilot required: False.
- Exact ms-swift training-template encoding and the planned A100 memory checks remain prerequisites for training; no training was launched.

## Stage 8

Sampling and pair construction completed. Wait for user approval before A100 training. Pod left running. No GitHub/HF upload.
Checksums: `reports/d2p_artifacts_sha256.json`.

# Bước 3 — Huấn luyện DPO D2′ trên nền D1

Cập nhật: 2026-11-09. Đây là file giao việc cho agent thực hiện **bước 3**, thay cho hướng dẫn sampling và pair construction cũ.

## 1. Phạm vi và trạng thái

- Repo đã chứa dữ liệu D2′ và hồ sơ bước 1–2 trong `work/train_RL_20261106/`, giữ nguyên nội dung và checksum. Không cần bundle hoặc file đầu vào từ máy local sau clone.
- **Chưa thuê pod, chưa chạy training, chưa có xác nhận duyệt báo cáo bước 1–2 trong lần chuẩn bị này.** Yêu cầu chuẩn bị file không phải là lệnh chạy GPU.
- Khi người dùng xác nhận báo cáo và yêu cầu chạy bước 3, tiếp tục trong phạm vi đã được cho phép; không hỏi lại những phần đã được xác nhận.
- Kết thúc training sau khi lấy adapter cuối, merge, kiểm tra vLLM load và đóng gói kết quả để tải về; chỉ xác nhận bàn giao về máy người dùng khi đã đối chiếu hash thực tế. Evaluation là bước 4, thực hiện riêng trên RTX 4090.

Đọc trước khi thực hiện:

1. `content.md`, `process.md`, `CLAUDE.md`.
2. `work/train_RL_20261106/PLAN_D2p.md`, `HANDOFF_D2p.md` (mục 3).
3. `work/train_RL_20261106/reports/D2P_STAGE1_REPORT.md` và `d2p_token_check.json`.
4. `work/train_RL_20261106/reports/D1_STAGE3_TRAIN_REPORT.md`, `reports/d1_full/args.json` và `HANDOFF_D1.md` để tái dùng cách thiết lập, ghi log và merge.
5. `work/train_RL_20261106/reports/D2P_STAGE3_PREPARATION.json` để biết file đã nhập, checksum và các thiếu hụt local.

## Khởi động trên pod A100 80GB

Dữ liệu D2′, pair index, sampling raw/groups/metadata/manifest/checksums, config và báo cáo đều có trong repo. Agent không cần SCP đầu vào từ máy local.

Trong Bash trên pod:

```bash
cd /workspace
git clone https://github.com/OnlyDev321/mm.git
cd /workspace/mm
```

Nếu đã clone, dùng checkout hiện có và cập nhật `main` khi working tree sạch, không ghi đè thay đổi hoặc kết quả trên pod.

Trong terminal **Bash trên pod**, tại repo `/workspace/mm`, nhập HF token qua prompt ẩn:

```bash
set +x
read -r -s -p "HF token (read access to MMDL-team4): " HF_TOKEN
printf '\n'
export HF_TOKEN
export HF_HOME=/workspace/hf_cache
python3 -c 'import os; assert os.environ.get("HF_TOKEN"), "Missing HF_TOKEN"; print("HF_TOKEN is set; value hidden")'
```

Dùng token mới có quyền đọc các repo private của nhóm. Không chép token từng gửi trong chat vào file này. Không lưu token vào Git, shell startup file hoặc lệnh có giá trị token trực tiếp.

Khởi chạy agent từ **chính shell đã export** để tiến trình con thừa hưởng `HF_TOKEN`/`HF_HOME`. Export ở terminal khác, ở máy local hoặc ở một phiên SSH khác không tự truyền token sang tiến trình agent đang chạy. Nếu agent đã chạy, cấp biến môi trường qua cơ chế khởi chạy của phiên đó hoặc khởi chạy lại từ shell trên. Agent chỉ kiểm tra token có tồn tại, không chạy `env`, `printenv HF_TOKEN`, `set -x` hay in HTTP Authorization headers.

Sau khi đã duyệt báo cáo bước 1–2 và đồng ý dùng pod này để training, giao việc:

> Tôi đã duyệt báo cáo bước 1–2 và đồng ý chi phí pod A100 80GB hiện tại cho bước 3. Hãy thực hiện `task.md` đến hết bước 3. HF_TOKEN và HF_HOME đã có trong môi trường. Tự tải script, adapter, ảnh và base còn thiếu từ các nguồn đã chỉ định; kiểm tra hash, template và cấu hình rồi train 2 epoch. Không cần hỏi lại các thao tác tải, cài đúng môi trường, merge và training nằm trong phạm vi này. Dừng nếu thiếu quyền truy cập, sai hash, OOM/NaN hoặc cần thay đổi cấu hình; dừng sau báo cáo bước 3, không evaluation, upload hay Stop/Terminate pod.

Đây là mẫu lệnh để người dùng gửi khi bắt đầu trên pod, không phải xác nhận rằng training đã được thực hiện. Token chỉ cấp quyền tải dữ liệu, không thay thế xác nhận chạy GPU. Khi có lệnh thực thi và đủ điều kiện, agent tiếp tục tự động qua các mục 3–8; không chỉ báo thiếu file rồi kết thúc nếu có thể tải từ nguồn đã chỉ định.

## 2. Đầu vào bước 3 và hồ sơ bước 2

Workspace: `work/train_RL_20261106/`. Toàn bộ file trong bảng dưới có sẵn sau clone; giữ nguyên cấu trúc đường dẫn.

| Đường dẫn trong workspace | Mục đích | Có khi clone GitHub? |
| --- | --- | --- |
| `data/d2p_pairs_swift.jsonl` | 696 cặp dùng cho training | Có |
| `data/d2p_pairs_index.jsonl` | Ánh xạ cặp với câu hỏi và chosen/rejected trong raw | Có |
| `scripts/d2p_config.json` | Config D2′ đã chuẩn bị từ bước 2 | Có |
| `reports/D2P_STAGE1_REPORT.md` | Báo cáo sampling và tạo pairs | Có |
| `reports/d2p_artifacts_sha256.json` | Hash dữ liệu, config và sampling artifacts | Có |
| `reports/d2p_weights_verified.json` | Hash D1 adapter và merged shards đã ghi nhận | Có |
| `reports/d2p_token_check.json` | Cận độ dài và trạng thái kiểm tra template | Có |
| `runs/d2p_k4/raw.jsonl` | Đối chiếu nguyên văn, reward và token của chosen/rejected | Có |
| `runs/d2p_k4/groups.json` | Đối chiếu nhóm | Có |
| `runs/d2p_k4/d2p_train.jsonl` | Metadata 521 câu và ảnh, bao gồm hash ảnh | Có |
| `runs/d2p_k4/manifest.json` | Provenance sampling; đọc cùng receipt D1, không dựa riêng trường model của manifest | Có |
| `runs/d2p_k4/SHA256SUMS.json` | Hash đầu ra sampling | Có |
| `README.txt` | Ghi chú phạm vi bundle, không chứa weights hoặc ảnh | Có |

Giữ các file này nguyên bản. Không chạy lại bước 1–2 để tạo dữ liệu mới.

### File cần có trên pod training

- **Dữ liệu training:** `data/d2p_pairs_swift.jsonl` và 521 ảnh mà trường `images` tham chiếu. JSONL chứa prompt/chosen/rejected, không chứa nội dung ảnh.
- **Model:** base M0 và D1 adapter để merge lại D1 trên pod, kèm tokenizer/processor đúng revision.
- **Mã chạy và cấu hình:** `scripts/dpo_train.py`, dependencies và `scripts/d2p_config.json`.
- **Hồ sơ nhỏ để kiểm tra:** báo cáo bước 1–2, hash dữ liệu/model, token check; `runs/d2p_k4/d2p_train.jsonl` có metadata/hash của các ảnh cần dùng. Metadata này phục vụ kiểm tra, không truyền làm dataset training.

`data/d2p_pairs_swift.jsonl` là dataset truyền vào Trainer. Pair index và sampling artifacts phục vụ kiểm tra/provenance; chúng cũng có trên GitHub để agent kiểm tra trực tiếp trên pod.

**Sau clone:** đối chiếu toàn bộ 7 mục trong `reports/d2p_artifacts_sha256.json` và 3 mục trong `runs/d2p_k4/SHA256SUMS.json`. Không cần tải lại pairs từ HF, không tạo pairs mới và không yêu cầu file đầu vào từ máy local. Chỉ tải các phụ thuộc HF ở mục 3. `reports/D2P_STAGE3_PREPARATION.json` là receipt lịch sử chuẩn bị local, không phải kết quả preflight của pod hiện tại.

Kết quả kiểm tra local khi chuẩn bị:

- 13 file được chép; 7 hash trong artifact manifest và 3 hash sampling đều khớp; bản chép khớp nguồn.
- 6.000 response duy nhất trên 1.500 câu; 696 cặp từ 521 câu, đạt ngưỡng ≥300.
- Nội dung chosen/rejected trùng raw, chosen có reward ghi nhận bằng 1 và không kết thúc `length`, rejected có reward ghi nhận bằng 0; không tái dùng response, tối đa 2 cặp/câu.
- Đây là kiểm tra reward **đã lưu**, chưa phải tái tính bằng `reward_v2.py`.
- Có 521 đường dẫn ảnh duy nhất trong pairs; **cả 521 ảnh hiện thiếu ở local**. Hash ảnh tham chiếu có trong `d2p_train.jsonl`.
- Cận nhánh lớn nhất từ sampling là **9.740 token**; config giữ `max_length=10240`. **Chưa xác minh encoding thực tế bằng template ms-swift**.

## 3. Các phần agent tự tải từ Hugging Face

Những phần sau chưa có trên GitHub và phải tải trên pod bằng `HF_TOKEN`:

- Để chạy training: `scripts/dpo_train.py` và dependencies mà script import; `scripts/dpo_config.json` gốc để đối chiếu config D1 với config D2′ đã có.
- Để hoàn tất kiểm tra bước 2 còn thiếu: `scripts/reward_v2.py` và dependencies. Script reward dùng để tái kiểm tra pairs trên pod, không truyền làm dataset cho Trainer.
- D1 adapter đầy đủ trong `adapter/d1_full/`, gồm weights, adapter config và metadata cần thiết.
- 521 ảnh mà pairs tham chiếu; giữ đúng đường dẫn tương đối `work/...` đã lưu trong JSONL.
- Base M0 đúng revision và D1 merged weights trên pod sẽ dùng để train.

Nguồn đã được handoff chỉ định:

| Nguồn | Nội dung / đối chiếu |
| --- | --- |
| HF model private `MMDL-team4/hypB-d1-dpo`, branch `run-yoonseok` | D1 adapter đầy đủ → `adapter/d1_full/`; lấy config/hồ sơ training gốc để đối chiếu |
| HF dataset private `MMDL-team4/hypB-rl-data`, branch `run-yoonseok` | `dpo_train.py`, `dpo_config.json`, `reward_v2.py` và dependencies → `scripts/` theo layout nguồn; ảnh B/C về đường dẫn `images[].path` trong metadata |
| HF dataset private `MMDL-team4/hypB-train-data`, v1.1, commit prefix `1833bac3` | Ảnh nguồn A; chọn theo metadata và khôi phục đúng đường dẫn ảnh trong pairs |
| `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` | Base M0 cùng tokenizer/processor → cache `HF_HOME`; dùng để merge D1 trên pod |

Agent dùng `HF_TOKEN` từ môi trường để xác thực với Hugging Face, tải trực tiếp trên pod các file còn thiếu và tái dùng file đã có sau khi kiểm tra hash. Kiểm tra quyền truy cập cả ba repo private; nếu token thiếu hoặc bị từ chối, chỉ báo tên repo và lỗi đã loại thông tin nhạy cảm. Không yêu cầu người dùng gửi token trong chat.

Trước khi tải, liệt kê file của từng repo qua API Hugging Face và chốt revision thực tế. Đường dẫn bên trong HF phải tra từ listing/manifest thật, không đoán URL. Tải chọn lọc script và dependencies, toàn bộ thành phần adapter, tokenizer/processor và shards M0; với ảnh, chỉ cần 521 ảnh được pairs tham chiếu. Nếu ảnh nằm trong archive thì tải archive cần thiết và trích đúng file. Xác định nguồn ảnh bằng metadata/hash và manifests HF, không suy nguồn chỉ từ tên thư mục local.

Khôi phục đúng layout theo `HANDOFF_D2p.md`, đối chiếu `SHA256SUMS.txt` của từng nguồn và hash ảnh trong train metadata. Ghi full revision tải thực tế vào receipt. Nếu nguồn không có file cần thiết hoặc hash không khớp, báo rõ phần thiếu; không tự viết pipeline thay thế rồi coi là script gốc.

Chỉ tải dữ liệu/checkpoint cần thiết; không cần tải lại toàn bộ sampling M0 cho bước 3. Không đưa token hoặc API key vào file, lệnh được log hay báo cáo. Bước 3 không cần OpenAI Judge key.

## 4. Điều kiện trước khi chạy GPU

1. Đã có xác nhận cho báo cáo bước 1–2 và chi phí GPU, đang chạy trên pod đã được người dùng cho phép (hoặc có SSH nếu agent điều khiển từ local). File task cũ không được dùng làm ủy quyền cho bước 3.
2. GPU là một A100 hoặc H100 **80GB**, xác nhận bằng `nvidia-smi`; đủ disk/RAM để chứa M0, D1 merge, D2′ merge, cache và checkpoints.
3. Toàn bộ dữ liệu, script và ảnh cần thiết đã có; kiểm tra lại hash trên pod sau transfer.
4. Đã xác minh template, reward và phạm vi LoRA như các mục dưới.
5. Output lần chạy mới chưa chứa kết quả khác; không xóa hoặc ghi đè D1 hay lần D2′ trước.

Các điều kiện này dành cho lúc thực thi training, không ngăn hoàn tất chuẩn bị local.

## 5. Thiết lập pod và merge D1

Dùng layout dự kiến `/workspace/mm/`, workspace `/workspace/mm/work/train_RL_20261106/`. Nếu layout khác, ghi rõ mapping và kiểm tra đường dẫn; không sửa pairs gốc để chữa đường dẫn.

Tái dùng môi trường D1: Torch `2.8.0+cu128`, ms-swift `3.12.3`, transformers `4.57.1`, PEFT `0.17.1`, TRL `0.22.2`, accelerate `1.11.0`, qwen-vl-utils `0.0.14`; môi trường kiểm tra load dùng vLLM `0.11.0`. Dùng venv riêng `--system-site-packages` như D1, không tự thay Torch hoặc nâng phiên bản. Lưu package versions, pip check, GPU và lệnh chạy thực tế.

Merge **M0 + D1 adapter** sang bf16 trên pod training bằng quy trình D1 đã ghi nhận. Kiểm tra:

| File | SHA-256 bắt buộc |
| --- | --- |
| D1 `adapter_model.safetensors` | `2d4bcce4972b36b6ef49b0b73d46c9d72a3803f43632b6847ae6f4d65e0132e8` |
| D1 `model-00001-of-00002.safetensors` | `31d4561504484269e16e1cc00baef16d694cd44f25c9d282cfd45977caaab351` |
| D1 `model-00002-of-00002.safetensors` | `1280742e74baab97037603e9bc2912626f2c85d6347d9762a1286566b4b7f431` |

Đầu ra D1: `/workspace/models/D1`. Giữ tokenizer/processor đúng snapshot gốc và lưu receipt hash. Hash không khớp thì dừng trước training.

## 6. Hoàn tất kiểm tra còn thiếu và preflight training

Theo `content.md`, bước 1–2 đã tạo dữ liệu; bước 3 dùng lại kết quả đó. Không sampling hoặc tạo pairs lại. Handoff yêu cầu xác minh reward/template, nhưng báo cáo hiện tại còn thiếu các bằng chứng này nên cần bổ sung trước training.

### Kiểm tra hồ sơ bước 2 trực tiếp trên pod

- Dùng pairs, index, raw và train metadata có trong repo: xác minh ánh xạ, nguyên văn chosen/rejected, tối đa 2 cặp/câu và không reuse response.
- Tái tính reward bằng `reward_v2.py` gốc đã tải từ HF: chosen=1, rejected=0. Ghép bằng row/repeat trong index với raw và ground truth trong train metadata; giữ nguyên API/logic reward. Lưu báo cáo/hash inputs; phân biệt tái tính với việc đọc cờ reward đã lưu. Không cần lấy thêm raw/index từ máy local.

### Kiểm tra trên pod trước training

- Kiểm tra hash pairs sau transfer, số cặp ≥300, ảnh đúng đường dẫn/hash/kích thước, model và config đúng kế hoạch.
- Dùng đúng ms-swift `qwen3_vl` template/processor và thiết lập ảnh của D1 để encode **toàn bộ 696 cặp**; tính riêng độ dài prompt+chosen và prompt+rejected, gồm visual tokens và EOS theo preprocessing thực tế. Không cộng độ dài hai nhánh với nhau.
- Đối chiếu token độc lập ít nhất 5 cặp, gồm cặp dài nhất; kiểm tra prompt/ảnh nhất quán với sampling. Lưu trường hợp dài nhất và phân bố độ dài.
- Nếu nhánh dài nhất ≤10.240: giữ `max_length=10240`, không cần thêm pilot theo kế hoạch.
- Nếu 10.240 < nhánh dài nhất ≤12.288: báo cáo sai khác với cận sampling, đề xuất `max_length` đủ chứa và không quá 12.288; sau khi được xác nhận mới chạy memory pilot 20 optimizer step có cặp dài nhất. Báo cáo pilot và nhận xác nhận trước bản full; full bắt đầu lại từ D1, không tiếp tục weights pilot.
- Nếu >12.288: dừng và báo cáo. Không tự truncate, loại cặp hoặc sửa nội dung để qua kiểm tra.

Đối chiếu `scripts/d2p_config.json` với config D1 khôi phục từ nguồn gốc: thay đổi thí nghiệm chỉ ở start model D1, dataset D2′ và max_length nếu được chấp thuận. Chỉ định output mới riêng cho D2′ khi chạy; config bundle không có `output_dir`. Lưu config thực thi và diff, giữ config đã nhập nguyên bản.

Cấu hình cố định:

| Mục | Giá trị |
| --- | --- |
| Start model | `/workspace/models/D1`, bf16 |
| Reference | D1 merged, tắt **adapter D2′ mới**; không dùng M0 làm reference |
| Adapter khởi đầu / resume | `adapters=[]`, `ref_adapters=[]`, `ref_model=null`, `resume_from_checkpoint=null` |
| Dataset | `data/d2p_pairs_swift.jsonl` (`dataset`, không phải `custom_dataset`) |
| Phương pháp | ms-swift DPO, sigmoid, beta=0.1, rpo_alpha=1.0 |
| LoRA | r=16, alpha=32, dropout=0.05; 36 tầng LM, q/k/v/o/gate/up/down = 252 modules; ViT/merger cố định |
| Learning rate / scheduler | 5e-6 / cosine, warmup_ratio=0.03 |
| Batch | 1 × gradient_accumulation_steps 16, 1 GPU |
| Epoch / seed | 2 / 3407 (cả data_seed) |
| Length / truncation | 10240 / raise, trừ ngoại lệ được duyệt ở trên |
| Checkpoint | Chỉ dùng checkpoint cuối, không chọn theo validation |
| Khác | Giữ nguyên dropout, masking, shuffle và các giá trị còn lại của D1 |

Kiểm tra regex thực sự chọn đúng 252 modules trước khi train; lưu danh sách và trainable parameter counts. Đặt `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` cho tiến trình training.

## 7. Chạy full training và theo dõi

Khôi phục rồi đọc CLI thật của `scripts/dpo_train.py` (bao gồm `--help`) trước khi lập lệnh. Kiểm tra working directory và cách script resolve đường dẫn: ảnh trong JSONL tính từ repo root `/workspace/mm`, còn dataset trong config tính từ workspace RL; truyền đường dẫn tuyệt đối hoặc dùng cơ chế resolve có sẵn của script để cả hai đúng, không sửa JSONL gốc. Không suy đoán flag từ tên file. Lưu lệnh chính xác và args ms-swift thực tế vào `reports/d2p_full/`.

- Chạy đủ 2 epoch từ D1 merged với LoRA D2′ mới.
- Output training ở vùng D2′ riêng, ví dụ `outputs/d2p_full/`; không dùng output D1.
- Ghi từng optimizer step: loss, reward margin/accuracy, chosen/rejected logp của policy và reference, chosen NLL, grad norm, thời gian, VRAM allocated/reserved và `nvidia-smi`.
- Lưu stdout/stderr, timestamps, exit code, args, package versions, GPU và receipt input/script hashes.
- Thông báo tiến độ ngắn khi chạy lâu. D1 trước đây mất khoảng 2 giờ 54 phút trên A100 cho 707 cặp; đây chỉ là tham chiếu, không phải dự báo đã đo cho D2′.
- OOM, NaN/Inf, lỗi dữ liệu, sai hash hoặc LoRA/reference không đúng: dừng và báo cáo; không tự đổi lr, beta, batch, precision, LoRA hoặc max_length để chạy tiếp.

## 8. Checkpoint cuối, merge và bàn giao

1. Xác định checkpoint cuối từ Trainer state/log; không hard-code checkpoint-90 của D1.
2. Thu adapter cuối vào `adapter/d2p_full/`, kèm adapter config và config/provenance cần thiết.
3. Merge **D1 merged + D2′ adapter** sang bf16 vào `/workspace/models/D2p`. Không merge D2′ trực tiếp lên M0; không nạp D1 adapter lần nữa trên D1 merged.
4. Ghi hash adapter D2′ và toàn bộ merged shards; đây là hash mới, không so chúng với hash D1.
5. Kiểm tra load model D2′ bằng vLLM `0.11.0` theo cấu hình đánh giá chuẩn; **không gửi yêu cầu generation hoặc chạy benchmark** trên A100/H100.
6. Lưu receipt vị trí/hash merged model vào hồ sơ training để chuyển sang RTX 4090 ở bước 4. Không đưa merged model lớn vào ZIP bàn giao.
7. Hoàn tất `work/train_RL_20261106/reports/D2P_STAGE3_TRAIN_REPORT.md`: input hashes, 696 cặp, preflight template/reward, config diff, môi trường, LoRA/reference, epochs/steps, thời gian/VRAM, số liệu training, lỗi nếu có, checkpoint cuối, merge/load và trạng thái bàn giao. Nêu rõ chưa đánh giá MMMU; nếu chưa tải về máy người dùng thì ghi “chờ tải về local”.
8. **Bắt buộc đóng gói ZIP sau khi báo cáo và log đã ghi xong**, theo hướng dẫn dưới. Không yêu cầu thông tin đích nhận để hoàn tất việc tạo ZIP. Nếu thư mục kết quả đã tồn tại khác nội dung, dùng đường dẫn lần chạy riêng và phản ánh chính xác trong ZIP/báo cáo.
9. Dừng sau báo cáo training và gói ZIP đã kiểm tra. Chỉ đánh dấu transfer về local hoàn tất khi thực sự đối chiếu được hash pod/local. Không tự chạy bước 4, gọi Judge, upload GitHub/HF hoặc Stop/Terminate pod.

### ZIP kết quả để người dùng tải về local

Tạo trên pod:

```text
/workspace/mm/exports/D2P_STAGE3_results_<YYYYMMDD_HHMMSS>.zip
/workspace/mm/exports/D2P_STAGE3_results_<YYYYMMDD_HHMMSS>.zip.sha256
```

Dùng timestamp thực tế khi đóng gói để không ghi đè lần chạy khác. ZIP phải chứa đủ **ba phần**, giữ đường dẫn tính từ repo root để giải nén vào workspace local:

```text
work/train_RL_20261106/adapter/d2p_full/                  # toàn bộ adapter cuối, weights + config
work/train_RL_20261106/reports/d2p_full/                  # toàn bộ log, metrics, args/config, receipts
work/train_RL_20261106/reports/D2P_STAGE3_TRAIN_REPORT.md  # báo cáo cuối đã hoàn tất
SHA256SUMS.json                                         # SHA-256 từng file của ba phần trên
```

Agent thực hiện tuần tự:

1. Kiểm tra ba phần tồn tại, adapter có weights và `adapter_config.json`, log/metrics có nội dung và báo cáo đã hoàn tất; không đóng gói thiếu phần rồi báo thành công. Nếu kết quả là symlink đến checkpoint, đóng gói nội dung file thật, không tạo ZIP phụ thuộc đường dẫn checkpoint ngoài gói.
2. Tạo ZIP bằng `zipfile` của Python chuẩn hoặc công cụ ZIP sẵn có; chỉ lấy ba phần nêu trên và manifest. Không đưa M0/D1/D2′ merged model, dataset, ảnh, cache, venv, token hoặc file credentials vào ZIP.
3. Tính SHA-256 từng file và đưa manifest `SHA256SUMS.json` vào ZIP. Đóng ZIP rồi tính hash toàn bộ ZIP, lưu vào file `.zip.sha256` cạnh nó. Không ghi hash ZIP vào báo cáo nằm bên trong ZIP để tránh phụ thuộc vòng.
4. Mở lại ZIP, kiểm tra CRC (`ZipFile.testzip()` nếu dùng Python), danh sách file và SHA-256 nội dung so với manifest; kiểm tra đủ cả ba phần. Nếu lỗi, sửa hoặc báo lỗi, không thông báo gói sẵn sàng.
5. Trả cho người dùng **đường dẫn ZIP, dung lượng, SHA-256 và lệnh SCP cho cả ZIP lẫn `.zip.sha256`**. Dùng IP/port/key thực tế khi đã biết; nếu chưa biết, ghi rõ placeholder cần thay. Lệnh SCP chạy trên máy người dùng, không chạy trên pod để giả lập tải về.

Mẫu lệnh người dùng chạy ở local (agent thay timestamp và thông tin kết nối nếu có):

```bash
scp -P <PORT> -i <KEY_PATH> root@<IP>:/workspace/mm/exports/D2P_STAGE3_results_<TIMESTAMP>.zip .
scp -P <PORT> -i <KEY_PATH> root@<IP>:/workspace/mm/exports/D2P_STAGE3_results_<TIMESTAMP>.zip.sha256 .
shasum -a 256 -c D2P_STAGE3_results_<TIMESTAMP>.zip.sha256
```

File checksum ngoài ZIP phải dùng **tên ZIP, không dùng đường dẫn tuyệt đối trên pod**, để lệnh kiểm tra chạy được ở local. Giữ nguyên gói trên pod để người dùng tải; không tự xóa hay tắt pod.

## 9. Checklist hoàn tất

- [x] Đưa đủ 13 file dữ liệu/config/hồ sơ bước 1–2 vào repo; checksum giữ nguyên.
- [x] Ghi receipt chuẩn bị local và danh sách phụ thuộc còn thiếu.
- [ ] Báo cáo bước 1–2 và chi phí GPU đã được xác nhận; có pod/SSH phù hợp.
- [ ] Khôi phục script gốc, D1 adapter, 521 ảnh; kiểm tra hash và đường dẫn.
- [ ] Merge D1 trên pod và khớp hash chuẩn.
- [ ] Có báo cáo reward tái tính đạt trên pod; encoding ms-swift xác minh đầy đủ; giải quyết pilot nếu cần.
- [ ] Config thực thi đúng D1→D2′, LoRA 252 modules và reference D1.
- [ ] Full training 2 epoch hoàn tất, log hữu hạn và đủ metrics.
- [ ] Adapter cuối merge trên D1; vLLM load thành công, không generation.
- [ ] Báo cáo hoàn tất; ZIP chứa đủ ba phần và manifest, CRC/hash nội dung đã kiểm tra.
- [ ] Đã cung cấp đường dẫn ZIP, dung lượng, checksum và lệnh SCP tải về.
- [ ] Sau khi người dùng tải: adapter/log/báo cáo về local khớp hash; nếu chưa tải thì trạng thái bàn giao vẫn còn chờ.

Luôn giữ nguyên raw, pairs, D1 và pipeline benchmark gốc. Không dùng MMMU/MMMU-Pro để train, không đổi cấu hình hoặc chọn checkpoint theo validation. Người dùng đã cho phép đưa toàn bộ dữ liệu/hồ sơ D2′ trong mục 2 lên GitHub để bootstrap bước 3. Quyền này không bao gồm weights, token, ảnh tải từ HF hoặc kết quả training mới; giữ các phần đó ngoài Git và không tự upload.

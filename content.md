# Kế hoạch thực hiện DPO lần 2 (D2')

## 1. Tóm tắt bối cảnh

Nhóm đã chuyển hướng từ SFT sang RL (DPO - Direct Preference
Optimization). Đợt DPO lần 1 (D1) có cải thiện nhưng kết quả vẫn nằm
trong phạm vi sai số: tăng 2 câu, từ 600 lên 602/900 trên tập MMMU
validation. Kết quả này chưa đạt mục tiêu đề ra là khôi phục ít nhất 14
câu trong số 128 câu từng bị cắt do giới hạn độ dài.

Nhiệm vụ tiếp theo là thực hiện DPO lần 2 (D2' - Iterative DPO) theo quy
trình:

1.  Sinh mẫu từ mô hình D1.
2.  Xây dựng các cặp dữ liệu preference.
3.  Huấn luyện DPO tiếp trên nền D1.
4.  Đánh giá kết quả và chia sẻ báo cáo với nhóm.

## 2. Các bước thực hiện

### Bước 0. Chuẩn bị tài nguyên và dữ liệu

#### Tài nguyên cần có

-   **RunPod RTX 4090:** Dùng cho sampling và evaluation.
-   **RunPod A100 80GB:** Dùng cho training, chỉ thuê sau khi Bước 1 và
    Bước 2 được kiểm tra và báo cáo.
-   **OpenAI API Key:** Dùng cho mô hình Judge `gpt-4.1-mini-2025-04-14`
    trong khâu đánh giá; chi phí dự kiến khoảng 0,5 USD.
-   **Hugging Face Token:** Đảm bảo tài khoản có quyền truy cập
    Organization `MMDL-team4` để tải model và dataset private, nhánh
    `run-yoonseok`.

Chi phí GPU dự kiến khoảng 9 USD, cần kiểm tra lại giá thực tế trước khi
chạy.

#### Kiểm tra dữ liệu local

-   Kiểm tra các thư mục trong `work/train_RL_20261106/`.
-   Nếu thiếu dữ liệu, tải từ Hugging Face private repository.
-   Đối chiếu SHA-256 theo hướng dẫn trong `HANDOFF_D2p.md`.

### Bước 1. Sampling trên RunPod RTX 4090

**Mục đích:** Cho mô hình D1 sinh lời giải mới để tạo dữ liệu on-policy.

1.  Thuê một pod RTX 4090 trên RunPod và kết nối SSH.
2.  Merge mô hình gốc M0 (`Qwen3-VL-4B-Instruct`) với adapter D1 ở định
    dạng bf16.
3.  Kiểm tra SHA-256 của mô hình đã merge, đối chiếu với hash được ghi
    nhận (`2d4bcce4...`).
4.  Copy `sample_m0.py` thành `scripts/sample_policy.py`.
5.  Chỉ bổ sung tùy chọn đường dẫn mô hình D1; tuyệt đối không sửa file
    gốc `sample_m0.py`.
6.  Chạy inference bằng vLLM trên 1.500 câu hỏi, mỗi câu sinh `K=4` mẫu,
    tổng cộng 6.000 outputs.

Cấu hình sampling cần giữ giống D1:

  Tham số             Giá trị
  ----------------- ---------
  Temperature             0.7
  Top-p                   0.8
  Penalty                 1.5
  Max tokens             8192
  Số câu hỏi            1.500
  Số mẫu mỗi câu            4
  Tổng số outputs       6.000

7.  Chạy `build_groups.py` để phân loại câu hỏi thành `Target`, `Keep`
    và `Discard`.
8.  Tạo tập train `d2p_train.jsonl`.
9.  Tải kết quả về máy local và kiểm tra checksum bằng
    `SHA256SUMS.json`.

### Bước 2. Pair construction trên CPU local

**Mục đích:** Tạo các cặp preference gồm `Chosen` và `Rejected`.

-   `Chosen`: câu trả lời đúng và không bị cắt do giới hạn độ dài.
-   `Rejected`: câu trả lời bị cắt do giới hạn độ dài hoặc trả lời sai,
    theo quy tắc trong script và kế hoạch đã thống nhất.

Các công việc:

1.  Chạy `build_dpo_pairs.py` với dữ liệu đầu vào từ Bước 1.
2.  Xuất dữ liệu thành `data/d2p_pairs_swift.jsonl`.
3.  Đếm số lượng cặp được tạo.

> **Quy tắc dừng bắt buộc:** Nếu tổng số cặp nhỏ hơn 300, dừng ngay và
> báo cáo cho nhóm. Không tự ý tiến hành training.

4.  Nếu có từ 300 cặp trở lên, tiếp tục kiểm tra độ dài dữ liệu.
5.  Kiểm tra `max_length`. Nếu chuỗi dài nhất không vượt quá 10.240
    token thì giữ cấu hình 10.240.
6.  Tạo cấu hình training `scripts/d2p_config.json`. Chỉ thay đổi đường
    dẫn checkpoint đầu vào (D1) và đường dẫn dữ liệu D2'.
7.  Lập bảng so sánh thống kê mẫu giữa M0 và D1.
8.  Viết báo cáo `reports/D2P_STAGE1_REPORT.md`.
9.  Dừng tại đây và gửi báo cáo Bước 1, Bước 2 cho nhóm/người phụ trách
    xác nhận trước khi thuê hoặc bật máy A100 để training.

### Bước 3. Huấn luyện DPO D2' trên A100/H100 80GB

**Chỉ thực hiện sau khi Bước 1 và Bước 2 đã được báo cáo và xác nhận.**

1.  Thuê pod A100 80GB (hoặc H100 80GB nếu kế hoạch cho phép).
2.  Merge lại D1 trên pod này và kiểm tra hash.
3.  Chạy training DPO trong 2 epoch bằng framework `ms-swift`.

Cấu hình đã nêu trong kế hoạch:

  Tham số           Giá trị
  --------------- ---------
  LoRA rank              16
  Learning rate      `5e-6`
  Beta                `0.1`
  Epochs                  2

Không tự ý thay đổi hyperparameters đã được thống nhất.

4.  Sau khi training, lấy checkpoint cuối cùng.
5.  Merge checkpoint vào mô hình D1.
6.  Kiểm tra khả năng load mô hình bằng vLLM.
7.  Tải adapter và log về local.
8.  Viết báo cáo `reports/D2P_STAGE3_TRAIN_REPORT.md`.

### Bước 4. Đánh giá trên RunPod RTX 4090

**Mục đích:** Đánh giá mô hình D2' trên 900 câu MMMU validation và so
sánh với M0, D1.

1.  Chuyển sang pod RTX 4090.
2.  Dùng đúng RTX 4090 để duy trì tính nhất quán với baseline M0.
3.  Chạy pipeline đánh giá trên 900 câu MMMU validation.
4.  Sử dụng OpenAI API với model Judge `gpt-4.1-mini-2025-04-14` để chấm
    điểm.
5.  Tổng hợp kết quả và kiểm tra các tiêu chí thành công.

#### Tiêu chí đánh giá

**Tiêu chí chính**

Trong 128 câu từng bị tràn độ dài ở M0, số câu được khôi phục điểm ròng
(`net_gain`) phải đạt ít nhất 14.

`net_gain = số câu sai thành đúng - số câu đúng thành sai`

**Tiêu chí an toàn**

Tổng điểm trên 900 câu không được giảm quá 1,7 điểm phần trăm so với M0.

-   Baseline M0: 600/900 = 66,67%.
-   Mức giảm tối đa theo tiêu chí: 1,7 điểm phần trăm.

**Cơ chế kiểm tra**

Số câu bị cắt do giới hạn độ dài phải nhỏ hơn 128.

6.  Nếu được nhóm đồng ý, đẩy adapter lên private Hugging Face
    repository.
7.  Chia sẻ kết quả và báo cáo với cả nhóm.

## 3. Các nguyên tắc quan trọng (Do & Don't)

### Không được làm

-   Không commit hoặc push dữ liệu hay adapter lên GitHub công khai.
-   Không sử dụng tập MMMU hoặc MMMU-Pro để huấn luyện, nhằm tuân thủ
    quy chế đồ án.
-   Không tự ý điều chỉnh hyperparameters (learning rate, beta, v.v.)
    khi kết quả validation không tốt.
-   Không in API key hoặc Hugging Face token ra console.
-   Không lưu API key hoặc token vào Git.

### Cần đặc biệt chú ý

-   Kiểm tra kỹ trước khi chạy các bước phát sinh chi phí GPU hoặc API
    Judge.
-   Không khởi chạy training A100 trước khi Bước 1 và Bước 2 được báo
    cáo, xác nhận.
-   Tuân thủ quy tắc dừng nếu tạo được dưới 300 cặp preference.
-   Giữ nguyên các script gốc, đặc biệt là `sample_m0.py`.

## 4. Hành động tiếp theo

1.  Xác nhận với thành viên nhóm rằng đã nắm rõ quy trình D2' theo
    `HANDOFF_D2p.md`.
2.  Chuẩn bị RunPod RTX 4090.
3.  Thực hiện Bước 1 (Sampling) và Bước 2 (Pair construction).
4.  Kiểm tra số lượng cặp, checksum và các thống kê cần thiết.
5.  Hoàn thành `D2P_STAGE1_REPORT.md`.
6.  Gửi báo cáo cho nhóm/người phụ trách và chờ xác nhận trước khi tiến
    hành training trên A100.

### Tin nhắn xác nhận với nhóm

> Tôi đã nắm rõ quy trình D2' từ `HANDOFF_D2p.md`. Tôi sẽ bắt đầu chuẩn
> bị RunPod RTX 4090 để chạy Bước 1 (Sampling) và Bước 2 (Pair
> construction), sau đó gửi báo cáo `D2P_STAGE1_REPORT.md` cho bạn trước
> khi train A100.

------------------------------------------------------------------------

## 5. Checklist thực hiện

-   [ ] Xác nhận quyền truy cập Hugging Face Organization `MMDL-team4`.
-   [ ] Kiểm tra dữ liệu trong `work/train_RL_20261106/`.
-   [ ] Đối chiếu SHA-256 của dữ liệu và checkpoint.
-   [ ] Chuẩn bị RunPod RTX 4090.
-   [ ] Merge M0 + adapter D1 và xác minh hash.
-   [ ] Tạo `scripts/sample_policy.py` mà không sửa `sample_m0.py`.
-   [ ] Sampling 1.500 câu hỏi, 4 mẫu/câu.
-   [ ] Chạy `build_groups.py` và tạo `d2p_train.jsonl`.
-   [ ] Tải dữ liệu về local và xác minh checksum.
-   [ ] Chạy `build_dpo_pairs.py`.
-   [ ] Kiểm tra số cặp: nếu dưới 300 thì dừng và báo cáo.
-   [ ] Kiểm tra `max_length` và tạo `scripts/d2p_config.json`.
-   [ ] Viết `reports/D2P_STAGE1_REPORT.md`.
-   [ ] Gửi báo cáo và chờ xác nhận trước khi training.
-   [ ] Chỉ sau khi được xác nhận mới chuẩn bị A100/H100 và thực hiện
    DPO training.
-   [ ] Đánh giá D2' trên RTX 4090.
-   [ ] Hoàn thành báo cáo và chia sẻ kết quả với nhóm.

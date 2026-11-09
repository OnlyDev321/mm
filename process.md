# D2P -- Process Workflow

## 1. Mục tiêu

Tài liệu này mô tả cách phân chia thư mục và quy trình thực hiện D2P lần
2 (D2') từ bước Sampling, xây dựng DPO pairs, Training đến Evaluation.

Nguyên tắc quan trọng: **không chỉnh sửa trực tiếp pipeline benchmark
chuẩn trong thư mục `MMDL/`**. Các bước chuẩn bị dữ liệu và training
được thực hiện trong `work/train_RL_20261106/`; thư mục `MMDL/` được sử
dụng ở bước Evaluation.

------------------------------------------------------------------------

## 2. Phân chia thư mục theo từng bước

### Bước 1 -- Sampling

**Thư mục làm việc chính:** `work/train_RL_20261106/`

Các công việc:

-   Lấy 1.500 câu hỏi và ảnh từ Hugging Face private dataset:
    -   Repository: `MMDL-team4/hypB-rl-data`
    -   Branch: `run-yoonseok`
-   Đặt dữ liệu và các script sampling trong thư mục làm việc.
-   Sử dụng `sample_m0.py` làm cơ sở và copy thành `sample_policy.py`
    khi cần sampling từ policy model.
-   Lưu kết quả sampling trong `runs/d2p_k4/`.

**Đầu ra dự kiến:**

-   Các kết quả sampling trong `runs/d2p_k4/`.
-   Dữ liệu đầu vào cần thiết cho bước tính reward và xây dựng groups.

### Bước 2 -- Reward và xây dựng DPO pairs

**Thư mục làm việc chính:** `work/train_RL_20261106/`

Các script liên quan:

-   `reward_v2.py`: tính reward cho các kết quả sampling.
-   `build_groups.py`: gom các kết quả theo group để chuẩn bị so sánh.
-   `build_dpo_pairs.py`: tạo các cặp preference dùng cho DPO.

Các công việc:

1.  Chạy tính reward cho dữ liệu đã sampling.
2.  Xây dựng groups từ các kết quả và reward.
3.  Tạo DPO preference pairs.
4.  Kiểm tra định dạng và chất lượng dữ liệu trước khi training.

**Đầu ra dự kiến:**

-   `data/d2p_pairs_swift.jsonl`
-   Báo cáo: `reports/D2P_STAGE1_REPORT.md`

### Bước 3 -- DPO Training

**Thư mục làm việc chính:** `work/train_RL_20261106/`

Các công việc:

-   Sử dụng dữ liệu `data/d2p_pairs_swift.jsonl` đã tạo ở Bước 2.
-   Chạy `dpo_train.py` bằng framework `ms-swift`.
-   Lưu checkpoint và log training trong khu vực output được cấu hình
    cho lần chạy D2P'.

**Lưu ý:**

-   Kiểm tra đúng model đầu vào, dataset, cấu hình LoRA/DPO và đường dẫn
    output trước khi chạy.
-   Không ghi đè checkpoint hoặc kết quả của các lần thí nghiệm trước.
-   Ghi lại cấu hình thực tế và thông tin checkpoint để có thể tái lập
    thí nghiệm.

### Bước 4 -- Evaluation

**Thư mục/framework sử dụng:** `MMDL/code/`

Sau khi hoàn thành training, sử dụng pipeline benchmark chuẩn trong
`MMDL/code/` để đánh giá model D2P' trên 900 câu hỏi thuộc MMMU
validation.

Các thành phần trong base code có thể bao gồm:

-   `code/run_generate.sh`
-   `code/check_env.py`
-   `code/scoring/`
-   Các script và cấu hình evaluation liên quan.

Mục tiêu:

-   Đánh giá model sau DPO trên cùng benchmark MMMU validation.
-   So sánh kết quả với baseline M0 của nhóm.
-   Đảm bảo cách đánh giá nhất quán để kết quả có thể so sánh công bằng.

**Nguyên tắc:** Giữ nguyên cấu trúc và các file chuẩn trong `MMDL/`.
Không tự ý sửa pipeline benchmark dùng chung. Nếu cần thay đổi cấu hình
để chạy model mới, ưu tiên truyền tham số hoặc tạo cấu hình riêng thay
vì sửa trực tiếp code chuẩn.

------------------------------------------------------------------------

## 3. Bảng tóm tắt

  ----------------------------------------------------------------------------------------------
  Bước              Công việc         Thư mục chính               Script / dữ liệu quan trọng
  ----------------- ----------------- --------------------------- ------------------------------
  1                 Sampling          `work/train_RL_20261106/`   `sample_m0.py`,
                                                                  `sample_policy.py`

  2                 Reward & Pair     `work/train_RL_20261106/`   `reward_v2.py`,
                    Construction                                  `build_groups.py`,
                                                                  `build_dpo_pairs.py`

  3                 DPO Training      `work/train_RL_20261106/`   `dpo_train.py`,
                                                                  `data/d2p_pairs_swift.jsonl`

  4                 Evaluation        `MMDL/code/`                `run_generate.sh`,
                                                                  `check_env.py`, `scoring/`
  ----------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 4. Cấu trúc thư mục tham khảo

``` text
work/
└── train_RL_20261106/
    ├── sample_m0.py
    ├── sample_policy.py
    ├── reward_v2.py
    ├── build_groups.py
    ├── build_dpo_pairs.py
    ├── dpo_train.py
    ├── runs/
    │   └── d2p_k4/
    ├── data/
    │   └── d2p_pairs_swift.jsonl
    └── reports/
        └── D2P_STAGE1_REPORT.md

MMDL/
└── code/
    ├── run_generate.sh
    ├── check_env.py
    └── scoring/
```

> Đây là cấu trúc minh họa theo các đường dẫn đã nêu; các file hoặc thư
> mục phụ thuộc cấu hình thực tế có thể khác.

------------------------------------------------------------------------

## 5. Checklist trước khi hoàn tất

-   [ ] Đã lấy đúng dataset và branch `run-yoonseok`.
-   [ ] Đã lưu dữ liệu sampling trong `runs/d2p_k4/`.
-   [ ] Đã chạy reward và kiểm tra kết quả.
-   [ ] Đã tạo `data/d2p_pairs_swift.jsonl`.
-   [ ] Đã kiểm tra số lượng, định dạng và chất lượng DPO pairs.
-   [ ] Đã lưu báo cáo `reports/D2P_STAGE1_REPORT.md`.
-   [ ] Đã chạy DPO training bằng `ms-swift`.
-   [ ] Đã lưu checkpoint và log training.
-   [ ] Đã đánh giá trên 900 câu MMMU validation bằng pipeline
    `MMDL/code/`.
-   [ ] Đã so sánh kết quả D2P' với baseline M0.
-   [ ] Đã tổng hợp và chia sẻ kết quả cùng dữ liệu cần thiết cho nhóm.

------------------------------------------------------------------------

## 6. Ghi chú quan trọng

-   `work/train_RL_20261106/` là workspace cho Sampling, Reward, Pair
    Construction và DPO Training.
-   `MMDL/` là base code/pipeline benchmark chuẩn, được dùng chủ yếu ở
    bước Evaluation.
-   Không nên hiểu rằng Bước 1--3 hoàn toàn không liên quan đến model
    hoặc code gốc; chúng vẫn có thể cần tham chiếu model, tokenizer hoặc
    cấu hình đã được nhóm chuẩn hóa. Tuy nhiên, theo cách phân chia
    workflow hiện tại, các script và output của những bước này được quản
    lý trong workspace RL.
-   Các tên file và đường dẫn trong tài liệu phản ánh thông tin workflow
    đã cung cấp. Hãy đối chiếu với repository và cấu hình thực tế trước
    khi chạy lệnh hoặc ghi đè dữ liệu.

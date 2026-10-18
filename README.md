# 🥕 Hệ thống Phân loại Rau Củ Quả (Vegetable & Fruit Classification)

Đồ án học phần **Học máy** — phân loại ảnh rau/củ/quả bằng Deep Learning (CNN & Transfer Learning), có pipeline ML đầy đủ từ EDA đến demo application.

## Tổng quan

```
Dataset (Kaggle) → EDA → Preprocessing → Augmentation → Training
    → Evaluation → So sánh model → Chọn Final Model → Inference → Streamlit Demo
```

- **Dataset:** [Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) (Kaggle, ~36 class rau/củ/quả)
- **Framework:** TensorFlow / Keras
- **Model 1 — Baseline:** CNN tự xây (3 khối Conv2D-BN-MaxPool)
- **Model 2 — Transfer Learning:** MobileNetV2 (pretrained ImageNet)
- **Demo:** Streamlit web app

Xem lý do lựa chọn công nghệ chi tiết trong phần [Quyết định kỹ thuật](#quyết-định-kỹ-thuật) bên dưới.

## Cấu trúc project

```
vegetable-classifier/
├── configs/config.py          # TOÀN BỘ hyperparameter & đường dẫn — sửa ở đây, không hard-code nơi khác
├── data/raw/                  # Dataset (không commit — tải về theo hướng dẫn bên dưới)
├── notebooks/EDA.ipynb        # Exploratory Data Analysis
├── src/
│   ├── data/                  # dataset.py (loader + class_weight), augmentation.py, eda.py
│   ├── models/                # baseline_cnn.py, transfer_model.py
│   ├── training/train.py      # Training pipeline (cả 2 model)
│   ├── evaluation/            # metrics.py, confusion_matrix.py, error_analysis.py,
│   │                          # plot_history.py (training curves), report.py, evaluate.py
│   ├── inference/predict.py   # Pipeline dự đoán 1 ảnh (dùng chung cho CLI & Streamlit)
│   └── utils/logger.py
├── app/streamlit_app.py       # Demo application (3 tab: Dự đoán / Model / Đánh giá)
├── scripts/download_dataset.py
├── tests/                     # Unit test (chạy độc lập với dataset thật, dùng dữ liệu giả lập)
├── checkpoints/                # Model đã train (.keras) — sinh ra sau khi train
├── reports/                    # Metrics, confusion matrix, biểu đồ, so sánh model
├── requirements.txt
└── TRAINING_GUIDE.md          # Hướng dẫn train chi tiết từng bước
```

## Cài đặt nhanh

```bash
# 1. Tạo virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Tải dataset (xem chi tiết trong TRAINING_GUIDE.md)
python scripts/download_dataset.py

# 4. Chạy EDA
jupyter notebook notebooks/EDA.ipynb

# 5. Train model
python -m src.training.train --model both

# 6. Đánh giá & so sánh 2 model (sinh luôn reports/evaluation_report.md)
python -m src.evaluation.evaluate --compare

# 7. Chạy demo
streamlit run app/streamlit_app.py
```

## Tính năng chính

**Pipeline ML**

- EDA đầy đủ: class distribution, ảnh mẫu, phân bố kích thước, phát hiện ảnh lỗi & class imbalance
- Preprocessing tách bạch train/val/test — augmentation CHỈ áp dụng cho train (không data leakage)
- Tự động xử lý class imbalance bằng `class_weight` khi phát hiện dataset lệch
- Transfer learning 2 giai đoạn: feature extraction → fine-tuning với learning rate nhỏ
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

**Đánh giá & báo cáo**

- Accuracy, Precision, Recall, F1 (macro & weighted), Classification Report per-class
- Confusion matrix heatmap + tự động tìm các cặp class hay bị nhầm
- Error analysis: hiển thị ảnh dự đoán đúng/sai kèm confidence
- Biểu đồ Training vs Validation (Accuracy & Loss) + phân tích overfitting/underfitting tự động
- Báo cáo Markdown tổng hợp (`reports/evaluation_report.md`) sinh tự động

**Demo application**

- 3 tab: Dự đoán · Thông tin Model · Kết quả đánh giá
- Upload + preview ảnh, dự đoán kèm confidence và Top-K, lịch sử dự đoán trong phiên
- Chuyển đổi giữa 2 model để so sánh trực tiếp
- Xử lý lỗi đầy đủ: ảnh hỏng, sai định dạng, chưa có model, chưa có kết quả đánh giá

Hướng dẫn **chi tiết từng bước kèm giải thích** (kể cả khi bạn chưa quen deep learning) nằm trong [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) — bắt đầu từ đó nếu đây là lần đầu bạn train model này.

## Quyết định kỹ thuật

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Framework | TensorFlow/Keras | API cao cấp (Keras), tài liệu tiếng Việt/Anh phong phú, tích hợp sẵn `MobileNetV2` pretrained, phù hợp trình độ sinh viên. |
| Model 1 | CNN tự xây (baseline) | Bắt buộc theo yêu cầu môn học — có kết quả tham chiếu, thể hiện hiểu biết kiến trúc CNN cơ bản. |
| Model 2 | MobileNetV2 (transfer learning) | Dataset nhỏ (~100 ảnh/lớp) — train CNN sâu từ đầu dễ overfitting nặng. MobileNetV2 nhẹ, đã học đặc trưng thị giác tổng quát từ ImageNet, fine-tune hiệu quả với ít dữ liệu. |
| Demo UI | Streamlit | Đơn giản, 1 file, không cần frontend/backend tách riêng — đúng theo mức độ phức tạp phù hợp với đồ án môn học (tránh over-engineering). |
| Input size | 224×224 | Kích thước input mặc định của MobileNetV2 — dùng chung cho cả 2 model để pipeline preprocessing thống nhất. |

## Testing

```bash
pytest tests/ -v
```

Test suite dùng dataset giả lập (sinh tự động trong lúc test) nên chạy được **mà không cần** dataset thật hay GPU — dùng để kiểm tra logic pipeline (data loading, augmentation, model build, training, inference, xử lý lỗi ảnh không hợp lệ...) luôn đúng.

## Kết quả model (điền sau khi train trên dataset thật)

Bảng so sánh model đầy đủ được tự động sinh ra tại `reports/model_comparison.csv` sau khi chạy `python -m src.evaluation.evaluate --compare`. Xem `reports/*_summary.json` để biết class dễ nhầm lẫn, dấu hiệu overfitting/underfitting, và các phân tích khác.

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) | Training Time |
|---|---:|---:|---:|---:|---:|
| CNN Baseline | _(điền sau khi train)_ | | | | |
| MobileNetV2 Transfer | _(điền sau khi train)_ | | | | |

## Giấy phép & nguồn dữ liệu

Dataset thuộc về tác giả gốc trên Kaggle (kritikseth). Project này phục vụ mục đích học tập trong học phần Học máy.

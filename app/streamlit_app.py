"""Streamlit demo application — Hệ thống phân loại rau củ quả.

Chạy:
    streamlit run app/streamlit_app.py

Ứng dụng được thiết kế như một application hoàn chỉnh (không phải giao diện thử
nghiệm sơ sài), gồm 3 tab:

    1. Dự đoán       — upload ảnh, preview, predict, confidence, Top-K
    2. Thông tin Model — architecture, parameters, hyperparameters, danh sách class
    3. Kết quả đánh giá — metrics trên test set, confusion matrix, training curves

App luôn dùng ĐÚNG final model đã được đánh giá trong `src/evaluation` — số liệu
hiển thị ở tab 3 được đọc trực tiếp từ `reports/`, nên không thể lệch với model
đang chạy inference ở tab 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Cho phép chạy `streamlit run app/streamlit_app.py` từ bất kỳ đâu
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CONFIG  # noqa: E402
from src.inference.predict import VegetableClassifier, InvalidImageError  # noqa: E402

st.set_page_config(
    page_title="Phân loại Rau Củ Quả",
    page_icon="🥕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Bảng màu — dùng chung cho biểu đồ Top-K và các thành phần UI.
# Đây là bảng màu đã kiểm định đảm bảo người bị mù màu vẫn phân biệt được,
# và đủ tương phản trên cả nền sáng lẫn nền tối.
# ---------------------------------------------------------------------------
ACCENT = "#2a78d6"          # màu nhấn — dành cho kết quả dự đoán đứng đầu
MUTED_BAR = "#c3c2b7"       # màu xám — các kết quả còn lại (đóng vai trò ngữ cảnh)
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

CUSTOM_CSS = """
<style>
  .app-header {
      padding: 1.5rem 1.75rem;
      border-radius: 14px;
      background: linear-gradient(135deg, #1baf7a 0%, #2a78d6 100%);
      color: #ffffff;
      margin-bottom: 1.25rem;
  }
  .app-header h1 { margin: 0; font-size: 1.75rem; font-weight: 700; }
  .app-header p  { margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }

  .result-card {
      border: 1px solid rgba(11,11,11,0.10);
      border-radius: 14px;
      padding: 1.25rem 1.5rem;
      background: #fcfcfb;
  }
  .result-label {
      font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em;
      color: #52514e; margin-bottom: 0.25rem;
  }
  .result-class {
      font-size: 2rem; font-weight: 700; color: #0b0b0b;
      line-height: 1.2; margin-bottom: 0.25rem;
  }
  .result-confidence { font-size: 1.1rem; font-weight: 600; }

  /* Biểu đồ thanh ngang cho Top-K: nhãn trực tiếp trên từng thanh,
     thanh dẫn đầu dùng màu nhấn, các thanh còn lại xám (dạng "emphasis"). */
  .bar-row { margin-bottom: 0.7rem; }
  .bar-head {
      display: flex; justify-content: space-between;
      font-size: 0.9rem; margin-bottom: 0.25rem; color: #0b0b0b;
  }
  .bar-head .pct { font-variant-numeric: tabular-nums; color: #52514e; }
  .bar-track {
      height: 12px; background: #f0efec; border-radius: 6px; overflow: hidden;
  }
  .bar-fill { height: 100%; border-radius: 6px; }

  .empty-state {
      border: 2px dashed rgba(11,11,11,0.15);
      border-radius: 14px; padding: 2.5rem 1.5rem; text-align: center;
      color: #52514e;
  }
</style>
"""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# Thứ tự ưu tiên khi chọn model cho demo. Sau khi so sánh 2 model xong
# (reports/model_comparison.csv), đặt FINAL MODEL bạn đã chọn lên đầu danh sách.
CHECKPOINT_PRIORITY = ["mobilenetv2_transfer", "baseline_cnn"]


@st.cache_resource(show_spinner="Đang load model...")
def get_classifier(checkpoint_path: str) -> VegetableClassifier:
    return VegetableClassifier(checkpoint_path)


@st.cache_data
def load_report(name: str, suffix: str) -> dict | None:
    path = Path(CONFIG["paths"]["reports_dir"]) / f"{name}_{suffix}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_available_checkpoints() -> list[tuple[str, Path]]:
    """Trả về [(tên_model, đường_dẫn)] các checkpoint hiện có, theo thứ tự ưu tiên."""
    ckpt_dir = Path(CONFIG["paths"]["checkpoints_dir"])
    found = []
    for name in CHECKPOINT_PRIORITY:
        path = ckpt_dir / f"{name}.keras"
        if path.exists():
            found.append((name, path))
    return found


def render_topk_chart(top_k: list[dict]):
    """Biểu đồ thanh ngang cho Top-K predictions.

    Dùng dạng "emphasis": kết quả đứng đầu là thứ người dùng quan tâm nên được
    tô màu nhấn, các kết quả còn lại giữ màu xám làm ngữ cảnh so sánh. Mỗi thanh
    đều có nhãn trực tiếp (tên class + %), nên thông tin không phụ thuộc vào màu.
    """
    html = []
    for i, item in enumerate(top_k):
        pct = item["probability"] * 100
        color = ACCENT if i == 0 else MUTED_BAR
        html.append(
            f'<div class="bar-row">'
            f'  <div class="bar-head"><span>{item["class"]}</span>'
            f'  <span class="pct">{pct:.2f}%</span></div>'
            f'  <div class="bar-track">'
            f'    <div class="bar-fill" style="width:{max(pct, 0.6):.2f}%;background:{color};"></div>'
            f'  </div>'
            f'</div>'
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def confidence_color(confidence: float) -> str:
    if confidence >= 0.7:
        return GOOD
    if confidence >= 0.4:
        return WARNING
    return CRITICAL


# ---------------------------------------------------------------------------
# Tab 1 — Dự đoán
# ---------------------------------------------------------------------------
def tab_predict(classifier: VegetableClassifier, top_k: int):
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("1. Chọn ảnh")
        uploaded_file = st.file_uploader(
            "Upload ảnh rau/củ/quả",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Hỗ trợ JPG, PNG, BMP, WEBP. Ảnh nên rõ nét và chứa 1 loại rau/quả chính.",
        )

        image = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Preview — {uploaded_file.name}", use_container_width=True)
                st.caption(f"Kích thước gốc: {image.size[0]}×{image.size[1]} px · Định dạng: {image.format}")
            except Exception:  # noqa: BLE001
                st.error(
                    "❌ Không thể mở ảnh này. File có thể bị hỏng hoặc không phải "
                    "định dạng ảnh hợp lệ. Hãy thử một ảnh khác."
                )
                image = None
        else:
            st.markdown(
                '<div class="empty-state">📁<br><br>Chưa có ảnh nào được chọn.<br>'
                'Hãy upload một ảnh để bắt đầu.</div>',
                unsafe_allow_html=True,
            )

        predict_clicked = st.button(
            "🔍 Dự đoán", type="primary", use_container_width=True, disabled=image is None
        )

    with right:
        st.subheader("2. Kết quả")

        if not predict_clicked:
            st.markdown(
                '<div class="empty-state">Kết quả dự đoán sẽ hiển thị ở đây<br>'
                'sau khi bạn bấm nút <b>Dự đoán</b>.</div>',
                unsafe_allow_html=True,
            )
            return

        with st.spinner("Đang phân tích ảnh..."):
            try:
                result = classifier.predict(image, top_k=top_k)
            except InvalidImageError as e:
                st.error(f"❌ Ảnh không hợp lệ: {e}")
                return
            except Exception as e:  # noqa: BLE001
                st.error(f"❌ Có lỗi xảy ra khi dự đoán: {e}")
                return

        color = confidence_color(result.confidence)
        st.markdown(
            f'<div class="result-card">'
            f'  <div class="result-label">Kết quả dự đoán</div>'
            f'  <div class="result-class">{result.predicted_class}</div>'
            f'  <div class="result-confidence" style="color:{color};">'
            f'    Độ tin cậy: {result.confidence:.2%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if result.confidence < 0.4:
            st.warning(
                "⚠️ **Độ tin cậy thấp.** Ảnh có thể không thuộc các loại rau/quả model đã "
                "được học, hoặc chất lượng ảnh chưa tốt (mờ, thiếu sáng, có nhiều vật thể). "
                "Kết quả này không đáng tin cậy."
            )
        elif result.confidence < 0.7:
            st.info(
                "ℹ️ Độ tin cậy ở mức trung bình — nên tham khảo thêm các kết quả Top-K bên dưới."
            )

        st.markdown(f"##### Top-{len(result.top_k)} kết quả có khả năng nhất")
        render_topk_chart(result.top_k)

        # Lưu vào lịch sử phiên làm việc
        st.session_state.setdefault("history", [])
        st.session_state["history"].insert(0, {
            "class": result.predicted_class,
            "confidence": result.confidence,
        })
        st.session_state["history"] = st.session_state["history"][:10]

    history = st.session_state.get("history", [])
    if history:
        st.divider()
        st.markdown("##### 🕘 Lịch sử dự đoán trong phiên này")
        st.dataframe(
            [
                {"#": i + 1, "Loại rau/quả": h["class"], "Độ tin cậy": f"{h['confidence']:.2%}"}
                for i, h in enumerate(history)
            ],
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 2 — Thông tin Model
# ---------------------------------------------------------------------------
def tab_model_info(classifier: VegetableClassifier, model_name: str):
    info = classifier.model_info()
    meta = load_report(model_name, "metadata")

    st.subheader("Thông tin model đang sử dụng")

    c1, c2, c3 = st.columns(3)
    c1.metric("Số class", info["num_classes"])
    c2.metric("Input size", f"{info['input_size']}×{info['input_size']}")
    c3.metric("Số parameters", f"{info['num_parameters']:,}")

    if meta:
        st.markdown("##### Cấu hình huấn luyện")
        rows = [
            ("Architecture", meta.get("architecture", "—")),
            ("Optimizer", meta.get("optimizer", "—")),
            ("Learning rate", meta.get("learning_rate") or meta.get("learning_rate_stage1", "—")),
            ("Loss function", meta.get("loss_function", "—")),
            ("Batch size", meta.get("batch_size", "—")),
            ("Số epoch đã chạy", meta.get("epochs_ran") or meta.get("epochs_feature_extraction", "—")),
            ("Thời gian training", f"{meta.get('training_time_seconds', '—')} giây"),
            ("Áp dụng class_weight", "Có" if meta.get("class_weight_applied") else "Không"),
        ]
        st.dataframe(
            [{"Thông số": k, "Giá trị": str(v)} for k, v in rows],
            hide_index=True, use_container_width=True,
        )

        st.markdown("##### Kết quả train / validation")
        m1, m2 = st.columns(2)
        if isinstance(meta.get("final_train_accuracy"), (int, float)):
            m1.metric("Train accuracy", f"{meta['final_train_accuracy']:.2%}")
        if isinstance(meta.get("final_val_accuracy"), (int, float)):
            m2.metric("Validation accuracy", f"{meta['final_val_accuracy']:.2%}")
    else:
        st.info(
            "Chưa tìm thấy file metadata của model này trong `reports/`. "
            "Train lại model để sinh đầy đủ thông tin huấn luyện."
        )

    st.markdown("##### Danh sách các class model có thể nhận diện")
    st.dataframe(
        [{"#": i + 1, "Tên class": c} for i, c in enumerate(info["class_names"])],
        hide_index=True, use_container_width=True, height=280,
    )


# ---------------------------------------------------------------------------
# Tab 3 — Kết quả đánh giá
# ---------------------------------------------------------------------------
def tab_evaluation(model_name: str):
    st.subheader("Kết quả đánh giá trên tập test độc lập")

    summary = load_report(model_name, "summary")
    figures_dir = Path(CONFIG["paths"]["figures_dir"])

    if summary is None:
        st.info(
            "Chưa có kết quả đánh giá cho model này.\n\n"
            "Chạy lệnh sau để sinh đầy đủ metrics và biểu đồ:\n\n"
            "```bash\npython -m src.evaluation.evaluate --compare\n```"
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{summary['test_accuracy']:.2%}")
        c2.metric("Precision (macro)", f"{summary['test_precision_macro']:.3f}")
        c3.metric("Recall (macro)", f"{summary['test_recall_macro']:.3f}")
        c4.metric("F1-score (macro)", f"{summary['test_f1_macro']:.3f}")

        st.caption(
            f"Đánh giá trên {summary.get('num_test_samples', '—')} ảnh test · "
            f"{summary.get('num_misclassified', '—')} ảnh bị dự đoán sai · "
            f"Class tốt nhất: **{summary.get('best_class', '—')}** · "
            f"Class yếu nhất: **{summary.get('worst_class', '—')}**"
        )

        overfit = summary.get("overfitting_analysis")
        if overfit:
            status = overfit.get("status")
            message = overfit.get("explanation", "")
            if status == "good_fit":
                st.success(f"✅ {message}")
            else:
                st.warning(f"⚠️ {message}")

        confused = summary.get("top_confused_pairs") or []
        if confused:
            st.markdown("##### Các cặp class bị nhầm lẫn nhiều nhất")
            st.dataframe(
                [
                    {
                        "Nhãn thật": p["true_class"],
                        "Model dự đoán thành": p["predicted_class"],
                        "Số lần": p["count"],
                    }
                    for p in confused
                ],
                hide_index=True, use_container_width=True,
            )

    st.divider()
    st.markdown("##### Biểu đồ")

    figures = [
        (f"{model_name}_training_curves.png", "Training vs Validation (Accuracy & Loss)"),
        (f"{model_name}_confusion_matrix.png", "Confusion Matrix"),
        (f"{model_name}_misclassified.png", "Các ảnh bị dự đoán sai"),
        (f"{model_name}_correct_predictions.png", "Các ảnh dự đoán đúng"),
    ]

    any_figure = False
    for filename, caption in figures:
        path = figures_dir / filename
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
            any_figure = True

    if not any_figure:
        st.info(
            "Chưa có biểu đồ nào. Chạy `python -m src.training.train --model both` "
            "rồi `python -m src.evaluation.evaluate --compare` để sinh biểu đồ."
        )


# ---------------------------------------------------------------------------
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="app-header">'
        '<h1>🥕 Hệ thống Phân loại Rau Củ Quả</h1>'
        '<p>Đồ án học phần Học máy — Nhận diện loại rau/củ/quả từ ảnh bằng Deep Learning</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    available = find_available_checkpoints()

    if not available:
        st.error("⚠️ Chưa tìm thấy model đã train nào trong thư mục `checkpoints/`.")
        st.markdown(
            "**Cách khắc phục** — chạy lần lượt các lệnh sau ở thư mục gốc project:\n\n"
            "```bash\n"
            "python scripts/download_dataset.py       # tải dataset\n"
            "python -m src.training.train --model both  # train model\n"
            "```\n\n"
            "Xem hướng dẫn chi tiết trong `TRAINING_GUIDE.md`."
        )
        st.stop()

    with st.sidebar:
        st.markdown("### ⚙️ Cấu hình")

        display_names = {
            "mobilenetv2_transfer": "MobileNetV2 (Transfer Learning)",
            "baseline_cnn": "CNN Baseline",
        }
        options = [name for name, _ in available]
        selected = st.selectbox(
            "Model sử dụng",
            options,
            format_func=lambda n: display_names.get(n, n),
            help="Mặc định là final model đã chọn. Có thể đổi để so sánh trực tiếp 2 model.",
        )
        checkpoint_path = dict(available)[selected]

        classifier = None
        try:
            classifier = get_classifier(str(checkpoint_path))
        except Exception as e:  # noqa: BLE001
            st.error(f"Không load được model: {e}")

        if classifier is not None:
            max_k = min(10, classifier.model_info()["num_classes"])
            top_k = st.slider("Số kết quả Top-K", 1, max_k, min(5, max_k))
        else:
            top_k = 5

        st.divider()
        st.caption(
            "Ứng dụng sử dụng đúng model đã được đánh giá trong `src/evaluation` — "
            "số liệu ở tab *Kết quả đánh giá* được đọc trực tiếp từ `reports/`."
        )

    if classifier is None:
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🔍 Dự đoán", "🧠 Thông tin Model", "📊 Kết quả đánh giá"])
    with tab1:
        tab_predict(classifier, top_k)
    with tab2:
        tab_model_info(classifier, selected)
    with tab3:
        tab_evaluation(selected)


if __name__ == "__main__":
    main()

"""Kiểm thử bộ đặc trưng v2 (src/features/svm_features.py) và dự đoán SVM v2 (svm_predict.py)."""

from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from PIL import Image

from src.features import svm_features as sf

RNG = np.random.default_rng(0)


def _img(color, size=(128, 128)):
    """Ảnh nền trắng có một hình tròn màu `color`, kèm nhiễu nhẹ."""
    import cv2

    img = np.full((size[1], size[0], 3), 255, np.uint8)
    radius = int(35 * min(size) / 128)   # giữ tỉ lệ vật thể / khung ảnh như nhau ở mọi kích thước
    cv2.circle(img, (size[0] // 2 + int(RNG.integers(-10, 10)), size[1] // 2), radius, color, -1)
    return np.clip(img + RNG.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


def test_block_dims_match_extracted_vector():
    dims = sf.block_dims()
    assert dims == {"hog16": 1764, "hsv": 64, "hsv_grid": 160, "lbp": 28}
    vec = sf.extract_vector(_img((200, 30, 30)))
    assert vec.shape == (sum(dims.values()),)
    assert np.isfinite(vec).all()


def test_histograms_are_normalized_proportions():
    blocks = sf.extract_blocks(_img((40, 160, 40)))
    assert blocks["hsv"].min() >= 0
    np.testing.assert_allclose(blocks["hsv"].sum(), 3.0, atol=1e-5)        # 3 kênh, mỗi kênh tổng = 1
    np.testing.assert_allclose(blocks["hsv_grid"].sum(), 15.0, atol=1e-4)  # 5 vùng x 3 kênh
    np.testing.assert_allclose(blocks["lbp"].sum(), 2.0, atol=1e-5)        # 2 cấu hình LBP


def test_extract_blocks_rejects_non_rgb():
    with pytest.raises(ValueError):
        sf.extract_blocks(np.zeros((128, 128), np.uint8))


def test_hue_feature_separates_colors():
    red, green = sf.extract_blocks(_img((220, 20, 20))), sf.extract_blocks(_img((20, 200, 20)))
    assert np.argmax(red["hsv"][:32]) != np.argmax(green["hsv"][:32])


def _toy_dataset(n_per_class=12):
    colors = [(220, 20, 20), (20, 180, 20), (230, 210, 40)]
    X, y = [], []
    for label, c in enumerate(colors):
        for _ in range(n_per_class):
            X.append(sf.extract_vector(_img(c)))
            y.append(label)
    return np.array(X), np.array(y)


def test_pipeline_fits_and_predicts_probabilities():
    X, y = _toy_dataset()
    pipe = sf.build_pipeline(sf.DEFAULT_BLOCKS, sf.block_dims(), hog_pca=10)
    pipe.set_params(svc__probability=True)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X[:3])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (pipe.predict(X) == y).mean() > 0.9


def test_svm_predictor_with_v2_bundle(tmp_path):
    from src.inference.predict import InvalidImageError
    from src.inference.svm_predict import SVMVegetableClassifier
    from src.training.train_svm import train_v2

    X, y = _toy_dataset()
    args = SimpleNamespace(v2_blocks=sf.DEFAULT_BLOCKS, v2_hog_pca=10, seed=42, c=[1, 10], cv=3, n_jobs=1)
    bundle, best_params, *_ = train_v2(X, y, args, X_flip=X)
    bundle.update({"class_names": ["apple", "lime", "lemon"], "feature_params": sf.feature_params()})
    path = tmp_path / "classical_svm.pkl"
    joblib.dump(bundle, path)

    clf = SVMVegetableClassifier(str(path))
    result = clf.predict(Image.fromarray(_img((220, 20, 20), size=(256, 256))), top_k=3)
    assert result.predicted_class == "apple"
    assert len(result.top_k) == 3 and 0 <= result.confidence <= 1

    info = clf.model_info()
    assert info["kind"] == "svm" and info["pca_components"] == 10
    assert info["best_params"]["C"] == best_params["C"]

    import io
    with pytest.raises(InvalidImageError):
        clf.predict(io.BytesIO(b"khong phai anh"))


def test_svm_predictor_rejects_mismatched_feature_params(tmp_path):
    from src.inference.svm_predict import SVMVegetableClassifier

    X, y = _toy_dataset(6)
    pipe = sf.build_pipeline(sf.DEFAULT_BLOCKS, sf.block_dims(), hog_pca=5).fit(X, y)
    params = sf.feature_params()
    params["hsv_bins"] = [8, 8, 8]   # giả lập mô hình train với tham số khác code hiện tại
    bundle = {"backend": "v2", "pipeline": pipe, "blocks": sf.DEFAULT_BLOCKS, "class_names": ["a", "b", "c"],
              "feature_params": params}
    path = tmp_path / "classical_svm.pkl"
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="tham số đặc trưng"):
        SVMVegetableClassifier(str(path))

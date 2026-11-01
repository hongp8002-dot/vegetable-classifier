"""
src/features/svm_features.py
----------------------------
Bộ đặc trưng v2 cho SVM – kết quả của thí nghiệm scripts/svm_experiments.py (cấu hình E5).

So với bộ v1 (hog_features.py: HOG cell 8 + histogram RGB):
    - HOG thô hơn (cell 16x16): 1.764 chiều thay vì 8.100, ít nhiễu nền hơn, chất lượng tương đương.
    - Histogram HSV thay cho RGB: kênh Hue (sắc độ) ít phụ thuộc ánh sáng hơn -> đóng góp lớn nhất.
    - Histogram HSV theo vùng (4 góc phần tư + vùng giữa): biết màu nằm ở đâu, giảm ảnh hưởng của nền.
    - LBP (Local Binary Pattern): đặc trưng kết cấu bề mặt (vỏ sần, vân, hạt).

Mọi khối nhận cùng một ảnh RGB uint8 đã resize về CONFIG["svm"]["img_size"] (128x128) bằng
hog_features.load_and_resize, nên huấn luyện và dự đoán dùng đúng một cách tiền xử lý.

Pipeline mô hình (build_pipeline):
    HOG   -> PCA(50) -> chuẩn hóa L2  ┐
    HSV   -> Hellinger -> chuẩn hóa L2 ├-> ghép -> SVC RBF
    vùng  -> Hellinger -> chuẩn hóa L2 │
    LBP   -> Hellinger -> chuẩn hóa L2 ┘
Chuẩn hóa L2 theo từng khối giúp các khối có vai trò ngang nhau mà không cần StandardScaler
(StandardScaler làm mất thông tin màu – xem reports/svm_tuning_log.csv).
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from configs.config import CONFIG

_SVM_CFG = CONFIG["svm"]
IMG_SIZE = tuple(_SVM_CFG["img_size"])
DEFAULT_BLOCKS = list(_SVM_CFG.get("v2_blocks", ["hog16", "hsv", "hsv_grid", "lbp"]))
HOG_ORIENTATIONS = 9
HSV_BINS = (32, 16, 16)          # H, S, V cho histogram toàn ảnh
HSV_GRID_BINS = (16, 8, 8)       # H, S, V cho mỗi vùng
LBP_SETTINGS = ((8, 1), (16, 2))  # (số điểm lân cận P, bán kính R), dạng "uniform"


# ---------------------------------------------------------------------------
# Các khối đặc trưng
# ---------------------------------------------------------------------------
def _hog(gray: np.ndarray, cell: int) -> np.ndarray:
    return hog(gray, orientations=HOG_ORIENTATIONS, pixels_per_cell=(cell, cell),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def _hist(channel: np.ndarray, bins: int, value_range: list) -> np.ndarray:
    h = cv2.calcHist([np.ascontiguousarray(channel)], [0], None, [bins], value_range).ravel()
    return (h / max(h.sum(), 1.0)).astype(np.float32)


def _hsv_hist(hsv: np.ndarray, bins=HSV_BINS) -> np.ndarray:
    # OpenCV: H trong [0, 180), S và V trong [0, 256)
    return np.concatenate([_hist(hsv[..., 0], bins[0], [0, 180]),
                           _hist(hsv[..., 1], bins[1], [0, 256]),
                           _hist(hsv[..., 2], bins[2], [0, 256])])


def _regions(img: np.ndarray) -> list[np.ndarray]:
    """4 góc phần tư + vùng giữa (1/2 cạnh), thường là nơi có vật thể chính."""
    h, w = img.shape[:2]
    return [img[:h // 2, :w // 2], img[:h // 2, w // 2:], img[h // 2:, :w // 2], img[h // 2:, w // 2:],
            img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]]


def _lbp(gray: np.ndarray) -> np.ndarray:
    out = []
    for p, r in LBP_SETTINGS:
        codes = local_binary_pattern(gray, p, r, method="uniform")
        h, _ = np.histogram(codes, bins=p + 2, range=(0, p + 2))
        out.append(h / max(h.sum(), 1))
    return np.concatenate(out).astype(np.float32)


def extract_blocks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Ảnh RGB uint8 (đã resize) -> dict {tên khối: vector}. Có cả 2 kiểu HOG để làm thí nghiệm."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Cần ảnh RGB 3 kênh, nhận được shape {rgb.shape}")
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {
        "hog8": _hog(gray, 8),
        "hog16": _hog(gray, 16),
        "rgb": np.concatenate([_hist(rgb[..., c], 32, [0, 256]) for c in range(3)]),
        "hsv": _hsv_hist(hsv),
        "hsv_grid": np.concatenate([_hsv_hist(r, HSV_GRID_BINS) for r in _regions(hsv)]),
        "lbp": _lbp(gray),
    }


def extract_vector(rgb: np.ndarray, blocks: list[str] | None = None) -> np.ndarray:
    """Ghép các khối theo đúng thứ tự `blocks` thành 1 vector (dùng khi dự đoán 1 ảnh)."""
    feats = extract_blocks(rgb)
    return np.concatenate([feats[b] for b in (blocks or DEFAULT_BLOCKS)])


def block_dims(blocks: list[str] | None = None) -> dict[str, int]:
    dummy = np.zeros((*IMG_SIZE[::-1], 3), dtype=np.uint8)
    feats = extract_blocks(dummy)
    return {b: int(feats[b].shape[0]) for b in (blocks or DEFAULT_BLOCKS)}


def feature_params(blocks: list[str] | None = None) -> dict:
    """Tham số đặc trưng lưu kèm mô hình; svm_predict so sánh để tránh lệch train/dự đoán."""
    return {
        "feature_set": "v2",
        "img_size": list(IMG_SIZE),
        "blocks": list(blocks or DEFAULT_BLOCKS),
        "hog_orientations": HOG_ORIENTATIONS,
        "hsv_bins": list(HSV_BINS),
        "hsv_grid_bins": list(HSV_GRID_BINS),
        "lbp_settings": [list(s) for s in LBP_SETTINGS],
    }


# ---------------------------------------------------------------------------
# Trích đặc trưng hàng loạt (song song), kèm bản lật ngang để tăng dữ liệu
# ---------------------------------------------------------------------------
def _extract_path(path: str, blocks: list[str], with_flip: bool):
    from src.features.hog_features import load_and_resize
    try:
        rgb = load_and_resize(path, IMG_SIZE)
    except Exception:  # noqa: BLE001 – ảnh lỗi được bỏ qua và báo lại
        return None
    x = extract_vector(rgb, blocks)
    xf = extract_vector(np.ascontiguousarray(rgb[:, ::-1]), blocks) if with_flip else None
    return x, xf


def batch_extract(paths: list[str], blocks: list[str] | None = None, with_flip: bool = False,
                  n_jobs: int = -1):
    """Trả về (X, X_flip hoặc None, keep_mask)."""
    from joblib import Parallel, delayed

    blocks = blocks or DEFAULT_BLOCKS
    res = Parallel(n_jobs=n_jobs)(delayed(_extract_path)(p, blocks, with_flip) for p in paths)
    keep = np.array([r is not None for r in res], dtype=bool)
    ok = [r for r in res if r is not None]
    if not ok:
        raise ValueError("Không trích được đặc trưng từ ảnh nào.")
    X = np.stack([r[0] for r in ok]).astype(np.float32)
    X_flip = np.stack([r[1] for r in ok]).astype(np.float32) if with_flip else None
    return X, X_flip, keep


# ---------------------------------------------------------------------------
# Pipeline mô hình
# ---------------------------------------------------------------------------
def hellinger(x):
    """Biến đổi Hellinger: căn bậc hai tỉ lệ pixel -> bin nhỏ không bị bin lớn (màu nền) lấn át.
    Đặt ở mức module để pickle được (Streamlit nạp lại mô hình)."""
    return np.sqrt(np.clip(x, 0, None))


def build_pipeline(blocks: list[str], dims: dict[str, int], hog_pca: int = 50, seed: int = 42):
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, Normalizer
    from sklearn.svm import SVC

    parts, start = [], 0
    for b in blocks:
        cols = list(range(start, start + dims[b]))
        start += dims[b]
        if b.startswith("hog"):
            step = Pipeline([("pca", PCA(hog_pca, random_state=seed)), ("l2", Normalizer())])
        else:
            step = Pipeline([("hellinger", FunctionTransformer(hellinger)), ("l2", Normalizer())])
        parts.append((b, step, cols))
    return Pipeline([("features", ColumnTransformer(parts)),
                     ("svc", SVC(kernel="rbf", gamma="scale", random_state=seed))])


def model_dims(blocks: list[str], dims: dict[str, int], hog_pca: int) -> int:
    """Số chiều thực sự đưa vào SVC (HOG sau PCA + các histogram)."""
    return sum(hog_pca if b.startswith("hog") else dims[b] for b in blocks)

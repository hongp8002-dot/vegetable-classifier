"""
src/features/hog_features.py
----------------------------
NGƯỜI LÀM: TN 2

Mục đích:
    Rút trích đặc trưng từ ảnh bằng 2 phương pháp:
    - HOG (Histogram of Oriented Gradients): phát hiện đường viền và hình dạng trong ảnh
    - Color Histogram: phân bố màu sắc của ảnh

    Đặc trưng = thông tin dạng số mà SVM có thể học được.
    Ví dụ: ảnh cà rốt có màu cam và hình dài → vector số biểu diễn điều đó.

Cách dùng (sau khi làm xong):
    from src.features.hog_features import extract_combined_features
    features = extract_combined_features("path/to/carrot.jpg")
    # features là 1 mảng numpy, ví dụ có 2916 số
"""

import numpy as np
import cv2
from skimage.feature import hog
from pathlib import Path


# ============================================================
# CÁC THAM SỐ MẶC ĐỊNH — có thể thay đổi để thử nghiệm
# ============================================================
DEFAULT_IMG_SIZE = (128, 128)   # Resize ảnh về kích thước này trước khi xử lý
HOG_ORIENTATIONS = 9            # Số hướng gradient (thường dùng 8 hoặc 9)
HOG_PIXELS_PER_CELL = (8, 8)   # Kích thước mỗi ô nhỏ trong HOG
HOG_CELLS_PER_BLOCK = (2, 2)   # Số ô ghép lại thành 1 block để normalize
COLOR_HIST_BINS = 32            # Số bin cho histogram màu (mỗi kênh RGB)


def load_and_resize(image_path: str, img_size: tuple = DEFAULT_IMG_SIZE) -> np.ndarray:
    """
    Đọc ảnh từ đường dẫn và resize về kích thước cố định.

    Tại sao cần resize?
        SVM cần vector đầu vào có cùng số chiều.
        Nếu ảnh 200x300 và ảnh 100x100 → vector khác chiều → SVM không chạy được.

    Args:
        image_path: Đường dẫn đến file ảnh (jpg, png, ...)
        img_size: Kích thước mục tiêu, mặc định (128, 128)

    Returns:
        Ảnh numpy array shape (128, 128, 3), giá trị 0-255
    """
    # TODO: Đọc ảnh bằng OpenCV
    # Gợi ý: img = cv2.imread(str(image_path))
    # Lưu ý: OpenCV đọc ảnh theo BGR, cần convert sang RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, img_size)
    raise NotImplementedError("TN 2 cần implement hàm này")


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Tính HOG (Histogram of Oriented Gradients) từ ảnh.

    HOG là gì?
        HOG chia ảnh thành các ô nhỏ, trong mỗi ô tính xem có
        bao nhiêu đường thẳng theo từng hướng (0°, 45°, 90°, ...).
        Ảnh cà rốt sẽ có nhiều đường dọc (thân dài), ảnh cam sẽ tròn hơn.

    Args:
        image: Ảnh numpy array shape (H, W, 3), giá trị 0-255

    Returns:
        Vector 1D numpy array (ví dụ 2916 số với ảnh 128x128 và params mặc định)
    """
    # TODO: Convert ảnh sang grayscale rồi tính HOG
    # Gợi ý:
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # features = hog(
    #     gray,
    #     orientations=HOG_ORIENTATIONS,
    #     pixels_per_cell=HOG_PIXELS_PER_CELL,
    #     cells_per_block=HOG_CELLS_PER_BLOCK,
    #     block_norm='L2-Hys'
    # )
    raise NotImplementedError("TN 2 cần implement hàm này")


def extract_color_histogram(image: np.ndarray, bins: int = COLOR_HIST_BINS) -> np.ndarray:
    """
    Tính histogram màu sắc từ ảnh.

    Histogram màu là gì?
        Đếm xem có bao nhiêu pixel có màu đỏ nhạt, đỏ đậm, xanh lá nhạt, ...
        Ảnh ớt đỏ sẽ có nhiều pixel màu đỏ, ảnh rau cải sẽ có nhiều pixel xanh lá.

    Args:
        image: Ảnh numpy array shape (H, W, 3)
        bins: Số mức màu cho mỗi kênh (32 bins × 3 kênh = 96 số)

    Returns:
        Vector 1D numpy array (bins * 3 số), đã normalize về [0, 1]
    """
    # TODO: Tính histogram cho từng kênh R, G, B rồi ghép lại
    # Gợi ý:
    # hist_features = []
    # for channel in range(3):  # R, G, B
    #     hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
    #     hist = cv2.normalize(hist, hist).flatten()
    #     hist_features.extend(hist)
    # return np.array(hist_features)
    raise NotImplementedError("TN 2 cần implement hàm này")


def extract_combined_features(image_path: str, img_size: tuple = DEFAULT_IMG_SIZE) -> np.ndarray:
    """
    Hàm CHÍNH — gộp HOG + Color Histogram thành 1 vector đặc trưng.

    Đây là hàm TN 1 và TN 3 sẽ gọi, không cần biết chi tiết bên trong.

    Args:
        image_path: Đường dẫn đến file ảnh
        img_size: Kích thước ảnh sau resize

    Returns:
        Vector 1D numpy array gộp HOG + color hist
        Ví dụ: HOG (2916 số) + Color hist (96 số) = 3012 số tổng cộng
    """
    # TODO: Gọi các hàm phía trên và ghép kết quả
    # Gợi ý:
    # image = load_and_resize(image_path, img_size)
    # hog_feat = extract_hog_features(image)
    # color_feat = extract_color_histogram(image)
    # return np.concatenate([hog_feat, color_feat])
    raise NotImplementedError("TN 2 cần implement hàm này")


def batch_extract_features(image_paths: list, img_size: tuple = DEFAULT_IMG_SIZE) -> np.ndarray:
    """
    Rút trích đặc trưng cho NHIỀU ảnh cùng lúc (dùng khi xử lý toàn bộ dataset).

    Args:
        image_paths: List các đường dẫn ảnh
        img_size: Kích thước ảnh sau resize

    Returns:
        Ma trận 2D numpy array shape (n_images, n_features)
        Ví dụ: 1000 ảnh → ma trận (1000, 3012)
    """
    # TODO: Dùng vòng lặp hoặc list comprehension để xử lý từng ảnh
    # Gợi ý:
    # features = []
    # for i, path in enumerate(image_paths):
    #     feat = extract_combined_features(path, img_size)
    #     features.append(feat)
    #     if (i + 1) % 100 == 0:
    #         print(f"Đã xử lý {i+1}/{len(image_paths)} ảnh")
    # return np.array(features)
    raise NotImplementedError("TN 2 cần implement hàm này")


# ============================================================
# TEST NHANH — chạy file này trực tiếp để kiểm tra
# python src/features/hog_features.py
# ============================================================
if __name__ == "__main__":
    import sys
    print("=== Test HOG feature extractor ===")
    print("Chạy test với 1 ảnh mẫu...")
    print()
    print("TODO: Sau khi implement xong, chạy:")
    print("  features = extract_combined_features('path/to/any/image.jpg')")
    print("  print(f'Số chiều đặc trưng: {features.shape}')  # Phải ra (3012,) hoặc tương tự")
    print()
    print("Báo cho TN 1 biết số chiều để config SVM đúng.")

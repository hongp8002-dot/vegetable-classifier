"""
src/features/pca_wrapper.py
----------------------------
NGƯỜI LÀM: TN 2

Mục đích:
    Giảm số chiều của vector đặc trưng trước khi đưa vào SVM.

    Vấn đề: HOG + Color Histogram tạo ra ~3000 số cho mỗi ảnh.
    SVM train chậm hơn khi vector quá dài. PCA giúp giảm từ 3000 xuống còn ~200
    mà vẫn giữ được ~95% thông tin quan trọng.

    Nghĩ đơn giản: PCA tìm cách "nén" dữ liệu mà không mất nhiều thông tin.

Cách dùng (sau khi làm xong):
    from src.features.pca_wrapper import PCAWrapper

    pca = PCAWrapper(n_components=200)
    X_train_reduced = pca.fit_transform(X_train)  # Train PCA trên train set
    X_test_reduced = pca.transform(X_test)        # Chỉ transform, không fit lại!
    pca.save("checkpoints/pca_model.pkl")         # Lưu để dùng khi inference

    # Sau đó load lại:
    pca2 = PCAWrapper.load("checkpoints/pca_model.pkl")
"""

import numpy as np
import joblib
from pathlib import Path
# from sklearn.decomposition import PCA  # Uncomment khi dùng


class PCAWrapper:
    """
    Bọc sklearn.decomposition.PCA với các tính năng thêm:
    - Tự ghi lại % phương sai giữ lại
    - Có hàm save/load model
    - Có thể vẽ biểu đồ explained variance (dùng trong EDA)
    """

    def __init__(self, n_components: int = 200):
        """
        Args:
            n_components: Số chiều sau khi giảm.
                          Nếu đặt None, PCA sẽ tự chọn để giữ 95% phương sai.
                          Thường dùng 100-300 là đủ tốt.
        """
        self.n_components = n_components
        self.pca = None          # Sẽ là sklearn PCA object sau khi fit
        self.is_fitted = False   # Cờ kiểm tra đã fit chưa

    def fit(self, X: np.ndarray) -> "PCAWrapper":
        """
        Học PCA từ dữ liệu train.

        QUAN TRỌNG: Chỉ gọi fit() trên train set!
        Không được fit trên validation hoặc test set → gây data leakage.

        Args:
            X: Ma trận đặc trưng shape (n_samples, n_features)
               Ví dụ: (5000, 3012) — 5000 ảnh, mỗi ảnh 3012 đặc trưng

        Returns:
            self (để có thể chain: pca.fit(X).transform(X))
        """
        # TODO:
        # from sklearn.decomposition import PCA
        # self.pca = PCA(n_components=self.n_components, random_state=42)
        # self.pca.fit(X)
        # self.is_fitted = True
        # print(f"PCA fit xong: {X.shape[1]} chiều → {self.n_components} chiều")
        # variance_kept = sum(self.pca.explained_variance_ratio_) * 100
        # print(f"Phương sai giữ lại: {variance_kept:.1f}%")
        # return self
        raise NotImplementedError("TN 2 cần implement hàm này")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Áp dụng PCA đã học để giảm chiều dữ liệu mới.

        Args:
            X: Ma trận shape (n_samples, n_features_original)

        Returns:
            Ma trận shape (n_samples, n_components) — ít cột hơn
        """
        # TODO:
        # if not self.is_fitted:
        #     raise RuntimeError("Cần gọi fit() trước khi transform()")
        # return self.pca.transform(X)
        raise NotImplementedError("TN 2 cần implement hàm này")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Gọi fit() rồi transform() trong 1 bước (tiện hơn).
        Chỉ dùng cho train set!

        Args:
            X: Ma trận shape (n_samples, n_features)

        Returns:
            Ma trận shape (n_samples, n_components)
        """
        # TODO:
        # return self.fit(X).transform(X)
        raise NotImplementedError("TN 2 cần implement hàm này")

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Trả về % phương sai mà mỗi component giải thích.
        Dùng để vẽ biểu đồ trong EDA notebook.

        Returns:
            Array shape (n_components,), ví dụ [0.15, 0.08, 0.05, ...]
            Nghĩa là: component đầu tiên giữ 15% thông tin, thứ 2 giữ 8%, ...
        """
        # TODO:
        # if not self.is_fitted:
        #     raise RuntimeError("Cần gọi fit() trước")
        # return self.pca.explained_variance_ratio_
        raise NotImplementedError("TN 2 cần implement hàm này")

    def save(self, path: str) -> None:
        """
        Lưu PCA model ra file để dùng lại sau (inference, so sánh).

        Args:
            path: Đường dẫn lưu file, ví dụ "checkpoints/pca_model.pkl"
        """
        # TODO:
        # if not self.is_fitted:
        #     raise RuntimeError("Cần gọi fit() trước khi save()")
        # Path(path).parent.mkdir(parents=True, exist_ok=True)
        # joblib.dump(self, path)
        # print(f"Đã lưu PCA model tại: {path}")
        raise NotImplementedError("TN 2 cần implement hàm này")

    @classmethod
    def load(cls, path: str) -> "PCAWrapper":
        """
        Load PCA model đã lưu từ file.

        Args:
            path: Đường dẫn file .pkl đã lưu

        Returns:
            PCAWrapper object đã fit sẵn, dùng được ngay
        """
        # TODO:
        # model = joblib.load(path)
        # print(f"Đã load PCA model từ: {path}")
        # return model
        raise NotImplementedError("TN 2 cần implement hàm này")

    def plot_explained_variance(self, save_path: str = None):
        """
        Vẽ biểu đồ % phương sai tích lũy theo số component.
        Dùng trong EDA notebook để chọn n_components phù hợp.

        Ví dụ biểu đồ cho thấy: 200 component giữ 95% phương sai
        → chọn n_components=200 là hợp lý.

        Args:
            save_path: Nếu truyền vào, lưu ảnh ra file đó (dùng trong báo cáo)
        """
        # TODO:
        # import matplotlib.pyplot as plt
        # cumsum = np.cumsum(self.get_explained_variance_ratio()) * 100
        # plt.figure(figsize=(8, 5))
        # plt.plot(cumsum)
        # plt.xlabel('Số PCA components')
        # plt.ylabel('% Phương sai tích lũy')
        # plt.title('PCA: Số component cần thiết để giữ bao nhiêu thông tin')
        # plt.axhline(y=95, color='r', linestyle='--', label='95% threshold')
        # plt.legend()
        # plt.grid(True)
        # if save_path:
        #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.show()
        raise NotImplementedError("TN 2 cần implement hàm này")


# ============================================================
# TEST NHANH
# python src/features/pca_wrapper.py
# ============================================================
if __name__ == "__main__":
    print("=== Test PCA Wrapper ===")
    print("Tạo dữ liệu giả để test...")
    print()
    print("TODO: Sau khi implement xong, uncomment các dòng dưới:")
    print()
    print("  # Tạo data giả 100 samples, 3012 features")
    print("  X_fake = np.random.randn(100, 3012)")
    print("  pca = PCAWrapper(n_components=50)")
    print("  X_reduced = pca.fit_transform(X_fake)")
    print("  print(X_reduced.shape)  # Phải ra (100, 50)")
    print("  pca.plot_explained_variance()")
    print()
    print("Nếu in ra đúng (100, 50) thì hàm hoạt động đúng.")

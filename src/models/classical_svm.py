"""
src/models/classical_svm.py
----------------------------
NGƯỜI LÀM: TN 1 (bạn)

Mục đích:
    Model chính của môn học — dùng SVM để phân loại ảnh rau củ quả.

    Pipeline hoàn chỉnh:
        Ảnh → HOG + Color Histogram (TN 2 làm) → PCA giảm chiều (TN 2 làm)
            → SVM phân loại (file này)

    SVM là gì?
        SVM (Support Vector Machine) tìm đường biên (hyperplane) để chia
        dữ liệu thành các nhóm. Với kernel RBF, SVM có thể tạo ra đường biên
        phi tuyến tính, phù hợp với dữ liệu ảnh phức tạp.

    GridSearchCV là gì?
        Tự động thử nhiều tổ hợp C và gamma để tìm ra tổ hợp tốt nhất.
        Ví dụ thử: C=[0.1, 1, 10, 100] × gamma=['scale', 'auto'] = 8 tổ hợp.
        Mỗi tổ hợp dùng cross-validation để đánh giá → tốn thời gian nhưng đáng.

Phụ thuộc (cần TN 2 làm xong trước):
    from src.features.hog_features import batch_extract_features
    from src.features.pca_wrapper import PCAWrapper

Cách dùng (sau khi làm xong):
    model = ClassicalSVMClassifier()
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    model.save("checkpoints/svm_model.pkl")
"""

import numpy as np
import joblib
from pathlib import Path
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import LabelEncoder


class ClassicalSVMClassifier:
    """
    SVM classifier với pipeline đầy đủ: fit → predict → save/load.
    """

    def __init__(
        self,
        c_values: list = None,
        gamma_values: list = None,
        cv_folds: int = 5,
    ):
        """
        Args:
            c_values: Danh sách giá trị C để thử (C điều chỉnh độ rộng margin)
                      C nhỏ → margin rộng, có thể sai nhiều hơn
                      C lớn → margin hẹp, ôm sát dữ liệu hơn nhưng dễ overfit
            gamma_values: Danh sách giá trị gamma để thử
                          'scale' và 'auto' là 2 cách tính tự động của sklearn
            cv_folds: Số fold cross-validation (thường dùng 5)
        """
        # Lấy giá trị từ config nếu không truyền vào
        if c_values is None:
            try:
                from configs.config import SVM_C_LIST
                c_values = SVM_C_LIST
            except ImportError:
                c_values = [0.1, 1, 10, 100]

        if gamma_values is None:
            try:
                from configs.config import SVM_GAMMA_LIST
                gamma_values = SVM_GAMMA_LIST
            except ImportError:
                gamma_values = ['scale', 'auto']

        self.c_values = c_values
        self.gamma_values = gamma_values
        self.cv_folds = cv_folds

        self.model = None         # SVC object sau khi train
        self.best_params = None   # C và gamma tốt nhất tìm được
        self.classes = None       # Tên các class (36 loại rau củ)
        self.is_fitted = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> dict:
        """
        Train SVM với GridSearchCV để tìm hyperparameter tốt nhất.

        Args:
            X_train: Ma trận đặc trưng đã qua PCA, shape (n_samples, n_components)
                     Ví dụ: (5000, 200) — 5000 ảnh, mỗi ảnh 200 đặc trưng sau PCA
            y_train: Nhãn dạng số, shape (n_samples,)
                     Ví dụ: [0, 0, 1, 2, 2, ...] — 0=apple, 1=banana, ...
            verbose: In thông tin quá trình train

        Returns:
            dict chứa best_params, best_score và thời gian train
        """
        # TODO:
        # import time
        # from sklearn.svm import SVC
        # from sklearn.model_selection import GridSearchCV
        #
        # if verbose:
        #     print(f"Bắt đầu GridSearchCV với {len(self.c_values) * len(self.gamma_values)} tổ hợp...")
        #     print(f"Mỗi tổ hợp dùng {self.cv_folds}-fold CV → tổng {len(self.c_values) * len(self.gamma_values) * self.cv_folds} lần train")
        #     print("(Có thể mất 30 phút đến vài giờ tùy kích thước dataset)")
        #
        # param_grid = {'C': self.c_values, 'gamma': self.gamma_values}
        # svc = SVC(kernel='rbf', probability=True, random_state=42)  # probability=True để lấy confidence score
        # grid_search = GridSearchCV(svc, param_grid, cv=self.cv_folds, scoring='f1_macro', n_jobs=-1, verbose=2)
        #
        # start = time.time()
        # grid_search.fit(X_train, y_train)
        # elapsed = time.time() - start
        #
        # self.model = grid_search.best_estimator_
        # self.best_params = grid_search.best_params_
        # self.is_fitted = True
        #
        # if verbose:
        #     print(f"\nTrain xong! Thời gian: {elapsed/60:.1f} phút")
        #     print(f"Hyperparameter tốt nhất: C={self.best_params['C']}, gamma={self.best_params['gamma']}")
        #     print(f"F1-score trên validation (CV): {grid_search.best_score_:.4f}")
        #
        # return {'best_params': self.best_params, 'best_cv_score': grid_search.best_score_, 'train_time_seconds': elapsed}
        raise NotImplementedError("TN 1 cần implement hàm này")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán class cho ảnh mới.

        Args:
            X: Ma trận đặc trưng đã qua PCA, shape (n_samples, n_components)

        Returns:
            Array nhãn dự đoán, shape (n_samples,)
            Ví dụ: [0, 5, 12, ...] — chỉ số class
        """
        # TODO:
        # if not self.is_fitted:
        #     raise RuntimeError("Cần gọi train() trước khi predict()")
        # return self.model.predict(X)
        raise NotImplementedError("TN 1 cần implement hàm này")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán xác suất cho từng class (dùng để hiển thị confidence score).

        Args:
            X: Ma trận đặc trưng đã qua PCA, shape (n_samples, n_components)

        Returns:
            Ma trận xác suất shape (n_samples, n_classes)
            Mỗi hàng là xác suất cho 36 class, tổng = 1.0
        """
        # TODO:
        # if not self.is_fitted:
        #     raise RuntimeError("Cần gọi train() trước")
        # return self.model.predict_proba(X)
        raise NotImplementedError("TN 1 cần implement hàm này")

    def predict_single(self, feature_vector: np.ndarray, top_k: int = 3) -> list:
        """
        Dự đoán 1 ảnh và trả về Top-K class có xác suất cao nhất.
        Dùng trong Streamlit để hiển thị kết quả.

        Args:
            feature_vector: Vector 1D shape (n_components,) — đặc trưng 1 ảnh sau PCA
            top_k: Trả về bao nhiêu class cao nhất

        Returns:
            List of tuples: [(class_index, class_name, probability), ...]
            Ví dụ: [(5, 'Carrot', 0.92), (12, 'Paprika', 0.05), ...]
        """
        # TODO:
        # proba = self.predict_proba(feature_vector.reshape(1, -1))[0]
        # top_indices = np.argsort(proba)[::-1][:top_k]
        # if self.classes is not None:
        #     return [(int(i), self.classes[i], float(proba[i])) for i in top_indices]
        # else:
        #     return [(int(i), f"class_{i}", float(proba[i])) for i in top_indices]
        raise NotImplementedError("TN 1 cần implement hàm này")

    def save(self, path: str) -> None:
        """
        Lưu model đã train ra file để dùng lại sau.

        Args:
            path: Đường dẫn lưu file, ví dụ "checkpoints/svm_model.pkl"
        """
        # TODO:
        # if not self.is_fitted:
        #     raise RuntimeError("Cần train() trước khi save()")
        # Path(path).parent.mkdir(parents=True, exist_ok=True)
        # joblib.dump(self, path)
        # print(f"Đã lưu SVM model tại: {path}")
        raise NotImplementedError("TN 1 cần implement hàm này")

    @classmethod
    def load(cls, path: str) -> "ClassicalSVMClassifier":
        """
        Load model đã lưu từ file.

        Args:
            path: Đường dẫn file .pkl

        Returns:
            ClassicalSVMClassifier object đã train sẵn
        """
        # TODO:
        # model = joblib.load(path)
        # print(f"Đã load SVM model từ: {path}")
        # return model
        raise NotImplementedError("TN 1 cần implement hàm này")


def run_full_pipeline(
    data_dir: str = "data/raw",
    checkpoint_dir: str = "checkpoints",
    report_dir: str = "reports",
):
    """
    Hàm chạy toàn bộ pipeline từ đầu đến cuối:
    Load data → Extract features → PCA → Train SVM → Evaluate → Save

    TN 3 gọi hàm này để train với toàn bộ dataset.

    Args:
        data_dir: Thư mục chứa data (có train/ validation/ test/ bên trong)
        checkpoint_dir: Nơi lưu model sau khi train
        report_dir: Nơi lưu kết quả đánh giá
    """
    # TODO: Đây là hàm dài, cần implement sau khi các hàm trên xong
    # Các bước:
    # 1. Load danh sách file ảnh và nhãn từ data_dir
    # 2. batch_extract_features() cho train, val, test
    # 3. PCAWrapper().fit_transform(X_train)
    # 4. ClassicalSVMClassifier().train(X_train_pca, y_train)
    # 5. Evaluate trên test set
    # 6. Lưu kết quả vào report_dir
    raise NotImplementedError("Implement sau khi train() và predict() xong")


# ============================================================
# TEST NHANH
# python src/models/classical_svm.py
# ============================================================
if __name__ == "__main__":
    print("=== Test Classical SVM Classifier ===")
    print()
    print("TODO: Sau khi TN 2 làm xong hog_features.py và pca_wrapper.py,")
    print("bạn có thể test bằng cách:")
    print()
    print("  X_fake = np.random.randn(100, 200)  # 100 ảnh, 200 features sau PCA")
    print("  y_fake = np.random.randint(0, 36, 100)  # 36 class")
    print("  model = ClassicalSVMClassifier(c_values=[1], gamma_values=['scale'])")
    print("  model.train(X_fake, y_fake)")
    print("  preds = model.predict(X_fake[:5])")
    print("  print(preds)  # In ra 5 nhãn dự đoán")

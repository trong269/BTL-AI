import numpy as np
import pickle
import os

class MyLinearRegression:
    def __init__(self, input_dim=None):
        self.w = None
        self.input_dim = input_dim
        if input_dim is not None:
            self.initialize_weights(input_dim)

    def initialize_weights(self, input_dim):
        """Khởi tạo trọng số với số chiều đầu vào xác định"""
        np.random.seed(42)
        self.w = np.random.randn(1, input_dim)
        self.input_dim = input_dim

    def predict(self, X):
        """Dự đoán giá trị dựa trên trọng số hiện tại"""
        if self.w is None:
            raise ValueError("Mô hình chưa được huấn luyện. Hãy chạy phương thức train() trước.")
        y_pred = np.dot(X, self.w.T)
        return y_pred

    def mse_loss(self, y_true, y_pred):
        """Tính toán Mean Squared Error loss"""
        return np.mean(np.square(y_true - y_pred))

    def mse_gradient(self, y_pred, y_true, X_train):
        """Tính gradient của hàm MSE loss"""
        n = X_train.shape[0]
        gradient = (2/n) * X_train.T @ (y_pred - y_true)
        return gradient.T

    def train(self, train_X, train_y, test_X=None, test_y=None, learning_rate=0.01, epochs=1000):
        """Huấn luyện mô hình với Gradient Descent"""
        n = train_X.shape[0]
        d = train_X.shape[1]

        # Khởi tạo trọng số nếu chưa được khởi tạo
        if self.w is None:
            self.initialize_weights(d)
        elif self.w.shape[1] != d:
            print(f"Cảnh báo: Số chiều dữ liệu ({d}) khác với số chiều trọng số ({self.w.shape[1]}). Khởi tạo lại trọng số.")
            self.initialize_weights(d)

        np.random.seed(42)
        train_losses = []
        test_losses = []

        for i in range(epochs):
            y_pred = self.predict(train_X)
            train_loss = self.mse_loss(train_y, y_pred)
            train_losses.append(train_loss)

            if test_X is not None and test_y is not None:
                y_test_pred = self.predict(test_X)
                test_loss = self.mse_loss(test_y, y_test_pred)
                test_losses.append(test_loss)
            else:
                test_loss = None

            if (i + 1) % 500 == 0 or i + 1 == 1:
                if test_loss is not None:
                    print(f"epoch {i + 1}: train loss = {train_loss:.5f} | test loss = {test_loss:.5f}")
                else:
                    print(f"epoch {i + 1}: train loss = {train_loss:.5f}")

            grad = self.mse_gradient(y_pred, train_y, train_X)
            self.w = self.w - learning_rate * grad

        return train_losses, test_losses

    def save_model(self, file_path):
        """Lưu mô hình vào file"""
        model_data = {
            'weights': self.w,
            'input_dim': self.input_dim
        }

        # Đảm bảo thư mục tồn tại
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Mô hình đã được lưu tại: {file_path}")

    @classmethod
    def load_model(cls, file_path):
        """Tải mô hình từ file và trả về một instance mới"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {file_path}")

        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        # Tạo instance mới
        model = cls()
        model.w = model_data['weights']
        model.input_dim = model_data['input_dim']
        return model

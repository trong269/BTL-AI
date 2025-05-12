from processor import Processor
from LinearRegression import MyLinearRegression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Service:
    def __init__(self):
        self.processor = Processor()
        self.model = MyLinearRegression(input_dim=10)
    
    def process_data(self, X_train, X_test, y_train, y_test):
        # thêm một chiều cho bias vào trong X
        train_hat_X = np.concatenate( (np.ones((X_train.shape[0], 1)), X_train), axis = 1)
        test_hat_X = np.concatenate( (np.ones((X_test.shape[0], 1)), X_test), axis = 1)
        train_y = np.reshape(y_train, (y_train.shape[0], 1))
        test_y = np.reshape(y_test, (y_test.shape[0], 1))
        # tính số lượng mẫu n và số chiều d
        n = train_hat_X.shape[ 0 ]
        d = train_hat_X.shape[ 1 ]

        # in ra các tham số
        print( f"số lượng mẫu của bộ dữ liệu training là: {n}" )
        print( f"số chiều của mỗi mẫu là: {d}" )
        return train_hat_X, test_hat_X, train_y, test_y, n , d

    def train(self, df, test_size=0.2, learning_rate=1e-3, epochs=10000):
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = self.processor.process_for_train_test(df, test_size)
        # Tiền xử lý dữ liệu
        train_hat_X, test_hat_X, train_y, test_y, n , d = self.process_data(X_train, X_test, y_train, y_test)
        # training 
        train_losses, test_losses = self.model.train(train_hat_X, train_y, test_hat_X, test_y, learning_rate=learning_rate, epochs=epochs)
        # lưu biểu đồ loss
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss')
        plt.legend()
        plt.grid(True)
        # plt.savefig(r'.\images\train_test_losses.png')
        plt.close()
        # evaluate
        y_pred = self.model.predict(test_hat_X)
        y_true = test_y.copy()
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print("R2 score:", r2)
        print("MAE:" , mae)
        print("RMSE:", rmse)
        # Lưu mô hình
        model_path = r'./model/linear_regression_model.pkl'
        # self.model.save_model(model_path)      
        return self.model
    
    def inference(self, df ):
        # Tiền xử lý dữ liệu
        processed_X = self.processor.process_for_inference(df)
        # thêm một chiều cho bias vào trong X
        hat_X = np.concatenate( (np.ones((processed_X.shape[0], 1)), processed_X), axis = 1)
        # dự đoán
        y_pred = self.model.predict(hat_X)
        return y_pred

# if __name__ == "__main__":
#     service = Service()
#     # Giả sử df là DataFrame đã được tải từ file CSV
#     df = pd.read_csv(r'./data/housing.csv')
#     # Huấn luyện mô hình
#     model = service.train(df, test_size=0.2, learning_rate=1e-3, epochs=10000)

#     data = {
#         'longitude': -118.25,
#         'latitude': 34.05,
#         'housing_median_age': 20,
#         'total_rooms': 2000,
#         'total_bedrooms': 3,
#         'population': 500,
#         'households': 200,
#         'median_income': 3.5,
#         'ocean_proximity': 'NEAR BAY',
#     }
#     predicted_price = service.inference(pd.DataFrame([data]))
#     print(predicted_price)
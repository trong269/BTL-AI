import pandas as pd
import numpy as np
# Mô hình cho PreProcessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# phân cụm địa lý
from sklearn.cluster import KMeans

class Processor:
    def __init__(self, n_clusters: int = 8):
        self.numerical_pipeline = Pipeline([
                                        ('imputer', SimpleImputer(strategy="median")),
                                        ('scaler', StandardScaler())
                                    ])
        self.categorical_pipeline = Pipeline([
                                            ('onehot', OneHotEncoder(handle_unknown="ignore"))
                                        ])
        self.n_clusters = n_clusters
        self.geo_scaler = StandardScaler()
        self.cluster = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.preprocessor = None
    def process_for_train_test(self, df , test_size : float =0.2):
        # quá trình EDA
        # Tỷ lệ phòng ngủ trên tổng số phòng
        df['bedrooms_ratio'] = df['total_bedrooms'] / df['total_rooms']
        # Số phòng trung bình mỗi hộ gia đình
        df['rooms_per_household'] = df['total_rooms'] / df['households']
        # Số người trung bình mỗi phòng
        df['population_per_room'] = df['population'] / df['total_rooms']
        # Số người trung bình mỗi hộ
        df['population_per_household'] = df['population'] / df['households']
        # Tạo biến phân loại cho tuổi nhà
        age_bins = [0, 15, 40, 100] 
        age_labels = ['mới (<15 năm)', 'trung bình (15-40 năm)', 'cũ (>40 năm)']
        df['age_category'] = pd.cut(df['housing_median_age'], bins=age_bins, labels=age_labels, right=False)
        df['age_category'] = df['age_category'].astype('category')
        # Rời rạc hóa thu nhập theo phân vị (quantiles)
        income_bins = [0, 1, 2, 5, 8, np.inf]
        income_labels = ['rất thấp', 'thấp', 'trung bình', 'cao', 'rất cao']
        df['income_category'] = pd.cut(
            df['median_income'], 
            bins=income_bins, 
            labels=income_labels, 
            right=False
        )
        df['income_category'] = df['income_category'].astype('category')
        # phân cụm dịa lý
        geo_data = df[['longitude', 'latitude']].copy()
        # Chuẩn hóa dữ liệu tọa độ để cải thiện kết quả phân cụm
        geo_scaled = self.geo_scaler.fit_transform(geo_data)
        df['cluster_id'] = self.cluster.fit_predict(geo_scaled)        
        df['cluster_id'] = df['cluster_id'].astype('category')

        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X = df.drop("median_house_value", axis=1)
        y = df["median_house_value"]
        # phân loại thuộc tính
        numerical_cols = X.select_dtypes(include=["float64", 'int64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", 'category']).columns.tolist()
        # Tạo preprocessor
        self.preprocessor = ColumnTransformer([
                                            ("num", self.numerical_pipeline, numerical_cols),
                                            ("cat", self.categorical_pipeline, categorical_cols)
                                        ])
        # tiền xử lý dữ liệu
        processed_X = self.preprocessor.fit_transform(X)
        # Đếm tổng số giá trị NaN trong dữ liệu đã xử lý
        missing_count = np.isnan(processed_X).sum()
        print(f"Tổng số giá trị thiếu trong X_new: {missing_count}")
        return train_test_split(processed_X, y, test_size=test_size, random_state=42)
    
    def process_for_inference(self, df):
        # quá trình EDA
        # Tỷ lệ phòng ngủ trên tổng số phòng
        df['bedrooms_ratio'] = df['total_bedrooms'] / df['total_rooms']
        # Số phòng trung bình mỗi hộ gia đình
        df['rooms_per_household'] = df['total_rooms'] / df['households']
        # Số người trung bình mỗi phòng
        df['population_per_room'] = df['population'] / df['total_rooms']
        # Số người trung bình mỗi hộ
        df['population_per_household'] = df['population'] / df['households']
        # Tạo biến phân loại cho tuổi nhà
        age_bins = [0, 15, 40, 100] 
        age_labels = ['mới (<15 năm)', 'trung bình (15-40 năm)', 'cũ (>40 năm)']
        df['age_category'] = pd.cut(df['housing_median_age'], bins=age_bins, labels=age_labels, right=False)
        df['age_category'] = df['age_category'].astype('category')
        # Rời rạc hóa thu nhập theo phân vị (quantiles)
        income_bins = [0, 1, 2, 5, 8, np.inf]
        income_labels = ['rất thấp', 'thấp', 'trung bình', 'cao', 'rất cao']
        df['income_category'] = pd.cut(
            df['median_income'], 
            bins=income_bins, 
            labels=income_labels, 
            right=False
        )
        df['income_category'] = df['income_category'].astype('category')
        # phân cụm dịa lý
        geo_data = df[['longitude', 'latitude']].copy()
        # Chuẩn hóa dữ liệu tọa độ để cải thiện kết quả phân cụm
        
        geo_scaled = self.geo_scaler.transform(geo_data)
        df['cluster_id'] = self.cluster.predict(geo_scaled)
        df['cluster_id'] = df['cluster_id'].astype('category') 

        X = df.copy()
        # phân loại thuộc tính
        numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", 'category']).columns.tolist()

        # tiền xử lý dữ liệu
        processed_X = self.preprocessor.transform(X)
        return processed_X
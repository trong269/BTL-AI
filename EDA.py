import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r'.\data\housing.csv')

# 3. Phân tích trực quan dữ liệu (EDA)
# 3.1. Phân bố của biến mục tiêu
plt.figure(figsize=(10, 6))
sns.histplot(df['median_house_value'], kde=True)
plt.title('Phân bố giá nhà')
plt.xlabel('Giá nhà trung bình (USD)')
plt.ylabel('Tần số')
plt.savefig(r'.\images\house_price_distribution.png')
plt.close()


# 3.3. Phân bố không gian của dữ liệu
plt.figure(figsize=(12, 10))
plt.scatter(df['longitude'], df['latitude'], alpha=0.3,
            s=df['population']/100, c=df['median_house_value'], cmap='viridis')
plt.colorbar(label='Giá nhà trung bình (USD)')
plt.title('Phân bố không gian của giá nhà ở California')
plt.xlabel('Kinh độ')
plt.ylabel('Vĩ độ')
plt.savefig(r'.\images\spatial_distribution.png')
plt.close()

# 3.4. Ma trận tương quan
plt.figure(figsize=(14, 10))
numerical_features = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Ma trận tương quan giữa các biến số')
plt.savefig(r'.\images\correlation_matrix.png')
plt.close()

# 3.5. Boxplot cho ocean_proximity
plt.figure(figsize=(14, 8))
sns.boxplot(x='ocean_proximity', y='median_house_value', data=df)
plt.title('Giá nhà theo khoảng cách đến biển')
plt.xlabel('Khoảng cách đến biển')
plt.ylabel('Giá nhà trung bình (USD)')
plt.xticks(rotation=45)
plt.savefig(r'.\images\ocean_vs_price.png')
plt.close()

# 3.6. Phân tích mối quan hệ với biến mục tiêu
plt.figure(figsize=(16, 12))
numerical_features = ['housing_median_age', 'total_rooms', 'total_bedrooms',
                     'population', 'households', 'median_income']

for i, feature in enumerate(numerical_features):
    plt.subplot(3, 2, i+1)
    sns.scatterplot(x=feature, y='median_house_value', data=df, alpha=0.5)
    plt.title(f'Mối quan hệ: {feature} vs giá nhà')
plt.tight_layout()
plt.savefig(r'.\images\feature_relationships.png')
plt.close()

# 4. Tạo các đặc trưng mới
# 4.1. Tỷ lệ phòng ngủ trên tổng số phòng
df['bedrooms_ratio'] = df['total_bedrooms'] / df['total_rooms']

# 4.2. Số phòng trung bình mỗi hộ gia đình
df['rooms_per_household'] = df['total_rooms'] / df['households']

# 4.3. Số người trung bình mỗi phòng
df['population_per_room'] = df['population'] / df['total_rooms']

# 4.4. Số người trung bình mỗi hộ
df['population_per_household'] = df['population'] / df['households']

# 4.5. Biến tuổi nhà thành biến categorical
# Phân tích phân bố của tuổi nhà
plt.figure(figsize=(10, 6))
sns.histplot(df['housing_median_age'], bins=20, kde=True)
plt.title('Phân bố tuổi nhà')
plt.xlabel('Tuổi nhà trung bình')
plt.ylabel('Tần số')
plt.savefig(r'.\images\house_age_distribution.png')
plt.close()

# Xác định các nhóm tuổi nhà phù hợp
# Sử dụng phương pháp phân vị để chia thành 3 nhóm
age_bins = [0, 15, 40, 100] 
age_labels = ['mới (<15 năm)', 'trung bình (15-40 năm)', 'cũ (>40 năm)']

# Tạo biến phân loại cho tuổi nhà
df['age_category'] = pd.cut(df['housing_median_age'], bins=age_bins, labels=age_labels, right=False)

# Kiểm tra phân bố của các nhóm tuổi nhà
age_category_counts = df['age_category'].value_counts().sort_index()
print("\nPhân bố các nhóm tuổi nhà:")
print(age_category_counts)

# Biểu đồ thể hiện giá nhà theo nhóm tuổi
plt.figure(figsize=(12, 6))
sns.boxplot(x='age_category', y='median_house_value', data=df)
plt.title('Giá nhà theo nhóm tuổi')
plt.xlabel('Nhóm tuổi nhà')
plt.ylabel('Giá nhà trung bình (USD)')
plt.savefig(r'.\images\house_price_by_age_category.png')
plt.close()

# 4.6. Rời rạc hóa thu nhập trung bình (median_income)
# Phân tích phân bố của thu nhập trung bình
plt.figure(figsize=(10, 6))
sns.histplot(df['median_income'], kde=True, bins=30)
plt.title('Phân bố thu nhập trung bình')
plt.xlabel('Thu nhập trung bình (thousands of USD)')
plt.ylabel('Tần số')
plt.savefig(r'.\images\income_distribution.png')
plt.close()

# Xem các thống kê cơ bản của thu nhập
print("\nThống kê thu nhập trung bình:")
print(df['median_income'].describe())

# Rời rạc hóa thu nhập theo phân vị (quantiles)
# Chia thành 5 nhóm thu nhập dựa trên ngưỡng cụ thể
income_bins = [0, 1, 2, 5, 8, np.inf]
income_labels = ['rất thấp', 'thấp', 'trung bình', 'cao', 'rất cao']

df['income_category'] = pd.cut(
    df['median_income'], 
    bins=income_bins, 
    labels=income_labels, 
    right=False
)

# Kiểm tra phân bố của các nhóm thu nhập
income_category_counts = df['income_category'].value_counts().sort_index()
print("\nPhân bố các nhóm thu nhập:")
print(income_category_counts)

# Biểu đồ thể hiện giá nhà theo nhóm thu nhập
plt.figure(figsize=(12, 6))
sns.boxplot(x='income_category', y='median_house_value', data=df)
plt.title('Giá nhà theo nhóm thu nhập')
plt.xlabel('Nhóm thu nhập')
plt.ylabel('Giá nhà trung bình (USD)')
plt.savefig(r'.\images\house_price_by_income_category.png')
plt.close()


# 4.7. Phân cụm địa lý (Geographic Clustering)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Chuẩn bị dữ liệu cho phân cụm
geo_data = df[['longitude', 'latitude']].copy()

# Chuẩn hóa dữ liệu tọa độ để cải thiện kết quả phân cụm
geo_scaler = StandardScaler()
geo_scaled = geo_scaler.fit_transform(geo_data)


# Áp dụng KMeans với số lượng cụm là 8 (tốt nhất theo thử nghiệm trước đó)
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster_id'] = kmeans.fit_predict(geo_scaled)

# Phân tích các cụm
# Vẽ bản đồ phân cụm
plt.figure(figsize=(12, 10))
plt.scatter(df['longitude'], df['latitude'], c=df['cluster_id'],
            cmap='viridis', s=30, alpha=0.7)
plt.colorbar(label='Cluster ID')
plt.title(f'Phân cụm địa lý với {n_clusters} cụm')
plt.xlabel('Kinh độ')
plt.ylabel('Vĩ độ')
plt.savefig(r'.\images\geographic_clusters.png')
plt.close()

# Phân tích giá nhà theo cụm
plt.figure(figsize=(14, 8))
sns.boxplot(x='cluster_id', y='median_house_value', data=df)
plt.title('Giá nhà theo cụm địa lý')
plt.xlabel('Cluster ID')
plt.ylabel('Giá nhà trung bình (USD)')
plt.savefig(r'.\images\house_price_by_cluster.png')
plt.close()

# Thống kê mô tả cho từng cụm
cluster_stats = df.groupby('cluster_id')['median_house_value'].agg(['mean', 'median', 'std', 'count'])
print("\nThống kê giá nhà theo cụm địa lý:")
print(cluster_stats.sort_values(by='mean', ascending=False))

# Vẽ bản đồ phân cụm với màu sắc thể hiện giá nhà trung bình của mỗi cụm
cluster_means = df.groupby('cluster_id')['median_house_value'].mean().to_dict()
df['cluster_mean_price'] = df['cluster_id'].map(cluster_means)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(df['longitude'], df['latitude'], c=df['cluster_mean_price'],
                    cmap='viridis', s=30, alpha=0.7)
plt.colorbar(scatter, label='Giá nhà trung bình (USD)')
plt.title('Bản đồ phân cụm với giá nhà trung bình theo cụm')
plt.xlabel('Kinh độ')
plt.ylabel('Vĩ độ')
plt.savefig(r'.\images\geographic_clusters_with_price.png')
plt.close()

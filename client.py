import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px

# Tiêu đề ứng dụng
st.title('Dự đoán giá nhà California')
st.write('Nhập thông tin về căn nhà để dự đoán giá')

# URL của API backend
API_URL = "http://localhost:8000/predict"  # Thay đổi URL này phù hợp với backend của bạn

# Tạo layout với 2 cột
col1, col2 = st.columns(2)

# Cột 1: Nhập các thông tin cơ bản về nhà
with col1:
    st.subheader('Thông tin cơ bản')
    
    # Vị trí địa lý
    longitude = st.slider('Kinh độ (longitude)', -124.0, -114.0, -119.0, 0.1, 
                       help='Kinh độ của địa điểm (giá trị càng âm càng về phía tây)')
    latitude = st.slider('Vĩ độ (latitude)', 32.0, 42.0, 37.0, 0.1,
                      help='Vĩ độ của địa điểm (giá trị càng lớn càng về phía bắc)')
    
    # Tuổi nhà
    housing_median_age = st.slider('Tuổi nhà trung bình (năm)', 1, 52, 20, 
                                help='Tuổi trung bình của các căn nhà trong khu vực')
    
    # Khoảng cách đến biển
    ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
    ocean_proximity = st.selectbox('Khoảng cách đến biển', ocean_proximity_options,
                                help='Mô tả khoảng cách từ khu vực đến đại dương/biển/vịnh')

# Cột 2: Nhập thông tin về quy mô và thu nhập
with col2:
    st.subheader('Quy mô và thu nhập')
    
    # Tổng số phòng và phòng ngủ
    total_rooms = st.number_input('Tổng số phòng', min_value=1, value=2000, step=100,
                               help='Tổng số phòng trong khu vực')
    total_bedrooms = st.number_input('Tổng số phòng ngủ', min_value=1, value=500, step=50,
                                  help='Tổng số phòng ngủ trong khu vực')
    
    # Dân số và hộ gia đình
    population = st.number_input('Dân số', min_value=1, value=1000, step=100,
                              help='Tổng dân số trong khu vực')
    households = st.number_input('Số hộ gia đình', min_value=1, value=500, step=50,
                              help='Số hộ gia đình trong khu vực')
    
    # Thu nhập
    median_income = st.slider('Thu nhập trung bình (x $1000)', 0.0, 15.0, 4.0, 0.1,
                           help='Thu nhập trung bình của hộ gia đình trong khu vực (thousand of USD)')

# Tính các đặc trưng bổ sung
if total_rooms > 0 and households > 0 and total_bedrooms > 0 and population > 0:
    st.subheader('Đặc trưng phái sinh (tự động tính)')
    
    bedrooms_ratio = total_bedrooms / total_rooms
    rooms_per_household = total_rooms / households
    population_per_room = population / total_rooms
    population_per_household = population / households
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.metric("Tỷ lệ phòng ngủ/tổng số phòng", f"{bedrooms_ratio:.2f}")
        st.metric("Số người/phòng", f"{population_per_room:.2f}")
        
    with col4:
        st.metric("Số phòng/hộ", f"{rooms_per_household:.2f}")
        st.metric("Số người/hộ", f"{population_per_household:.2f}")

# Xác định nhóm thu nhập
def get_income_category(income):
    if income < 2:
        return "rất thấp"
    elif income < 3:
        return "thấp"
    elif income < 4:
        return "trung bình"
    elif income < 5:
        return "cao"
    else:
        return "rất cao"

income_category = get_income_category(median_income)
st.info(f"Nhóm thu nhập: {income_category}")

# Xác định nhóm tuổi nhà
def get_age_category(age):
    if age < 15:
        return "mới (<15 năm)"
    elif age < 40:
        return "trung bình (15-40 năm)"
    else:
        return "cũ (>40 năm)"

age_category = get_age_category(housing_median_age)
st.info(f"Nhóm tuổi nhà: {age_category}")


# Chuẩn bị dữ liệu để gửi đến API
input_data = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity,
}

# Hiển thị bản đồ vị trí
st.subheader('Vị trí')
df_map = pd.DataFrame({
    'lat': [latitude],
    'lon': [longitude]
})
st.map(df_map)

# Nút dự đoán
if st.button('Dự đoán giá nhà'):
    try:
        st.spinner('Đang dự đoán...')
        
        # Gọi API để dự đoán
        response = requests.post(API_URL, json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            predicted_price = result.get('predicted_price', 0)
            
            # Hiển thị kết quả
            st.success(f"Giá nhà dự đoán: ${predicted_price:,.3f}")
            
        else:
            st.error(f"Lỗi khi gọi API: {response.status_code} - {response.text}")
    
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {str(e)}")
        st.info("Đảm bảo rằng backend API đang chạy và URL đúng.")

# Giải thích các yếu tố ảnh hưởng đến giá nhà
with st.expander("Giải thích về các yếu tố ảnh hưởng đến giá nhà"):
    st.write("""
    - **Vị trí địa lý**: Khu vực ven biển và các thành phố lớn thường có giá cao hơn
    - **Khoảng cách đến biển**: Các khu vực gần biển thường có giá cao hơn
    - **Thu nhập trung bình**: Khu vực có thu nhập cao thường có giá nhà cao hơn
    - **Số phòng/hộ**: Phản ánh kích thước nhà, càng nhiều phòng thì giá càng cao
    - **Tuổi nhà**: Nhà mới thường có giá cao hơn nhà cũ (tùy khu vực)
    - **Mật độ dân số**: Khu vực quá đông đúc hoặc quá thưa thớt đều có thể ảnh hưởng tiêu cực
    """)

# Hướng dẫn sử dụng
with st.expander("Hướng dẫn sử dụng"):
    st.write("""
    1. Điều chỉnh các thanh trượt và nhập thông tin về căn nhà
    2. Xem các đặc trưng phái sinh được tính tự động
    3. Nhấn nút "Dự đoán giá nhà" để nhận kết quả
    4. Xem vị trí trên bản đồ để có cái nhìn trực quan hơn
    
    **Lưu ý**: Mô hình dự đoán dựa trên dữ liệu nhà ở California, kết quả có thể không chính xác cho các khu vực khác.
    """)

# Footer
st.markdown("---")
st.caption("Ứng dụng dự đoán giá nhà California | Dữ liệu từ bộ dữ liệu nhà ở California")
# Dự đoán giá nhà California

Đây là một dự án học máy phân tích và dự đoán giá nhà tại California dựa trên các yếu tố như vị trí địa lý, tuổi nhà, thu nhập trung bình khu vực và nhiều yếu tố khác.

## Cấu trúc dự án

```
├── client.py             # Ứng dụng web Streamlit
├── EDA.py                # Phân tích khám phá dữ liệu
├── LinearRegression.py   # Mô hình hồi quy tuyến tính
├── main.py               # API FastAPI
├── processor.py          # Tiền xử lý dữ liệu
├── service.py            # Dịch vụ xử lý và dự đoán
├── data/                 # Thư mục chứa bộ dữ liệu
│   └── housing.csv       # Dữ liệu nhà ở California
├── images/               # Thư mục chứa các biểu đồ
└── model/                # Thư mục lưu trữ mô hình
    └── linear_regression_model.pkl # Mô hình đã huấn luyện
```

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Pip (trình quản lý gói của Python)

## Cài đặt

1. **Sao chép mã nguồn:**

```bash
git clone https://github.com/trong269/BTL-AI.git
cd BTL-AI
```

2. **Tạo và kích hoạt môi trường ảo (tùy chọn nhưng khuyến khích):**

```bash
python -m venv venv
```

* Kích hoạt môi trường ảo:
  * Windows:
  ```bash
  venv\Scripts\activate
  ```
  * Linux/Mac:
  ```bash
  source venv/bin/activate
  ```

3. **Cài đặt các gói phụ thuộc:**

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Khởi động API backend

```bash
uvicorn main:app --reload
```

API sẽ chạy tại địa chỉ http://localhost:8000.

### 2. Khởi động giao diện người dùng Streamlit

Mở một terminal mới và chạy:

```bash
streamlit run client.py
```

Giao diện web sẽ tự động mở trong trình duyệt của bạn tại địa chỉ http://localhost:8501.

### 3. Sử dụng giao diện dự đoán giá nhà

- Điều chỉnh các thông số như vị trí địa lý, tuổi nhà, số phòng, thu nhập...
- Nhấn nút "Dự đoán giá nhà" để nhận kết quả dự đoán
- Xem vị trí trên bản đồ và các thông tin phái sinh

## Chạy phân tích dữ liệu (EDA)

Để chạy quá trình phân tích dữ liệu và tạo các biểu đồ trong thư mục `images/`:

```bash
python EDA.py
```

## Huấn luyện lại mô hình

Mô hình được huấn luyện tự động khi bạn khởi động API, nhưng bạn có thể thay đổi các tham số trong tập tin `main.py`:

```python
service.train(df=df, test_size=0.2, learning_rate=1e-3, epochs=10000)
```

## Các tính năng chính

1. **Phân tích dữ liệu trực quan**:
   - Phân bố giá nhà
   - Phân tích không gian
   - Ma trận tương quan
   - Mối quan hệ giữa các đặc trưng

2. **Dự đoán giá nhà**:
   - Sử dụng mô hình hồi quy tuyến tính
   - Xử lý dữ liệu đầu vào
   - Cung cấp kết quả dự đoán chính xác

3. **Giao diện người dùng trực quan**:
   - Điều khiển trượt và các trường nhập liệu dễ sử dụng
   - Hiển thị vị trí trên bản đồ
   - Tính toán các đặc trưng phái sinh
   - Phân loại thu nhập và tuổi nhà

## Dữ liệu

Dự án sử dụng bộ dữ liệu nhà ở California, bao gồm các thông tin:
- Vị trí địa lý (kinh độ, vĩ độ)
- Tuổi nhà trung bình
- Tổng số phòng và phòng ngủ
- Dân số và số hộ gia đình
- Thu nhập trung bình
- Khoảng cách đến biển
- Giá nhà trung bình (biến mục tiêu)

## Đóng góp

Nếu bạn muốn đóng góp vào dự án, vui lòng:
1. Fork dự án
2. Tạo một nhánh tính năng (`git checkout -b feature/amazing-feature`)
3. Commit thay đổi của bạn (`git commit -m 'Add some amazing feature'`)
4. Push lên nhánh của bạn (`git push origin feature/amazing-feature`)
5. Mở một Pull Request


## Liên hệ

Nếu có bất kỳ câu hỏi hoặc đóng góp nào, vui lòng liên hệ qua email: trongbg2692004@gmail.com
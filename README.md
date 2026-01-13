# Đồ án: Ứng dụng Python và Power BI trong phân tích và dự đoán xu hướng doanh thu sách trên Tiki

## Thông tin sinh viên
- Họ tên: Châu Gia Bảo 
- Mã lớp: DA22TTA  
- Email: giabao36925@gmail.com  
- Điện thoại: 0948017324

## 📋 Mô tả dự án

Dự án phân tích dữ liệu sách trên Tiki nhằm:
- Thu thập và làm sạch dữ liệu sách từ Tiki
- Phân tích xu hướng thị trường sách
- Xây dựng dashboard trực quan hóa với Power BI
- Xây dựng mô hình Machine Learning dự đoán doanh thu

## Cấu trúc thư mục

- `src/`: Mã nguồn tiền xử lý, EDA và mô hình ML  
- `data/`: Dữ liệu gốc và dữ liệu xử lý  
- `soft/`: Power BI Dashboard  
- `progress-report/`: Báo cáo tiến độ hàng tuần  
- `thesis/`: Các tài liệu đồ án (doc, pdf, ppt, refs)  
- `README.md`: Tập tin hướng dẫn sử dụng

## 🚀 Hướng dẫn chạy ứng dụng

### Chạy ứng dụng web (Streamlit)

```bash
# Di chuyển vào thư mục src
cd src

# Chạy app
streamlit run app.py
```

### Tính năng chính

- **📊 Dashboard**: Trực quan hóa dữ liệu bán hàng
- **🤖 Dự đoán doanh thu**: Sử dụng Random Forest để dự đoán
- **📈 Phân tích xu hướng**: Insight về thị trường sách

## 📦 Cấu trúc dữ liệu

- `data/book_data_cleaned_forPowerBI.csv`: Dữ liệu sách đã xử lý
- `data/comment_cleaned.csv`: Dữ liệu bình luận đã xử lý

## 🛠️ Công nghệ sử dụng

- **Python**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web App**: Streamlit
- **BI Tool**: Power BI
- **ML Model**: Random Forest Regressor, XGBoost

## 📊 Dashboards (Power BI)

### 1. Book Market Overview Dashboard
Dashboard tổng quan doanh thu, số lượng bán, đánh giá người dùng và phân bố doanh thu theo thể loại sách.

![Book Market Overview](thesis/abs/Book%20Market%20Overview%20Dashboard.png)

---

### 2. Revenue Analysis & Pricing Impact Dashboard
Phân tích mối quan hệ giữa giá bán, mức giảm giá và doanh thu, hỗ trợ đánh giá chiến lược định giá.

![Revenue Analysis & Pricing Impact](thesis/abs/Revenue%20Analysis%20%26%20Pricing%20Impact%20Dashboard.png)

---

### 3. Customer Sentiment Analysis Dashboard
Phân tích cảm xúc bình luận khách hàng bằng mô hình PhoBERT, kết hợp đánh giá sao và word cloud.

![Customer Sentiment Analysis](thesis/abs/Customer%20Sentiment%20Analysis%20Dashboard.png)

## 📧 Liên hệ

Nếu có thắc mắc, vui lòng liên hệ:
- Email: giabao36925@gmail.com
- Điện thoại: 0948017324

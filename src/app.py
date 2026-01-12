import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ================== LOAD MODELS ==================
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    xgb_path = os.path.join(BASE_DIR, "xgb_log_model_9features.pkl")
    rf_path = os.path.join(BASE_DIR, "rf_log_model_9features.pkl")
    
    xgb_model = joblib.load(xgb_path)
    
    # Try to load Random Forest, fallback to XGBoost if not found
    try:
        rf_model = joblib.load(rf_path)
    except FileNotFoundError:
        st.warning("⚠️ Random Forest model chưa được tạo. Chỉ sử dụng XGBoost. Chạy cell cuối trong model.ipynb để tạo RF model!")
        rf_model = None
    
    return xgb_model, rf_model

xgb_model, rf_model = load_models()

# ================== HÀM DỰ ĐOÁN ==================
def predict_book_revenue(n_review, current_price, avg_rating, discount, 
                         comment_rating_mean=0, comment_rating_std=0, model='random_forest'):
    """Dự đoán doanh thu sách với 9 features
    
    Parameters:
    - model: 'random_forest' (mặc định, tốt nhất) hoặc 'xgboost' (so sánh)
    """
    # Tính toán các features phụ
    price_review_ratio = current_price / (n_review + 1)
    rating_review_product = avg_rating * n_review
    discount_impact = discount * current_price
    
    # Tạo DataFrame với đúng thứ tự features
    input_data = pd.DataFrame({
        'n_review': [n_review],
        'current_price': [current_price],
        'avg_rating': [avg_rating],
        'discount': [discount],
        'comment_rating_mean': [comment_rating_mean],
        'comment_rating_std': [comment_rating_std],
        'price_review_ratio': [price_review_ratio],
        'rating_review_product': [rating_review_product],
        'discount_impact': [discount_impact]
    })
    
    # Chọn model
    selected_model = rf_model if model == 'random_forest' else xgb_model
    
    # Predict log(revenue)
    log_revenue_pred = selected_model.predict(input_data)
    revenue_pred = np.expm1(log_revenue_pred)[0]
    
    return revenue_pred

# ================== APP UI ==================
st.set_page_config(page_title="Dự đoán doanh thu sách", layout="wide")

st.title("📚 Dự đoán doanh thu sách với Machine Learning")
st.write("Sử dụng **Random Forest** (tốt nhất) hoặc **XGBoost** (so sánh) với **9 features**")

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("🤖 Chọn mô hình")
    
    # Only show RF option if model exists
    if rf_model is not None:
        model_options = ["Random Forest (Khuyến nghị ✅)", "XGBoost (So sánh 📊)"]
    else:
        model_options = ["XGBoost"]
        
    model_choice = st.radio(
        "Lựa chọn thuật toán:",
        model_options,
        help="Random Forest cho kết quả tốt nhất trên test set" if rf_model else "Chỉ có XGBoost khả dụng"
    )
    
    # Map to model name
    selected_model = 'random_forest' if 'Random Forest' in model_choice else 'xgboost'
    current_model = rf_model if selected_model == 'random_forest' else xgb_model
    model_name = "Random Forest" if selected_model == 'random_forest' else "XGBoost"
    
    # Model info
    if selected_model == 'random_forest':
        st.success("✅ **Random Forest** - Mô hình tốt nhất!")
        st.metric("Test R²", "0.80")
        st.metric("MAE", "98.6M VND")
        st.metric("RMSE", "324.5M VND")
    else:
        st.info("📊 **XGBoost** - So sánh với RF")
        st.metric("Test R²", "0.77")
        st.metric("MAE", "109.0M VND")
        st.metric("RMSE", "348.1M VND")
    
    st.markdown("---")
    st.header("📊 Feature Importance")
    
    features = [
        'n_review', 'current_price', 'avg_rating', 'discount',
        'comment_rating_mean', 'comment_rating_std',
        'price_review_ratio', 'rating_review_product', 'discount_impact'
    ]
    importance = current_model.feature_importances_
    
    fi_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Bar chart với gradient màu
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi_df)))
    ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='white', linewidth=1)
    ax.set_xlabel('Importance', fontsize=10, fontweight='bold')
    ax.set_title(f'{model_name} - Feature Importance', fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', labelsize=8)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Top 3 features
    st.markdown("### 🏆 Top 3 quan trọng nhất:")
    for idx, (_, row) in enumerate(fi_df.head(3).iterrows(), 1):
        st.write(f"**{idx}. {row['Feature']}**: {row['Importance']*100:.1f}%")
    
    st.markdown("---")
    st.markdown("### 📈 So sánh 2 mô hình")
    st.write("**Random Forest:**")
    st.write("• R² = 0.80 (cao nhất)")
    st.write("• Ổn định, học tốt phi tuyến")
    st.write("")
    st.write("**XGBoost:**")
    st.write("• R² = 0.77")
    st.write("• Nhanh, file nhẹ hơn")
    
    st.info(
        "💡 Cả 2 model đều dùng 9 features và không cần biết số lượng đã bán (quantity)!"
    )

st.markdown("---")

# ================== INPUT - 6 BIẾN CHÍNH ==================
st.markdown("### 📝 Nhập thông tin sách")

col1, col2, col3 = st.columns(3)

with col1:
    n_review = st.number_input(
        "⭐ Số lượng review",
        min_value=0,
        value=45,
        step=1,
        help="Số lượng đánh giá từ khách hàng"
    )

with col2:
    current_price = st.number_input(
        "💰 Giá bán (VND)",
        min_value=0,
        value=120_000,
        step=1_000,
        help="Giá bán hiện tại của sách"
    )

with col3:
    avg_rating = st.slider(
        "🌟 Điểm trung bình",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.1,
        help="Điểm đánh giá trung bình (1-5 sao)"
    )

# Row 2 - Discount
col4, col5 = st.columns(2)

with col4:
    discount_percent = st.slider(
        "🎁 Mức giảm giá (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="Phần trăm giảm giá"
    )

with col5:
    discount = st.number_input(
        "💰 Số tiền giảm (VND)",
        min_value=0,
        value=int(current_price * discount_percent / 100),
        step=1_000,
        help="Số tiền được giảm (tự động tính từ %)"
    )

# ================== PREDICT ==================
if st.button("🚀 Dự đoán doanh thu", use_container_width=True, type="primary"):
    # Gọi hàm dự đoán với model đã chọn
    revenue_pred = predict_book_revenue(
        n_review=n_review,
        current_price=current_price,
        avg_rating=avg_rating,
        discount=discount,
        model=selected_model
    )

    st.markdown("---")
    st.markdown("## 📊 Kết quả dự đoán")

    # ✅ CHỈ HIỂN THỊ 1 KPI CHÍNH
    st.metric(
        f"🎯 Doanh thu dự đoán ({model_name})", 
        f"{revenue_pred:,.0f} VND",
        help=f"Model: {model_name} | R² = {'0.80' if selected_model == 'random_forest' else '0.77'}"
    )
    
    # ✅ SO SÁNH MODEL - CHỈ KHI USER BẬT
    with st.expander("🔍 So sánh với mô hình khác (nâng cao)", expanded=False):
        other_model = 'xgboost' if selected_model == 'random_forest' else 'random_forest'
        other_model_name = "XGBoost" if other_model == 'xgboost' else "Random Forest"
        revenue_pred_other = predict_book_revenue(
            n_review=n_review,
            current_price=current_price,
            avg_rating=avg_rating,
            discount=discount,
            model=other_model
        )
        
        diff = revenue_pred_other - revenue_pred
        diff_pct = (diff / revenue_pred * 100) if revenue_pred > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{model_name} (hiện tại)", f"{revenue_pred:,.0f} VND")
        with col2:
            st.metric(f"{other_model_name}", f"{revenue_pred_other:,.0f} VND", 
                     delta=f"{diff_pct:+.1f}%")
        
        # Insight về sự khác biệt
        if abs(diff_pct) < 5:
            st.info(f"💡 Kết quả tương đồng (chênh lệch {abs(diff_pct):.1f}%)")
        elif diff_pct > 5:
            st.warning(f"⚠️ {other_model_name} dự đoán cao hơn {abs(diff_pct):.1f}%")
        else:
            st.success(f"✅ {model_name} dự đoán cao hơn {abs(diff_pct):.1f}%")

    # Thông tin chi tiết
    st.markdown("---")
    st.markdown("### 💡 Phân tích đầu vào")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Thông tin sách:**")
        st.write(f"- Reviews: {n_review} đánh giá")
        st.write(f"- Rating: {avg_rating}/5.0 ⭐")
        st.write(f"- Giá bán: {current_price:,.0f} VND")
        st.write(f"- Giảm giá: {discount:,.0f} VND ({discount/current_price*100:.1f}%)")
    
    with col2:
        st.markdown("**🔍 Features tính toán:**")
        price_review_ratio = current_price / (n_review + 1)
        rating_review_product = avg_rating * n_review
        
        st.write(f"- Giá/Review: {price_review_ratio:,.0f}")
        st.write(f"- Rating×Review: {rating_review_product:,.1f}")
    
    # Insight dựa trên rating và review
    if avg_rating >= 4.5 and n_review >= 50:
        st.success(
            "✅ **Sách chất lượng cao!** Rating và review đều tốt, sách được ưa chuộng."
        )
    elif avg_rating < 3.5:
        st.warning(
            "⚠️ **Lưu ý:** Rating thấp có thể ảnh hưởng doanh thu. Cần cải thiện chất lượng."
        )
    elif n_review < 10:
        st.info(
            "📊 **Sách mới hoặc ít review:** Khuyến khích khách hàng review để tăng độ tin cậy!"
        )

    # ================== WHAT-IF ANALYSIS ==================
    st.markdown("---")
    st.markdown("## 🔮 What-If Analysis: Ảnh hưởng của Review & Rating")
    
    # Hiển thị baseline rõ ràng
    st.info(f"📌 **Điểm tham chiếu (Baseline):** Doanh thu hiện tại = **{revenue_pred:,.0f} VND** với {n_review} reviews và {avg_rating}⭐. Tất cả % thay đổi được tính so với giá trị này.")
    
    tab1, tab2 = st.tabs(["📈 Thay đổi Review", "⭐ Thay đổi Rating"])
    
    with tab1:
        st.info("Nếu tăng số lượng review, doanh thu sẽ thay đổi như thế nào? (giữ nguyên các yếu tố khác)")
                                                                                                      
        review_scenarios = [5, 10, 25, 50, 100, 200, 500]
        scenario_results = []
        
        for review_count in review_scenarios:
            pred = predict_book_revenue(
                n_review=review_count,
                current_price=current_price,
                avg_rating=avg_rating,
                discount=discount,
                model=selected_model
            )
            change_pct = ((pred - revenue_pred) / revenue_pred * 100) if revenue_pred > 0 else 0
            change_vnd = pred - revenue_pred
            is_current = '✅' if review_count == n_review else ''
            
            # Xác định xu hướng với text màu
            if change_vnd < 0:
                trend = '🔻'  # Giảm
            elif change_vnd > 0:
                trend = '🔺'  # Tăng
            else:
                trend = '➡️'  # Không đổi
            
            scenario_results.append({
                'Số Review': f"{review_count} {is_current}",
                '% Thay đổi': f"{change_pct:+.1f}%",
                'Thay đổi Doanh thu': f"{change_vnd:+,.0f} VND",
                'Xu hướng': trend
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        
        # Hàm tô màu toàn bộ dòng dựa trên xu hướng
        def color_rows(row):
            if '🔻' in str(row['Xu hướng']):
                return ['background-color: #ffebee'] * len(row)  # Đỏ nhạt cho cả dòng
            elif '🔺' in str(row['Xu hướng']):
                return ['background-color: #e8f5e9'] * len(row)  # Xanh nhạt cho cả dòng
            else:
                return [''] * len(row)
        
        styled_df = scenario_df.style.apply(color_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Kết luận cho Review
        st.markdown("""  
        **🔍 Kết luận:**
        - **Review là yếu tố quan trọng NHẤT** ảnh hưởng đến doanh thu sách.
        - Ngưỡng tối thiểu: **50 review** để có lợi nhuận dương.
        - Sweet spot: **100-200 review** - ROI cực kỳ cao (tăng 2-5 lần doanh thu).
        - **💡 Hành động:** Ưu tiên chiến lược khuyến khích khách hàng review (email marketing, voucher, loyalty program).
        """)
    
    with tab2:
        st.info("⭐ Nếu cải thiện rating, doanh thu sẽ thay đổi như thế nào? (giữ nguyên các yếu tố khác)")
        
        rating_scenarios = [3.0, 3.5, 4.0, 4.5, 5.0]
        rating_results = []
        
        for rating_val in rating_scenarios:
            pred = predict_book_revenue(
                n_review=n_review,
                current_price=current_price,
                avg_rating=rating_val,
                discount=discount,
                model=selected_model
            )
            change_pct = ((pred - revenue_pred) / revenue_pred * 100) if revenue_pred > 0 else 0
            change_vnd = pred - revenue_pred
            is_current = '✅' if abs(rating_val - avg_rating) < 0.01 else ''
            
            # Xác định xu hướng với text màu
            if change_vnd < 0:
                trend = '🔻'  # Giảm
            elif change_vnd > 0:
                trend = '🔺'  # Tăng
            else:
                trend = '➡️'  # Không đổi
            
            rating_results.append({
                'Rating': f"{rating_val:.1f}⭐ {is_current}",
                '% Thay đổi': f"{change_pct:+.1f}%",
                'Thay đổi Doanh thu': f"{change_vnd:+,.0f} VND",
                'Xu hướng': trend
            })
        
        rating_df = pd.DataFrame(rating_results)
        
        # Hàm tô màu toàn bộ dòng dựa trên xu hướng
        def color_rows_rating(row):
            if '🔻' in str(row['Xu hướng']):
                return ['background-color: #ffebee'] * len(row)  # Đỏ nhạt
            elif '🔺' in str(row['Xu hướng']):
                return ['background-color: #e8f5e9'] * len(row)  # Xanh nhạt
            else:
                return [''] * len(row)
        
        styled_rating_df = rating_df.style.apply(color_rows_rating, axis=1)
        st.dataframe(styled_rating_df, use_container_width=True, hide_index=True)
        
        # Kết luận cho Rating
        st.markdown("""  
        **🔍 Kết luận:**
        - **Rating có tác động YẾU hơn nhiều so với Review** (chỉ +1-2% khi tăng rating).
        - Duy trì rating ≥ 4.5⭐ là đủ - không cần quá lo lắng về 5.0 sao hoàn hảo.
        - Rating < 4.0 gây thiệt hại nhưng KHÔNG nghiêm trọng bằng thiếu review.
        - **💡 Hành động:** Tập trung vào **tăng SỐ LƯỢNG review** thay vì chỉ cải thiện rating.
        """)

    # Insights tổng thể
    st.markdown("---")
    st.markdown("### 🎯 Insight Tổng thể - Khuyến nghị Chiến lược")
    
    # So sánh tác động
    best_review = 200  # Sweet spot
    best_rating = 5.0
    
    pred_best_review = predict_book_revenue(best_review, current_price, avg_rating, discount, model=selected_model)
    pred_best_rating = predict_book_revenue(n_review, current_price, best_rating, discount, model=selected_model)
    
    review_potential = ((pred_best_review - revenue_pred) / revenue_pred * 100) if revenue_pred > 0 else 0
    rating_potential = ((pred_best_rating - revenue_pred) / revenue_pred * 100) if revenue_pred > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "📈 Tác động Review", 
            f"{review_potential:+.1f}%",
            delta="Tăng lên 200 reviews",
            help="Review có tác động CỰC KỲ MẠNH đến doanh thu"
        )
    
    with col2:
        st.metric(
            "⭐ Tác động Rating", 
            f"{rating_potential:+.1f}%",
            delta="Tăng lên 5.0 sao",
            help="Rating có tác động nhẹ hơn nhiều"
        )
    
    # Khuyến nghị chiến lược
    st.success(f"""
    **🚀 CÔNG THỨC THÀNH CÔNG:**
    
    ✅ **Ưu tiên #1: TĂNG SỐ LƯỢNG REVIEW**  
    → Mục tiêu: Đạt 100-200 review (hiện tại: {n_review})  
    → ROI kỳ vọng: Tăng **{review_potential:.0f}%** doanh thu ({(revenue_pred * review_potential / 100):,.0f} VND)
    
    ✅ **Ưu tiên #2: DUY TRÌ RATING ≥ 4.5⭐**  
    → Rating hiện tại: {avg_rating}/5.0 ⭐ - {'Đã tốt!' if avg_rating >= 4.5 else 'Cần cải thiện'}  
    → Tác động: Nhỏ hơn nhiều so với Review
    
    💡 **Hành động cụ thể:**
    - Email marketing sau mua hàng (yêu cầu review)
    - Tặng voucher/quà cho khách review
    - Chương trình loyalty: tích điểm khi review
    - Theo dõi và phản hồi review nhanh chóng
    """)

# ================== EXAMPLE SCENARIOS ==================
st.markdown("---")
st.markdown("## 📚 Kịch bản tham khảo")

example_scenarios = pd.DataFrame([
    {"Loại": "🔥 Bestseller", "Reviews": 200, "Rating": "4.8⭐", "Giá": "150K", "Giảm": "30K"},
    {"Loại": "📖 Phổ thông", "Reviews": 30, "Rating": "4.0⭐", "Giá": "120K", "Giảm": "15K"},
    {"Loại": "🆕 Mới", "Reviews": 5, "Rating": "4.5⭐", "Giá": "180K", "Giảm": "0"},
    {"Loại": "💎 Cao cấp", "Reviews": 100, "Rating": "4.9⭐", "Giá": "500K", "Giảm": "100K"}
])

st.dataframe(example_scenarios, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 🤖 Về mô hình")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Số features", "9", delta="Interaction + Raw")
    
with col2:
    st.metric("Random Forest R²", "0.80", delta="Tốt nhất ✅")
    
with col3:
    st.metric("XGBoost R²", "0.77", delta="So sánh 📊")

with col4:
    st.metric("Technique", "Log Transform", delta="Cả 2 model")

st.caption("💡 Tip: Thử chuyển đổi giữa Random Forest và XGBoost ở sidebar để so sánh kết quả!")

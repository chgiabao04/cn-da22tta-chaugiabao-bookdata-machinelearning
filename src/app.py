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
    
    # So sánh với model còn lại
    other_model = 'xgboost' if selected_model == 'random_forest' else 'random_forest'
    other_model_name = "XGBoost" if other_model == 'xgboost' else "Random Forest"
    revenue_pred_other = predict_book_revenue(
        n_review=n_review,
        current_price=current_price,
        avg_rating=avg_rating,
        discount=discount,
        model=other_model
    )

    st.markdown("---")
    st.markdown("## 📊 Kết quả dự đoán")

    # Main metric with comparison
    col_main1, col_main2 = st.columns(2)
    
    with col_main1:
        st.metric(
            f"🎯 {model_name} (đang chọn)", 
            f"{revenue_pred:,.0f} VND",
            delta="Kết quả chính"
        )
    
    with col_main2:
        diff = revenue_pred_other - revenue_pred
        diff_pct = (diff / revenue_pred * 100) if revenue_pred > 0 else 0
        st.metric(
            f"📊 {other_model_name} (so sánh)", 
            f"{revenue_pred_other:,.0f} VND",
            delta=f"{diff:+,.0f} VND ({diff_pct:+.1f}%)"
        )
    
    # Insight về sự khác biệt
    if abs(diff_pct) < 5:
        st.info(f"💡 **Kết quả tương đồng:** Cả 2 model đều dự đoán gần giống nhau (chênh lệch {abs(diff_pct):.1f}%)")
    elif diff_pct > 5:
        st.warning(f"⚠️ {other_model_name} dự đoán cao hơn {abs(diff_pct):.1f}% - Cân nhắc kiểm tra thêm")
    else:
        st.success(f"✅ {model_name} dự đoán cao hơn {abs(diff_pct):.1f}% - Tự tin với lựa chọn này")

    # Thông tin chi tiết
    st.markdown("---")
    st.markdown("### 💡 Phân tích")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Các yếu tố chính:**")
        st.write(f"- Reviews: {n_review} đánh giá")
        st.write(f"- Rating: {avg_rating}/5.0 ⭐")
        st.write(f"- Giá bán: {current_price:,.0f} VND")
        st.write(f"- Giảm giá: {discount:,.0f} VND")
    
    with col2:
        st.markdown("**🔍 Tính toán nâng cao:**")
        price_review_ratio = current_price / (n_review + 1)
        rating_review_product = avg_rating * n_review
        discount_impact = discount * current_price
        
        st.write(f"- Giá/Review: {price_review_ratio:,.0f}")
        st.write(f"- Rating×Review: {rating_review_product:,.1f}")
        st.write(f"- Tác động giảm giá: {discount_impact:,.0f}")
    
    # Insight dựa trên rating và review
    if avg_rating >= 4.5 and n_review >= 50:
        st.success(
            f"✅ **Sách chất lượng cao!** Rating tốt ({avg_rating}/5) và nhiều review ({n_review}) "
            "cho thấy sách rất được ưa chuộng."
        )
    elif avg_rating < 3.5:
        st.warning(
            f"⚠️ **Lưu ý:** Rating thấp ({avg_rating}/5) có thể ảnh hưởng đến doanh thu. "
            "Cần cải thiện chất lượng sản phẩm hoặc dịch vụ."
        )
    elif n_review < 10:
        st.info(
            f"📊 **Sách mới hoặc ít review:** Chỉ có {n_review} đánh giá. "
            "Khuyến khích khách hàng để lại review để tăng độ tin cậy!"
        )

    # ================== WHAT-IF ANALYSIS ==================
    st.markdown("---")
    st.markdown("## 🔮 What-If Analysis: Ảnh hưởng của Review & Rating")
    
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
            scenario_results.append({
                'Số review': review_count,
                'Doanh thu dự đoán': pred,
                'Chênh lệch': pred - revenue_pred
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        
        # Highlight current
        def highlight_current(row):
            if row['Số review'] == n_review:
                return ['background-color: #d4edda'] * len(row)
            return [''] * len(row)
        
        styled_scenario = scenario_df.style.format({
            'Doanh thu dự đoán': '{:,.0f} VND',
            'Chênh lệch': '{:+,.0f} VND'
        }).apply(highlight_current, axis=1)
        
        st.dataframe(styled_scenario, use_container_width=True, hide_index=True)
        
        # Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(scenario_df['Số review'], scenario_df['Doanh thu dự đoán'], 
                marker='o', linewidth=2, markersize=8, color='#2196F3')
        ax.axvline(x=n_review, color='green', linestyle='--', linewidth=2, label=f'Hiện tại: {n_review} reviews')
        ax.set_xlabel('Số lượng Review', fontsize=11, fontweight='bold')
        ax.set_ylabel('Doanh thu dự đoán (VND)', fontsize=11, fontweight='bold')
        ax.set_title('Ảnh hưởng của Review đến Doanh thu', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.info("Nếu cải thiện rating, doanh thu sẽ thay đổi như thế nào? (giữ nguyên các yếu tố khác)")
        
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
            rating_results.append({
                'Rating': rating_val,
                'Doanh thu dự đoán': pred,
                'Chênh lệch': pred - revenue_pred
            })
        
        rating_df = pd.DataFrame(rating_results)
        
        # Highlight current rating
        def highlight_rating(row):
            if abs(row['Rating'] - avg_rating) < 0.01:
                return ['background-color: #d4edda'] * len(row)
            return [''] * len(row)
        
        styled_rating = rating_df.style.format({
            'Rating': '{:.1f} ⭐',
            'Doanh thu dự đoán': '{:,.0f} VND',
            'Chênh lệch': '{:+,.0f} VND'
        }).apply(highlight_rating, axis=1)
        
        st.dataframe(styled_rating, use_container_width=True, hide_index=True)
        
        # Chart
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(rating_df['Rating'], rating_df['Doanh thu dự đoán'], 
                marker='s', linewidth=2, markersize=8, color='#FF9800')
        ax2.axvline(x=avg_rating, color='green', linestyle='--', linewidth=2, label=f'Hiện tại: {avg_rating} ⭐')
        ax2.set_xlabel('Rating (⭐)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Doanh thu dự đoán (VND)', fontsize=11, fontweight='bold')
        ax2.set_title('Ảnh hưởng của Rating đến Doanh thu', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Insights
    max_revenue_idx = scenario_df['Doanh thu dự đoán'].idxmax()
    max_revenue_row = scenario_df.iloc[max_revenue_idx]
    
    potential_increase = max_revenue_row['Doanh thu dự đoán'] - revenue_pred
    potential_pct = (potential_increase / revenue_pred * 100) if revenue_pred > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "🎯 Doanh thu tối đa (theo model)",
            f"{max_revenue_row['Doanh thu dự đoán']:,.0f} VND",
            delta=f"Khi có {max_revenue_row['Số review']:.0f} reviews"
        )
    
    with col2:
        st.metric(
            "📈 Tiềm năng tăng trưởng",
            f"{potential_increase:,.0f} VND",
            delta=f"+{potential_pct:.1f}%"
        )
    
    if potential_pct > 10:
        st.success(
            f"💡 **Insight:** Tăng số lượng review từ {n_review} lên {max_revenue_row['Số review']:.0f} "
            f"có thể giúp tăng doanh thu thêm **{potential_pct:.1f}%** ({potential_increase:,.0f} VND). "
            f"Nên khuyến khích khách hàng để lại review!"
        )
    else:
        st.info(
            f"📊 Số lượng review hiện tại ({n_review}) đã khá tối ưu. "
            f"Tăng thêm review chỉ cải thiện nhẹ doanh thu."
        )

# ================== EXAMPLE SCENARIOS ==================
st.markdown("---")
st.markdown("## 📚 Ví dụ tham khảo")

st.info("🎯 Các kịch bản mẫu để bạn tham khảo (dựa trên dữ liệu thực tế)")

example_scenarios = pd.DataFrame([
    {"Loại sách": "🔥 Bestseller", "Reviews": 200, "Rating": "4.8⭐", "Giá": "150,000", "Discount": "30,000"},
    {"Loại sách": "📖 Trung bình", "Reviews": 30, "Rating": "4.0⭐", "Giá": "120,000", "Discount": "15,000"},
    {"Loại sách": "🆕 Mới ra mắt", "Reviews": 5, "Rating": "4.5⭐", "Giá": "180,000", "Discount": "0"},
    {"Loại sách": "💰 Giá rẻ", "Reviews": 80, "Rating": "3.8⭐", "Giá": "50,000", "Discount": "5,000"},
    {"Loại sách": "💎 Cao cấp", "Reviews": 100, "Rating": "4.9⭐", "Giá": "500,000", "Discount": "100,000"}
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

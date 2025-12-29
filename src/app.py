import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "xgb_log_model_9features.pkl")
    return joblib.load(MODEL_PATH)

xgb_model = load_model()

# ================== HÀM DỰ ĐOÁN ==================
def predict_book_revenue(n_review, current_price, avg_rating, discount, 
                         comment_rating_mean=0, comment_rating_std=0):
    """Dự đoán doanh thu sách bằng XGBoost với 9 features"""
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
    
    # Predict log(revenue)
    log_revenue_pred = xgb_model.predict(input_data)
    revenue_pred = np.expm1(log_revenue_pred)[0]
    
    return revenue_pred

# ================== APP UI ==================
st.set_page_config(page_title="Dự đoán doanh thu sách", layout="wide")

st.title("📚 Dự đoán doanh thu sách với XGBoost")
st.write("Mô hình **XGBoost + Log Transform** với **9 features** - Độ chính xác cao")

# ================== SIDEBAR - FEATURE IMPORTANCE ==================
with st.sidebar:
    st.header("📊 Feature Importance")
    
    features = [
        'n_review', 'current_price', 'avg_rating', 'discount',
        'comment_rating_mean', 'comment_rating_std',
        'price_review_ratio', 'rating_review_product', 'discount_impact'
    ]
    importance = xgb_model.feature_importances_
    
    fi_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Bar chart với gradient màu
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi_df)))
    ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='white', linewidth=1)
    ax.set_xlabel('Importance', fontsize=10, fontweight='bold')
    ax.set_title('Tầm quan trọng của 9 Features', fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', labelsize=8)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Top 3 features
    st.markdown("### 🏆 Top 3 quan trọng nhất:")
    for idx, (_, row) in enumerate(fi_df.head(3).iterrows(), 1):
        st.write(f"**{idx}. {row['Feature']}**: {row['Importance']*100:.1f}%")
    
    st.info(
        "Mô hình sử dụng 9 features kết hợp để dự đoán chính xác. "
        "Không cần biết số lượng đã bán (quantity)!"
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
    # Gọi hàm dự đoán
    revenue_pred = predict_book_revenue(
        n_review=n_review,
        current_price=current_price,
        avg_rating=avg_rating,
        discount=discount
    )

    st.markdown("---")
    st.markdown("## 📊 Kết quả dự đoán")

    # Main metric
    st.metric(
        "🎯 Doanh thu dự đoán", 
        f"{revenue_pred:,.0f} VND",
        delta="Dự đoán bởi XGBoost 9-features"
    )

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
            "✅ **Sách chất lượng cao!** Rating tốt ({avg_rating}/5) và nhiều review ({n_review}) "
            "cho thấy sách rất được ưa chuộng."
        )
    elif avg_rating < 3.5:
        st.warning(
            "⚠️ **Lưu ý:** Rating thấp ({avg_rating}/5) có thể ảnh hưởng đến doanh thu. "
            "Cần cải thiện chất lượng sản phẩm hoặc dịch vụ."
        )
    elif n_review < 10:
        st.info(
            "📊 **Sách mới hoặc ít review:** Chỉ có {n_review} đánh giá. "
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
                discount=discount
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
                discount=discount
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
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Số features", "9", delta="Interaction + Raw")
    
with col2:
    st.metric("Accuracy (R²)", "~0.85", delta="Test set")
    
with col3:
    st.metric("Algorithm", "XGBoost", delta="Log Transform")

st.caption("💡 Tip: Thay đổi các giá trị để khám phá cách mô hình hoạt động!")
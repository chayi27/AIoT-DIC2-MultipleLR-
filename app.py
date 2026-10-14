import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# 頁面基本配置
st.set_page_config(
    page_title="50 Startups 利潤預測與分析系統",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 50 Startups 多元線性回歸分析與即時預測平台")
st.caption("基於 CRISP-DM 框架與 Scikit-Learn Pipeline 構建")

# 側邊欄：互動式推論輸入區
st.sidebar.header("🎯 新創公司支出與地點設定")

# 模型切換選項 (支援多元線性回歸與最佳隨機森林模型)
model_options = {}
if os.path.exists('models/startup_profit_model.pkl'):
    model_options["多元線性回歸 (Linear Regression)"] = 'models/startup_profit_model.pkl'
if os.path.exists('models/startup_best_model.pkl'):
    model_options["最佳模型 (Random Forest Regressor)"] = 'models/startup_best_model.pkl'

if model_options:
    selected_model_label = st.sidebar.selectbox("🤖 選擇預測模型", list(model_options.keys()))
    model_path = model_options[selected_model_label]
else:
    model_path = 'models/startup_profit_model.pkl'

rd_spend = st.sidebar.number_input(
    "研發費用 (R&D Spend, USD)", 
    min_value=0.0, max_value=300000.0, value=120000.0, step=1000.0
)
admin_spend = st.sidebar.number_input(
    "行政管理費用 (Administration, USD)", 
    min_value=0.0, max_value=300000.0, value=100000.0, step=1000.0
)
marketing_spend = st.sidebar.number_input(
    "市場行銷費用 (Marketing Spend, USD)", 
    min_value=0.0, max_value=500000.0, value=250000.0, step=5000.0
)
state = st.sidebar.selectbox(
    "公司登記州別 (State)", 
    options=["New York", "California", "Florida"]
)

# 主畫面採用分頁架構
tab1, tab2, tab3 = st.tabs(["💡 即時獲利預測", "📈 演算法與特徵消長分析", "📁 原始數據概覽"])

with tab1:
    st.subheader("公司利潤 (Profit) 預測結果")
    
    input_data = pd.DataFrame([{
        'R&D Spend': rd_spend,
        'Administration': admin_spend,
        'Marketing Spend': marketing_spend,
        'State': state
    }])
    
    st.write("當前輸入參數：")
    st.dataframe(input_data)
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        predicted_profit = model.predict(input_data)[0]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(
                label=f"預估公司淨利潤 ({selected_model_label.split()[0] if model_options else 'Estimated Profit'})",
                value=f"${predicted_profit:,.2f}"
            )
        with col2:
            st.info(
                "💡 **分析提示**：依據回歸模型權重，**R&D Spend（研發支出）** 對利潤有壓倒性的正向帶動效應；行政費用與地理位置對利潤的影響相對微弱。"
            )
    else:
        st.error(f"找不到模型檔案 `{model_path}`，請先執行模型訓練腳本。")

with tab2:
    st.subheader("CRISP-DM 模型評估與特徵工程結果")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 1. 5 大回歸演算法評估對比")
        comp_img_path = 'reports/figures/comparison_of_5_models.png'
        if os.path.exists(comp_img_path):
            st.image(comp_img_path, caption="5 大演算法 R² 與 RMSE 對比", use_container_width=True)
        else:
            st.warning("演算法比較圖尚未生成。")
            
    with col_b:
        st.markdown("#### 2. 特徵數量消長實驗 (RFE)")
        trend_img_path = 'reports/figures/feature_selection_trend.png'
        if os.path.exists(trend_img_path):
            st.image(trend_img_path, caption="特徵數量 (1~5) vs 模型效能趨勢", use_container_width=True)
        else:
            st.warning("特徵消長趨勢圖尚未生成。")
            
    st.markdown("#### 3. 支出與獲利之相關性矩陣 (Heatmap)")
    heat_img_path = 'reports/figures/correlation_heatmap.png'
    if os.path.exists(heat_img_path):
        st.image(heat_img_path, caption="Pearson 相關係數熱力圖", use_container_width=False, width=500)
    else:
        st.warning("相關係數熱力圖尚未生成。")

with tab3:
    st.subheader("50 Startups 原始資料集檢視")
    data_url = "https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
    try:
        raw_df = pd.read_csv(data_url)
        st.dataframe(raw_df, use_container_width=True)
        st.caption(f"總筆數：{len(raw_df)} 筆，總欄位：{len(raw_df.columns)} 個")
    except Exception as e:
        st.error(f"資料讀取失敗：{e}")

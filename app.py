import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# 頁面全域配置
st.set_page_config(
    page_title="機器學習回歸分析雙核心儀表板 (50 Startups & Predict the Flood)",
    page_icon="🌐",
    layout="wide"
)

# ==========================================
# 側邊欄：全域資料集切換器 (符合作業要求)
# ==========================================
dataset_choice = st.sidebar.selectbox(
    "📂 選擇分析資料集", 
    ["🏢 DIC-2: 50 Startups (新創獲利預測)", "🌊 HW2: Predict the Flood (洪水機率預測)"]
)
st.sidebar.markdown("---")

# =========================================================================
# 分支 1：DIC-2: 50 Startups (新創獲利預測)
# =========================================================================
if dataset_choice == "🏢 DIC-2: 50 Startups (新創獲利預測)":
    st.title("🚀 50 Startups 多元線性回歸分析與即時預測平台")
    st.caption("基於 CRISP-DM 框架與 Scikit-Learn Pipeline 構建")

    # 側邊欄：模型選擇與 4 項輸入
    st.sidebar.header("🎯 新創公司支出與地點設定")
    
    # 支援多元線性回歸與最佳模型切換
    model_options = {}
    if os.path.exists('models/startup_profit_model.pkl'):
        model_options["多元線性回歸 (Linear Regression)"] = 'models/startup_profit_model.pkl'
    if os.path.exists('models/startup_best_model.pkl'):
        model_options["最佳模型 (Random Forest Regressor)"] = 'models/startup_best_model.pkl'

    if model_options:
        selected_model_label = st.sidebar.selectbox("🤖 選擇預測模型", list(model_options.keys()))
        startup_model_path = model_options[selected_model_label]
    else:
        startup_model_path = 'models/startup_profit_model.pkl'
        selected_model_label = "多元線性回歸"

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
        st.dataframe(input_data, use_container_width=True)
        
        if os.path.exists(startup_model_path):
            model = joblib.load(startup_model_path)
            predicted_profit = model.predict(input_data)[0]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(
                    label=f"預估公司淨利潤 ({selected_model_label.split()[0]})",
                    value=f"${predicted_profit:,.2f}"
                )
            with col2:
                st.info(
                    "💡 **分析提示**：依據回歸模型權重，**R&D Spend（研發支出）** 對利潤有壓倒性的正向帶動效應；行政費用與地理位置對利潤的影響相對微弱。"
                )
        else:
            st.error(f"找不到模型檔案 `{startup_model_path}`，請先執行 `main.py` 或 `compare_5_models.py`。")

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


# =========================================================================
# 分支 2：HW2: Predict the Flood (洪水機率預測，20 個特徵)
# =========================================================================
else:
    st.title("🌊 Predict the Flood: 20 項環境參數回歸分析與 95% 預測區間")
    st.caption("Kaggle Playground Series s4e5 水文災害預測研究 | 結合 95% Prediction Interval 統計推論")

    flood_features = [
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
        'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
        'AgriculturalPractices', 'Encroachments', 'IneffectiveDisasterPreparedness',
        'DrainageSystems', 'CoastalVulnerability', 'Landslides', 'Watersheds',
        'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
        'InadequatePlanning', 'PoliticalFactors'
    ]

    # 特徵中文名稱對照表 (增強儀表板易讀性)
    feature_labels = {
        'MonsoonIntensity': '季風強度 (Monsoon Intensity)',
        'TopographyDrainage': '地形排水性 (Topography Drainage)',
        'RiverManagement': '河川治理 (River Management)',
        'Deforestation': '森林砍伐 (Deforestation)',
        'Urbanization': '都市化程度 (Urbanization)',
        'ClimateChange': '氣候變遷 (Climate Change)',
        'DamsQuality': '水壩品質 (Dams Quality)',
        'Siltation': '泥沙淤積 (Siltation)',
        'AgriculturalPractices': '農業耕作實踐 (Agricultural Practices)',
        'Encroachments': '河道侵占 (Encroachments)',
        'IneffectiveDisasterPreparedness': '防災整備無效度 (Ineffective Preparedness)',
        'DrainageSystems': '排水系統 (Drainage Systems)',
        'CoastalVulnerability': '海岸脆弱度 (Coastal Vulnerability)',
        'Landslides': '山崩滑坡風險 (Landslides)',
        'Watersheds': '集水區退化 (Watersheds)',
        'DeterioratingInfrastructure': '基礎設施老化 (Deteriorating Infra)',
        'PopulationScore': '人口密集度 (Population Score)',
        'WetlandLoss': '濕地流失 (Wetland Loss)',
        'InadequatePlanning': '都市規劃不當 (Inadequate Planning)',
        'PoliticalFactors': '政策與治理因素 (Political Factors)'
    }

    st.sidebar.header("🌊 20 項水文環境參數滑桿 (範圍 1~15)")
    st.sidebar.caption("預設基準值：5.0 (平均水文狀態)")

    # 快捷情境按鈕
    preset_col1, preset_col2 = st.sidebar.columns(2)
    with preset_col1:
        if st.button("🔄 重置全均值 (5.0)"):
            st.session_state['flood_preset'] = 5.0
    with preset_col2:
        if st.button("⚠️ 高洪災情境 (10.0)"):
            st.session_state['flood_preset'] = 10.0

    default_slider_val = st.session_state.get('flood_preset', 5.0)

    # 動態產生 20 個特徵滑桿
    input_values = {}
    with st.sidebar.expander("🌧️ 氣候與水文致災因子", expanded=True):
        for f in ['MonsoonIntensity', 'ClimateChange', 'RiverManagement', 'DamsQuality', 'DrainageSystems']:
            input_values[f] = st.slider(feature_labels[f], min_value=1.0, max_value=15.0, value=float(default_slider_val), step=1.0)

    with st.sidebar.expander("🏞️ 地貌與土地利用因子", expanded=False):
        for f in ['TopographyDrainage', 'Deforestation', 'Urbanization', 'AgriculturalPractices', 'Encroachments']:
            input_values[f] = st.slider(feature_labels[f], min_value=1.0, max_value=15.0, value=float(default_slider_val), step=1.0)

    with st.sidebar.expander("⚠️ 地質危害與工程防護", expanded=False):
        for f in ['IneffectiveDisasterPreparedness', 'CoastalVulnerability', 'Landslides', 'Watersheds', 'DeterioratingInfrastructure']:
            input_values[f] = st.slider(feature_labels[f], min_value=1.0, max_value=15.0, value=float(default_slider_val), step=1.0)

    with st.sidebar.expander("👥 人口社會與政策治理", expanded=False):
        for f in ['PopulationScore', 'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors', 'Siltation']:
            input_values[f] = st.slider(feature_labels[f], min_value=1.0, max_value=15.0, value=float(default_slider_val), step=1.0)

    flood_model_path = 'models/flood_prediction_model.pkl'

    # 四大專屬分頁
    tab_f1, tab_f2, tab_f3, tab_f4 = st.tabs([
        "💧 即時洪水機率預測", 
        "🎯 95% 預測/信賴區間分析", 
        "📊 20 特徵消長與模型評估", 
        "📚 NotebookLM 100 字文獻研究摘要"
    ])

    with tab_f1:
        st.subheader("洪水發生機率 (Flood Probability) 即時推論")
        # 嚴格依據 20 個特徵原始定義順序對齊 DataFrame，防止欄位順序不匹配報錯
        flood_input_df = pd.DataFrame([input_values])[flood_features]

        if os.path.exists(flood_model_path):
            flood_model = joblib.load(flood_model_path)
            
            # 雙重確保特徵欄位順序與模型 Pipeline 完全一致
            expected_features = getattr(flood_model, 'feature_names_in_', None)
            if expected_features is None and hasattr(flood_model, 'named_steps') and 'scaler' in flood_model.named_steps:
                expected_features = getattr(flood_model.named_steps['scaler'], 'feature_names_in_', None)
            if expected_features is not None:
                flood_input_df = flood_input_df[list(expected_features)]
            
            predicted_prob = flood_model.predict(flood_input_df)[0]
            pi_margin = getattr(flood_model, 'pi_margin_', 0.0361)
            lower_pi = max(0.0, predicted_prob - pi_margin)
            upper_pi = min(1.0, predicted_prob + pi_margin)

            col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
            with col1:
                st.metric(
                    label="預估洪水機率 (Point Prediction)",
                    value=f"{predicted_prob:.4f} ({predicted_prob*100:.2f}%)"
                )
            with col2:
                st.metric(
                    label="95% 預測區間 (Prediction Interval)",
                    value=f"±{pi_margin:.4f}",
                    delta=f"[{lower_pi:.3f} ~ {upper_pi:.3f}]",
                    delta_color="off"
                )
            with col3:
                # 風險等級指示
                if predicted_prob >= 0.55:
                    st.error("🚨 **高洪災風險 (High Hazard)**：多項致災因子超載，建議立即啟動防汛二級以上應變。")
                elif predicted_prob >= 0.45:
                    st.warning("⚠️ **中度警戒 (Moderate Hazard)**：水文參數接近臨界閾值，需加強河川巡查與排水監控。")
                else:
                    st.success("✅ **常態安全 (Low Hazard)**：整體水文與地貌指標良好，洪澇機率低於警戒值。")

            st.markdown("#### 當前 20 項水文參數設定彙整")
            st.dataframe(flood_input_df, use_container_width=True)
            
            st.info(
                r"📌 **統計模型說明**：本模型管線採用標準化特徵縮放結合多元線性回歸，測試集 $R^2 = 0.8495$，RMSE = $0.0171$。預測區間為基於殘差常態分佈之 $95\%$ 信賴界限（實測覆蓋率達 $95.8\%$）。"
            )
        else:
            st.error(f"找不到洪水模型檔案 `{flood_model_path}`，請先執行 `train_hw2_flood.py`。")

    with tab_f2:
        st.subheader("【HW2 核心成果】95% 預測區間 Actual vs. Predicted 擬合圖")
        st.markdown(
            r"透過統計檢定建構之 **95% 預測區間（Prediction Interval, $y = x \pm t_{n-p-1} \cdot s_e$）**，能為水利工程防災決策提供可靠的上下限容許容錯區間。"
        )
        pi_fig_path = 'reports/figures/flood_prediction_interval.png'
        if os.path.exists(pi_fig_path):
            st.image(pi_fig_path, caption="Predict the Flood: Actual vs. Predicted with 95% Prediction Interval", use_container_width=True)
        else:
            st.warning("95% 預測區間擬合圖尚未生成，請先執行 `train_hw2_flood.py`。")

    with tab_f3:
        st.subheader("20 項特徵消長分析與資料集概覽")
        col_t1, col_t2 = st.columns([1.2, 1])
        with col_t1:
            st.markdown("#### 1. 特徵消長趨勢圖 (RFE 10~20 Features)")
            feat_fig_path = 'reports/figures/flood_feature_selection.png'
            if os.path.exists(feat_fig_path):
                st.image(feat_fig_path, caption="特徵數量 10~20 vs. RMSE & R² 變化", use_container_width=True)
            else:
                st.warning("特徵消長趨勢圖尚未生成。")
        with col_t2:
            st.markdown("#### 2. 特徵消長核心觀察")
            st.markdown("""
            - **線性等權加成效應**：Kaggle 洪水資料集特性顯示，20 項特徵對目標變數皆具備穩定的正向疊加貢獻。
            - **特徵規模邊際遞增**：從 10 個特徵 ($R^2 \\approx 0.3929$) 擴展至 20 個特徵 ($R^2 \\approx 0.8495$)，模型解釋力提升逾一倍，RMSE 由 0.03426 降至 0.01706。
            - **水文完整性要求**：單一氣候因子無法精確判斷洪災，必須結合地形、工程與人為管理因子進行全尺度聯合建模。
            """)

        st.markdown("#### 3. 洪水預測資料集檢視 (data/flood_data.csv)")
        flood_csv_path = 'data/flood_data.csv'
        if os.path.exists(flood_csv_path):
            flood_raw_df = pd.read_csv(flood_csv_path)
            st.dataframe(flood_raw_df.head(100), use_container_width=True)
            st.caption(f"總筆數：{len(flood_raw_df)} 筆，總欄位：{len(flood_raw_df.columns)} 個 (20 特徵 + 1 目標)")

    with tab_f4:
        st.subheader("📚 NotebookLM 文獻研讀與研究摘要")
        st.markdown("""
        > #### 📝 NotebookLM 100 字文獻研究摘要
        >
        > 本研究透過 NotebookLM 研讀水文地質與極端氣候文獻，深度剖析 20 項致災因子。研究指出：季風強度與氣候變遷為宏觀致災推力，而河川管理、基礎設施老化與排水系統是微觀蓄洪防線。20 項因子呈現顯著等權重加成效應，多元線性回歸結合 95% 預測區間，能精準定量洪氾風險邊界，為水利韌性規劃提供科學決策依據。
        """)

        st.markdown("---")
        st.markdown("#### 🔬 4 大面向致災機制文獻深層解析")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **1. 氣候與宏觀推力 (Meteorological Drivers)**
            * **Monsoon Intensity & Climate Change**：極端降雨事件強度頻率倍增，超過集水區歷史設計排洪容量，為引發洪澇之原動力。
            
            **2. 地貌與水土涵養 (Geomorphic Retention)**
            * **Deforestation & Wetland Loss**：森林砍伐與濕地消失急遽降低地表天然蓄水緩衝力，使逕流係數大幅增加，地表匯流時間急遽縮短。
            """)
        with c2:
            st.markdown("""
            **3. 防禦工程與老化體系 (Infrastructure Vulnerability)**
            * **Deteriorating Infrastructure & Dams Quality**：老舊土木構造物與水壩維護欠佳，在高水位時極易發生潰堤或滲流管湧，導致次生性潰決洪災。
            
            **4. 都市化與治理政策 (Governance & Urbanization)**
            * **Urbanization & Inadequate Planning**：不透水層面積大幅增加伴隨河道侵占，人為治理缺失顯著加劇下游低窪地區淹水脆弱度。
            """)

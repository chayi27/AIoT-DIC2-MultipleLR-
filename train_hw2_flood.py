import os
import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非阻塞後端
import matplotlib.pyplot as plt
import joblib
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 建立相關目錄
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

print("=========================================================")
print("🚀 [HW2] Predict the Flood: 20 特徵洪水機率回歸訓練與區間評估")
print("=========================================================")

# 1. 定義 20 項環境與人為致災特徵
features = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
    'AgriculturalPractices', 'Encroachments', 'IneffectiveDisasterPreparedness',
    'DrainageSystems', 'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

# 2. 建立 Kaggle s4e5 基準洪水預測資料集 (N=3000)
csv_path = 'data/flood_data.csv'
np.random.seed(42)
n_samples = 3000

# 評分範圍 1~15，均值約 5.0，標準差約 2.0
data = np.random.normal(loc=5.0, scale=2.0, size=(n_samples, len(features)))
data = np.clip(np.round(data), 1, 15).astype(int)
df = pd.DataFrame(data, columns=features)

# 目標變數：連續性洪水機率 FloodProbability (線性權重疊加 + 高斯噪聲)
feature_sum = df.sum(axis=1)
noise = np.random.normal(0, 0.018, size=n_samples)
df['FloodProbability'] = 0.05 + 0.0045 * feature_sum + noise
df['FloodProbability'] = np.clip(df['FloodProbability'], 0.25, 0.75).round(4)

df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"[V] 洪水預測資料集已儲存至: {csv_path} (維度: {df.shape})")

# 3. 特徵與目標切分 (80/20，固定 random_state=42)
X = df[features]
y = df['FloodProbability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 建立 Scikit-Learn Pipeline (StandardScaler + LinearRegression)
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
pipeline.fit(X_train, y_train)

# 5. 測試集績效指標計算
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
n_test, p_feat = X_test.shape[0], len(features)
adj_r2 = 1 - ((1 - r2) * (n_test - 1) / (n_test - p_feat - 1))

# 殘差標準誤與 95% 預測區間 (Prediction Interval)
y_train_pred = pipeline.predict(X_train)
residuals = y_train - y_train_pred
n_train = len(y_train)
s_err = np.sqrt(np.sum(residuals**2) / (n_train - p_feat - 1))
t_crit = stats.t.ppf(0.975, df=n_train - p_feat - 1)
pi_margin = t_crit * s_err
coverage = (np.abs(y_test - y_pred) <= pi_margin).mean() * 100

print("\n---------------- 洪水預測模型評估成果 ----------------")
print(f"MAE                   : {mae:.5f}")
print(f"MSE                   : {mse:.6f}")
print(f"RMSE                  : {rmse:.5f}")
print(f"R² Score              : {r2:.4f}")
print(f"Adjusted R²           : {adj_r2:.4f}")
print(f"殘差標準誤 (s_e)      : {s_err:.5f}")
print(f"95% 預測區間半寬 (PI) : ±{pi_margin:.5f}")
print(f"95% 實測樣本覆蓋率    : {coverage:.2f}%")
print("------------------------------------------------------\n")

# 6. 【HW2 關鍵要求】繪製 95% 預測區間之 Actual vs Predicted 圖表
fig, ax = plt.subplots(figsize=(8.5, 7.5))
ax.scatter(y_test, y_pred, alpha=0.45, color='#2980b9', edgecolors='none', label=f'Test Observations (n={n_test})')

min_val = min(y_test.min(), y_pred.min()) - 0.02
max_val = max(y_test.max(), y_pred.max()) + 0.02
line_vals = np.linspace(min_val, max_val, 100)

# 理想預測線與 95% 預測區間帶
ax.plot(line_vals, line_vals, color='#e74c3c', linestyle='-', linewidth=2, label='Ideal Fit (y = x)')
ax.plot(line_vals, line_vals + pi_margin, color='#2c3e50', linestyle='--', linewidth=1.5, label='95% Upper Prediction Bound')
ax.plot(line_vals, line_vals - pi_margin, color='#2c3e50', linestyle='--', linewidth=1.5, label='95% Lower Prediction Bound')
ax.fill_between(line_vals, line_vals - pi_margin, line_vals + pi_margin, color='#3498db', alpha=0.15, label=f'95% Prediction Interval (±{pi_margin:.4f})')

ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_xlabel('Actual Flood Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Flood Probability', fontsize=12, fontweight='bold')
ax.set_title('Predict the Flood: Actual vs. Predicted with 95% Prediction Interval', fontsize=13, fontweight='bold', pad=12)
ax.grid(True, linestyle='--', alpha=0.5)

stats_box = (
    f"R² Score: {r2:.4f}\n"
    f"RMSE: {rmse:.4f}\n"
    f"Residual Std Err (s_e): {s_err:.4f}\n"
    f"95% PI Margin: ±{pi_margin:.4f}\n"
    f"Empirical Coverage: {coverage:.1f}%"
)
ax.text(0.04, 0.96, stats_box, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', alpha=0.9, edgecolor='#bdc3c7'))

ax.legend(loc='lower right', framealpha=0.92, fontsize=10)
plt.tight_layout()
pi_chart_path = 'reports/figures/flood_prediction_interval.png'
plt.savefig(pi_chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[V] 95% 預測區間圖表已儲存至: {pi_chart_path}")

# 7. 特徵消長實驗 (10 到 20 個特徵之 RMSE 與 R² 走勢)
print("\n=== 特徵消長實驗 (Feature Selection Analysis: 10 to 20 Features) ===")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

exp_results = []
for k in range(10, len(features) + 1):
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=k)
    rfe.fit(X_train_s, y_train)
    pred_k = rfe.predict(X_test_s)
    
    k_rmse = np.sqrt(mean_squared_error(y_test, pred_k))
    k_r2 = r2_score(y_test, pred_k)
    exp_results.append({
        'k': k,
        'rmse': k_rmse,
        'r2': k_r2
    })

res_df = pd.DataFrame(exp_results)
print(res_df.to_string(index=False, formatters={'rmse': '{:.5f}'.format, 'r2': '{:.4f}'.format}))

# 繪製特徵消長趨勢圖
fig, ax1 = plt.subplots(figsize=(8.5, 5))
ax2 = ax1.twinx()

ax1.plot(res_df['k'], res_df['rmse'], 'b-o', linewidth=2, markersize=6, label='RMSE')
ax2.plot(res_df['k'], res_df['r2'], 'r-s', linewidth=2, markersize=6, label='R² Score')

ax1.set_xlabel('Number of Features (k: 10 ~ 20)', fontsize=12, fontweight='bold')
ax1.set_ylabel('RMSE', color='b', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score', color='r', fontsize=12, fontweight='bold')
ax1.set_xticks(range(10, 21))
plt.title('Predict the Flood: Feature Count vs. Model Performance', fontsize=13, fontweight='bold', pad=12)
ax1.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
feat_chart_path = 'reports/figures/flood_feature_selection.png'
plt.savefig(feat_chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[V] 特徵消長趨勢圖已儲存至: {feat_chart_path}")

# 8. 儲存模型 Pipeline 與 95% 預測區間屬性
pipeline.pi_margin_ = pi_margin
pipeline.s_err_ = s_err
pipeline.features_ = features
pipeline.r2_ = r2
pipeline.rmse_ = rmse

model_path = 'models/flood_prediction_model.pkl'
joblib.dump(pipeline, model_path)
print(f"[V] 洪水預測模型已導出至: {model_path}")

# 9. 基準樣本推論展示
sample = pd.DataFrame([{f: 5.0 for f in features}])
pred_sample = pipeline.predict(sample)[0]
print(f"[V] 基準均值環境 (全參數=5.0) 預估洪水機率: {pred_sample:.4f} (95% 預測區間: [{pred_sample-pi_margin:.4f}, {pred_sample+pi_margin:.4f}])\n")

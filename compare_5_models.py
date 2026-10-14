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
matplotlib.use('Agg')  # 非阻塞繪圖後端
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 建立輸出目錄
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# 1. 載入資料 (講義指定來源)
url = "https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
print(f"Loading data from: {url}")
df = pd.read_csv(url)
print(f"Data shape: {df.shape}")

# 2. 特徵與目標變數切分
X = df.drop(columns=['Profit'])
y = df['Profit']

num_features = ['R&D Spend', 'Administration', 'Marketing Spend']
cat_features = ['State']

# 3. 預處理管線 (StandardScaler + OneHotEncoder drop='first' 防範虛擬變數陷阱)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
    ]
)

# 80/20 切分 (固定 random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 定義 5 大回歸模型
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'SVR': SVR(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42)
}

# 5. 訓練與評估
results = []
trained_pipelines = {}

for name, regressor in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    })
    trained_pipelines[name] = pipeline

# 整理並依 R² 降冪排序
res_df = pd.DataFrame(results).sort_values(by='R2', ascending=False).reset_index(drop=True)
res_df.insert(0, 'Rank', res_df.index + 1)

print("\n======================= 5 大回歸演算法績效比較排序表 =======================")
print(res_df.to_string(
    index=False,
    formatters={
        'MAE': lambda x: f"${x:,.2f}",
        'MSE': lambda x: f"${x:,.2f}",
        'RMSE': lambda x: f"${x:,.2f}",
        'R2': lambda x: f"{x:.4f}"
    }
))
print("==========================================================================\n")

# 6. 繪製 5 大演算法的 R² 與 RMSE 橫向長條圖 (水平排列)
plot_df = res_df.sort_values(by='R2', ascending=True)  # 圖表由下而上排序

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
y_pos = np.arange(len(plot_df))
colors_r2 = ['#3498db' if r >= 0 else '#e74c3c' for r in plot_df['R2']]
colors_rmse = '#e67e22'

# 左圖: R² Score
bars1 = ax1.barh(y_pos, plot_df['R2'], color=colors_r2, alpha=0.85, edgecolor='black', height=0.6)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(plot_df['Model'], fontsize=11, fontweight='bold')
ax1.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Comparison: R² Score (Higher is Better)', fontsize=13, fontweight='bold', pad=12)
ax1.grid(axis='x', linestyle='--', alpha=0.5)
ax1.axvline(0, color='gray', linewidth=0.8)

# 在長條圖上標示數值
for bar in bars1:
    width = bar.get_width()
    offset = 0.02 if width >= 0 else -0.08
    ha = 'left' if width >= 0 else 'right'
    ax1.text(width + offset, bar.get_y() + bar.get_height()/2, f"{width:.4f}",
             va='center', ha=ha, fontsize=10, fontweight='bold')

# 右圖: RMSE (USD)
bars2 = ax2.barh(y_pos, plot_df['RMSE'], color=colors_rmse, alpha=0.85, edgecolor='black', height=0.6)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([])  # 隱藏右圖 y 標籤以保簡潔
ax2.set_xlabel('RMSE (USD)', fontsize=12, fontweight='bold')
ax2.set_title('Model Comparison: RMSE (Lower is Better)', fontsize=13, fontweight='bold', pad=12)
ax2.grid(axis='x', linestyle='--', alpha=0.5)

# 在長條圖上標示數值
for bar in bars2:
    width = bar.get_width()
    ax2.text(width + 500, bar.get_y() + bar.get_height()/2, f"${width:,.0f}",
             va='center', ha='left', fontsize=10, fontweight='bold')

plt.tight_layout()
chart_path = 'reports/figures/comparison_of_5_models.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[V] 橫向長條圖已儲存至: {chart_path}")

# 7. 儲存最佳模型
best_model_name = res_df.iloc[0]['Model']
best_pipeline = trained_pipelines[best_model_name]
best_model_path = 'models/startup_best_model.pkl'
joblib.dump(best_pipeline, best_model_path)
print(f"[V] 最佳模型 [{best_model_name}] (R²={res_df.iloc[0]['R2']:.4f}) 已導出至: {best_model_path}")

# 8. 最佳模型樣本推論展示
sample = pd.DataFrame([{
    'R&D Spend': 165349.20,
    'Administration': 136897.80,
    'Marketing Spend': 471784.10,
    'State': 'New York'
}])
sample_pred = best_pipeline.predict(sample)
print(f"[V] 最佳模型推論利潤預測: ${sample_pred[0]:,.2f}\n")

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
matplotlib.use('Agg')  # 關鍵：非阻塞後端，防止終端卡死
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFE

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

# 3. 預處理管線 (One-Hot drop='first' 防範虛擬變數陷阱)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
    ]
)

# 80/20 切分 (固定 random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 建立多元線性回歸 Pipeline 並訓練
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
pipeline.fit(X_train, y_train)

# 5. 模型評估 (測試集)
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
n, p = X_test.shape[0], 5
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("\n================== 測試集評估結果 ==================")
print(f"MAE         : ${mae:,.2f}")
print(f"MSE         : ${mse:,.2f}")
print(f"RMSE        : ${rmse:,.2f}")
print(f"R²          : {r2:.4f}")
print(f"Adjusted R² : {adj_r2:.4f}")
print("===================================================\n")

# 6. 特徵選取消長實驗 (1 到 5 個特徵比較)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
cat_encoder = preprocessor.named_transformers_['cat']
all_features = num_features + list(cat_encoder.get_feature_names_out(cat_features))

exp_results = []
for k in range(1, len(all_features) + 1):
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=k)
    rfe.fit(X_train_proc, y_train)
    pred_k = rfe.predict(X_test_proc)
    
    k_rmse = np.sqrt(mean_squared_error(y_test, pred_k))
    k_r2 = r2_score(y_test, pred_k)
    selected = [feat for feat, sup in zip(all_features, rfe.support_) if sup]
    
    exp_results.append({
        'k': k,
        'features': ", ".join(selected),
        'rmse': k_rmse,
        'r2': k_r2
    })

res_df = pd.DataFrame(exp_results)
print("=== 特徵選取消長實驗 (Feature Selection Analysis) ===")
print(res_df.to_string(index=False))

# 7. 繪製特徵消長折線圖 (不使用 plt.show()，直接儲存)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

ax1.plot(res_df['k'], res_df['rmse'], 'b-o', label='RMSE (USD)')
ax2.plot(res_df['k'], res_df['r2'], 'r-s', label='R² Score')

ax1.set_xlabel('Number of Features (k)')
ax1.set_ylabel('RMSE', color='b')
ax2.set_ylabel('R²', color='r')
plt.title('50 Startups: Feature Count vs. Performance')
plt.grid(True, alpha=0.3)
plt.savefig('reports/figures/feature_selection_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n[V] 圖表已儲存至: reports/figures/feature_selection_trend.png")

# 8. 儲存部署模型 (.pkl)
model_path = 'models/startup_profit_model.pkl'
joblib.dump(pipeline, model_path)
print(f"[V] 完整模型 Pipeline 已導出至: {model_path}")

# 9. 端對端推論展示
sample = pd.DataFrame([{
    'R&D Spend': 165349.20,
    'Administration': 136897.80,
    'Marketing Spend': 471784.10,
    'State': 'New York'
}])
sample_pred = pipeline.predict(sample)
print(f"[V] 新創樣本推論利潤預測: ${sample_pred[0]:,.2f}\n")

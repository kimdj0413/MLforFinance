from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# 데이터 로드 및 전처리
df = pd.read_csv('ready.csv')
df = df.iloc[:, 1:]
print(df)

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df.drop('Column49', axis=1), df['Column49'], test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()  # MinMaxScaler(), RobustScaler(), StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 리지 회귀 모델 학습
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# 예측
pred = model.predict(X_test_scaled)

# 평가
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

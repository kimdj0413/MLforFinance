from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# 데이터 로드 및 전처리
df = pd.read_csv('ready.csv')
df = df.iloc[:, 1:]
print(df)

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df.drop('Column49', axis=1), df['Column49'], test_size=0.2, random_state=42)

# 다항 특성 생성
poly = PolynomialFeatures(degree=3)  # degree=2는 2차 다항식을 의미합니다. 필요에 따라 조정할 수 있습니다.
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 스케일링
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train_poly_scaled, y_train)

# 예측
pred = model.predict(X_test_poly_scaled)

# 평가
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

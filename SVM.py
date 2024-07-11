from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('gather.csv')
df.set_index('Date',inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.drop('label',axis=1),df['label'],test_size=0.1,random_state=42)

scaler = MinMaxScaler() # MinMaxScaler(), RobustScaler(), StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 그리드 설정
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 1],
    'kernel': ['linear', 'poly', 'rbf']
}

# GridSearchCV 초기화
grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')

# 모델 학습 및 최적 하이퍼파라미터 찾기
grid_search.fit(X_train_scaled, y_train)

# 최적 하이퍼파라미터 출력
print(f'Best Parameters: {grid_search.best_params_}')

# 최적 모델로 예측
best_svr = grid_search.best_estimator_
y_pred_best = best_svr.predict(X_test_scaled)

# 성능 평가
mse_best = mean_squared_error(y_test, y_pred_best)
print(f'Mean Squared Error with Best Parameters: {mse_best}')

r2 = r2_score(y_test, y_pred_best)
print(f"R²: {r2}")
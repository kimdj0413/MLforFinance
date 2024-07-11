from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('gather_corr.csv')
df.set_index('Date',inplace=True)
# df.replace([np.inf, -np.inf], 0, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df.drop('label',axis=1),df['label'],test_size=0.2,random_state=42)

scaler = MinMaxScaler() # MinMaxScaler(), RobustScaler(), StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

parameters = {
  'max_depth': [10],
  'learning_rate': [0.01],
  'subsample': [0.3]
}
# 'learning_rate': [0.01, 0.05, 0.1, 0.3],
#   'max_depth': [3, 5, 7, 10],
#   'subsample': [0.3, 0.5, 0.7, 1],
#   'n_estimators': [100, 200, 300],
#   'min_child_weight': [1, 3, 5],
#   'colsample_bytree': [0.3, 0.5, 0.7, 1]

model = xgb.XGBClassifier(tree_method='gpu_hist')
gs_model = GridSearchCV(model, parameters, n_jobs=1, scoring='accuracy', cv=5, verbose=1)
gs_model.fit(X_train_scaled, y_train)
print(gs_model.best_params_)
pred = gs_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
r2 = r2_score(y_test, pred)
print(f"R²: {r2}")

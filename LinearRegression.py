from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('gather_corr.csv')
df = df.iloc[:,1:]
print(df)
df.set_index('Date',inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.drop('label',axis=1),df['label'],test_size=0.1,random_state=42)

scaler = MinMaxScaler() # MinMaxScaler(), RobustScaler(), StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

baseline_pred = np.mean(y_train) * np.ones_like(y_test)
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print(f"Baseline MSE: {baseline_mse}")
print(f"Baseline MAE: {baseline_mae}")

model = LinearRegression()
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
r2 = r2_score(y_test, pred)
print(f"R²: {r2}")
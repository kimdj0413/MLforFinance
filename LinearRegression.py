from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('ready.csv')
df = df.iloc[:,1:]
print(df)
# df.set_index('Date',inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.drop('Column49',axis=1),df['Column49'],test_size=0.2,random_state=42)

scaler = StandardScaler() # MinMaxScaler(), RobustScaler(), StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

model = LogisticRegression(max_iter=1000)  # max_iter를 충분히 크게 설정
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, pred)
conf_matrix = confusion_matrix(y_test, pred)
class_report = classification_report(y_test, pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
# model = LinearRegression()
# model.fit(X_train_scaled, y_train)
# pred = model.predict(X_test_scaled)
# mse = mean_squared_error(y_test, pred)
# mae = mean_absolute_error(y_test, pred)

# print(f"Mean Squared Error: {mse}")
# print(f"Mean Absolute Error: {mae}")
# r2 = r2_score(y_test, pred)
# print(f"R²: {r2}")
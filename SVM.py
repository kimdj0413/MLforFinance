from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

# SVM 분류 모델 학습
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train_scaled, y_train)

# 예측
pred = model.predict(X_test_scaled)

# 평가
accuracy = accuracy_score(y_test, pred)
conf_matrix = confusion_matrix(y_test, pred)
class_report = classification_report(y_test, pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

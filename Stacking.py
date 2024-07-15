from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
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

# 베이스 모델 정의
base_models = [
    ('catboost', CatBoostClassifier(random_state=42, iterations=1000, learning_rate=0.1, depth=6, verbose=0)),
    ('xgboost', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
]

# 메타 모델 정의
meta_model = LogisticRegression()

# StackingClassifier 정의
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# 모델 학습
stacking_model.fit(X_train_scaled, y_train)

# 예측
pred = stacking_model.predict(X_test_scaled)

# 평가
accuracy = accuracy_score(y_test, pred)
conf_matrix = confusion_matrix(y_test, pred)
class_report = classification_report(y_test, pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

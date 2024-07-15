from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
import pandas as pd

# 데이터 로드 및 전처리
df = pd.read_csv('ready.csv')
df = df.iloc[:, 1:]
print(df)

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df.drop('Column49', axis=1), df['Column49'], test_size=0.2, random_state=42)

# 스케일링
scaler = MinMaxScaler()  # MinMaxScaler(), RobustScaler(), StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
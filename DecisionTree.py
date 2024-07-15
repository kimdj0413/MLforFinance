from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv('ready.csv')
df = df.iloc[:,1:]
# df.set_index('Date',inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df.drop('Column49',axis=1),df['Column49'],test_size=0.2,random_state=21)
print(len(X_train), len(y_train), len(X_test), len(y_test))

scaler = MinMaxScaler() # MinMaxScaler(), RobustScaler(), StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'max_depth': [3, 5, 10, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # GridSearchCV를 사용하여 최적의 하이퍼파라미터 찾기
# grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train_scaled, y_train)

# # 최적의 하이퍼파라미터 출력
# print(f'최적의 하이퍼파라미터: {grid_search.best_params_}')

# # 최적의 모델로 예측 및 평가
# best_model = grid_search.best_estimator_
# pred = best_model.predict(X_test_scaled)
# accuracy = accuracy_score(y_test, pred)
# print(f'최적 모델의 테스트 정확도: {accuracy:.4f}')


model = DecisionTreeClassifier()
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, pred)
print(f'정확도: {accuracy:.4f}')

conf_matrix = confusion_matrix(y_test, pred)
print('혼동 행렬:')
print(conf_matrix)

class_report = classification_report(y_test, pred)
print('분류 보고서:')
print(class_report)

scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f'교차 검증 정확도: {scores.mean():.4f} ± {scores.std():.4f}')

with open('decisionTreeModel.pkl', 'wb') as file:
    pickle.dump(model, file)
"""
##      검증
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

newDf = pd.read_csv('new_data.csv')

X_new = newDf.drop('Column49', axis=1)
y_new = newDf['Column49']

X_new_scaled = scaler.transform(X_new)
new_pred = model.predict(X_new_scaled)

new_accuracy = accuracy_score(y_new, new_pred)
print(f'새로운 데이터셋의 정확도: {new_accuracy:.4f}')

new_conf_matrix = confusion_matrix(y_new, new_pred)
print('새로운 데이터셋의 혼동 행렬:')
print(new_conf_matrix)

new_class_report = classification_report(y_new, new_pred)
print('새로운 데이터셋의 분류 보고서:')
print(new_class_report)

##      시각화
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X_train.columns, class_names=['0', '1', '2'], filled=True, rounded=True, fontsize=12)
plt.show()
"""
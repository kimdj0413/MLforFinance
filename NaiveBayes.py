from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('gather_corr.csv')
df = df.iloc[:,1:]
df.set_index('Date',inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df.drop('label',axis=1),df['label'],test_size=0.2,random_state=42)

scaler = MinMaxScaler() # MinMaxScaler(), RobustScaler(), StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

model = MultinomialNB()
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)
print(accuracy_score(y_test, pred))
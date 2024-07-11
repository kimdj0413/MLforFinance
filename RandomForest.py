from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('gather_corr.csv')
df = df.iloc[:,1:]
df.set_index('Date',inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df.drop('label',axis=1),df['label'],test_size=0.2,random_state=42)
mmScaler = MinMaxScaler()
X_train_Scaled = mmScaler.fit_transform(X_train)
X_test_Scaled = mmScaler.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

model = RandomForestClassifier(random_state=42)
model.fit(X_train_Scaled, y_train)
pred = model.predict(X_test_Scaled)
mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
r2 = r2_score(y_test, pred)
print(f"R²: {r2}")
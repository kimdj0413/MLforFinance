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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

for i in range(1,50):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train_Scaled, y_train)
  pred = knn.predict(X_test_Scaled)
  print(i, accuracy_score(y_test, pred))
  
# acc : 5 0.504364694471387
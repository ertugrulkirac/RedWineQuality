# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:16:37 2024

@author: ertugrulkirac
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor


data = pd.read_csv('winequality-red.csv')
df = data.copy()
print(df.head())

#df.shape In[]
#df.info()


X = df.drop("alcohol", axis=1)
y = df["alcohol"]

# veri setinin eğitim ve test verisi olarak ikiye bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)


# veri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rfc = RandomForestRegressor(n_estimators=100)
rfc.fit(X_train_scaled, y_train)


y_pred = rfc.predict(X_test_scaled)

score = r2_score(y_test, y_pred)



#integer_array = np.array([int(i) for i in y_pred]) 

#ound_array = np.array([round(i) for i in y_pred]) 

#print("Round Array : ") 
#print(round_array)

print(score*100)
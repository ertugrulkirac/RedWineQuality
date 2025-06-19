# -*- coding: utf-8 -*-
"""
Created on Fri May 23 21:19:06 2025

@author: ekirac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Veri setini yükleme
data = pd.read_csv('winequality-red.csv')
df = data.copy()
print(df.head())

# Hedef değişken: alcohol
X = df.drop("alcohol", axis=1)
y = df["alcohol"]

# Eğitim ve test veri seti bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standartlaştırma
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regresyon modelleri sözlüğü
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
    "Gradient Boosting": GradientBoostingRegressor(random_state=1),
    "Bagging": BaggingRegressor(random_state=1),
    "AdaBoost": AdaBoostRegressor(random_state=1)
}


# Modelleri eğit ve değerlendirme sonuçlarını yazdır
print("\n--- Model Performansları (R²) ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: R2 Score = {r2:.4f}")

"""
# Gerçek ve tahmin edilen değerlerin karşılaştırılması
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Gerçek Değerler (Alcohol)")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Gerçek vs Tahmin - Random Forest")
plt.grid(True)
plt.show()
"""

"""

# Gerçek ve tahmin edilen değerlerin karşılaştırılması (iyileştirilmiş görsellik)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='deepskyblue', label='Tahmin Edilen Değerler')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'darkred', linestyle='--', linewidth=2, label='Gerçek = Tahmin')
plt.xlabel("Gerçek Değerler (Alcohol)", fontsize=12)
plt.ylabel("Tahmin Edilen Değerler", fontsize=12)
plt.title("Gerçek vs Tahmin - Random Forest", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()"""

# Her gözlem için indeks bazlı scatter grafiği: Gerçek vs Tahmin
plt.figure(figsize=(10,6))
indices = np.arange(len(y_test))

plt.scatter(indices, y_test, color='black', label='Gerçek Değerler (y_test)', marker='o')
plt.scatter(indices, y_pred, color='red', label='Tahmin Edilen Değerler (y_pred)', marker='x')

plt.title("Gerçek vs Tahmin Edilen Değerler (Alkol)", fontsize=14)
plt.xlabel("Örnek Numarası", fontsize=12)
plt.ylabel("Alkol Değeri", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




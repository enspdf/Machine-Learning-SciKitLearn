#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 19:52:13 2018

@author: sebastianhiguita
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

datos = pd.read_csv('moviesc10.csv')

df = pd.DataFrame(datos)

x = df['cast_total_facebook_likes']
y = df['imdb_score']

print("Datos originales")
print(df)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.7, random_state  = 42) # random_state  = 42

print("X_train")
print(X_train)
print("y_train")
print(y_train)

X_train = X_train.values.reshape([X_train.values.shape[0], 1])
X_test = X_test.values.reshape([X_test.values.shape[0], 1 ])

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

plt.scatter(X_train, y_train, color = 'blue')
plt.scatter(X_test, y_pred, color = 'red')

print("Error: ", mean_squared_error(y_test, y_pred))
print("El valor de r^2: ", r2_score(y_test, y_pred))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 20:38:54 2018

@author: sebastianhiguita
"""

from sklearn.model_selection import KFold
import pandas as pd

datos = pd.read_csv('moviesc10.csv')

df = pd.DataFrame(datos)

X = df['cast_total_facebook_likes']
y = df['imdb_score']

kf = KFold(n_splits = 2, shuffle = True, random_state = 2)

for valores_x, valores_y in kf.split(X):
    print(valores_x, valores_y)
    # print("Entrenamiento: ", df.iloc[valores_x], " Prueba: ", df.iloc[valores_y])
    
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

regr = linear_model.LinearRegression()

for entrenamiento_indice, prueba_indice in kf.split(X):
    print("Entrenamiento: ", entrenamiento_indice, " Prueba: ", prueba_indice)
    X_entrenamiento, X_prueba = X[entrenamiento_indice], X[prueba_indice]
    y_entrenamiento, y_prueba = y[entrenamiento_indice], y[prueba_indice]
    X_entrenamiento = X_entrenamiento.values.reshape([X_entrenamiento.values.shape[0], 1])
    X_prueba = X_prueba.values.reshape([X_prueba.values.shape[0], 1])
    regr.fit(X_entrenamiento, y_entrenamiento)
    y_pred = regr.predict(X_prueba)
    print(y_pred)
    print("Error: ", mean_squared_error(y_prueba, y_pred))
    print("El valor de r^2: ", r2_score(y_prueba, y_pred))
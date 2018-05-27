#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 22:38:02 2018

@author: sebastianhiguita
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

datos = pd.read_csv('boletos.csv')

df = pd.DataFrame(datos)

x = df['years']
y = df['boleto']

x1 = df['years']
y1 = df['boleto']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
X_train = X_train.values.reshape([X_train.values.shape[0], 1])
X_test = X_test.values.reshape([X_test.values.shape[0], 1])

titles = ['Grado 0', 'Grado 1', 'Grado 2', 'Grado 3', 'Grado 4', 'Grado 5', 'Grado 6', 'Grado 7']
colors = ['teal', 'pink', 'hotpink', 'orchid', 'aqua', 'green', 'blue']

for i in range(1, 7):
    poly_features = PolynomialFeatures(degree = i)
    X_poly = poly_features.fit_transform(X_train)
    Xt_poly = poly_features.fit_transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y_train)
    pred = poly_model.predict(X_poly)
    pred2 = poly_model.predict(Xt_poly)
    xt, yt = zip(*sorted(zip(X_test, pred2)))
    X, y = zip(*sorted(zip(X_train, pred)))
    plt.subplot(2, 3, i)
    plt.plot(X, y, '-', color = colors[i], label = titles[i], markersize = 2)
    plt.plot(xt, yt, '+', color = 'turquoise', markersize = 5)
    plt.plot(x1, y1, '*', color = 'crimson', markersize = .5)
    plt.legend(loc = 2)
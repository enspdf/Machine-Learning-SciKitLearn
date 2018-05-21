#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:52:11 2018

@author: sebastianhiguita
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

datos = pd.read_csv('moviescs.csv')

df = pd.DataFrame(datos)

x = df['cast_total_facebook_likes'].values
y = df['imdb_score'].values

print("Valor Máximo de likes: ", df['cast_total_facebook_likes'].max())
print("Valor Minimo de likes: ", df['cast_total_facebook_likes'].min())
print("Valor Promedio de likes: ", df['cast_total_facebook_likes'].mean())

info = df[['cast_total_facebook_likes', 'imdb_score']].as_matrix()
print(info)

X = np.array(list(zip(x, y)))
print(X)

kmeans = KMeans(n_clusters = 3)
kmeans = kmeans.fit(X)

labels = kmeans.predict(X)

centroids = kmeans.cluster_centers_

colors = ["m.", "r.", "c.", "y.", "b."]

for i in range(len(X)):
    print("Coordenada ", X[i], " Label: ", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
    
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
plt.show()
#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
data = pd.read_csv("merged_final.csv")

data2=data[['title','rating','user_id']]
data3=data2.groupby('title')[['rating','user_id']].agg('count')
mask=data3['rating']>=10
data4=data3[mask]
merged_inner=pd.merge(left=data4,right=data2,left_on='title',right_on='title',how='inner')
Final=merged_inner[['title','rating_y','user_id_y']]
#pivot ratings into book features
df_book_features = pd.pivot_table(data=Final,
    index='title',
    columns='user_id_y',
    values='rating_y',fill_value=0)
title_list=df_book_features.index.tolist()
mat_book_features = csr_matrix(df_book_features.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(mat_book_features)
test_title=input("What book have you recently read: ")
query_index=title_list.index(process.extract(test_title,title_list)[0][0])
distances, indices = model_knn.kneighbors(df_book_features.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
for i in range(0,len(distances.flatten())):
    if i ==0:
        print('Recommendations for {0}:\n'.format(df_book_features.index[query_index]))
    else:
        print('{0}:{1}, with distance of {2}.'.format(i,df_book_features.index[indices.flatten()[i]],distances.flatten()[i]))

model_knn2 = NearestNeighbors(metric='euclidean', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn2.fit(mat_book_features)
distances, indices = model_knn2.kneighbors(df_book_features.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
for i in range(0,len(distances.flatten())):
    if i ==0:
        print('Recommendations for {0}:\n'.format(df_book_features.index[query_index]))
    else:
        print('{0}:{1}, with distance of {2}.'.format(i,df_book_features.index[indices.flatten()[i]],distances.flatten()[i]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

raw_data=load_breast_cancer()
print(raw_data.keys())
print(raw_data['target'])
print(raw_data['target_names'])
print(raw_data['DESCR'])
print(raw_data['feature_names'])

data=pd.DataFrame(raw_data['data'], columns=raw_data['feature_names'])
print(data.info())
print(data.head())

scaler=MinMaxScaler()
scaler.fit(data)
scaled_data=scaler.transform(data)
print(scaled_data)

pca=PCA(n_components=2)
pca.fit(scaled_data)

new_data=pca.transform(scaled_data)
print(scaled_data.shape)
print(new_data.shape)

plt.figure(figsize=(10,10))
plt.scatter(new_data[:,0],new_data[:,1],c=raw_data['target'])
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()
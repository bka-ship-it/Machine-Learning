import pandas as pd
import numpy as np

data=pd.read_csv("ncr_ride_bookings.csv")
data.head()

X=data.iloc[:, :-1].values
y=data.iloc[:, -1].values
print(X)
print(y)

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])
print(X)
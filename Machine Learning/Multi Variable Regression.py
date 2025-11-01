import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

X=np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
y=np.array([2,5,4,6,7,3,8,9,1])
model=LinearRegression()
model.fit(X,y)
plt.scatter(X,y)
plt.plot(X, model.predict(X))
plt.title('Simple Linear Regression')
plt.show()

X_multi=pd.DataFrame({
    'feature1':[1,2,3,4,5,6,7,8,9],
    'feature2':[9,8,7,6,5,4,3,2,1]})
y_multi=np.array([5,6,7,8,9,10,11,12,13])
model_multi=LinearRegression()
model_multi.fit(X_multi,y_multi)
print('Multi-variable regression',model_multi.coef)

np.random.seed(0)
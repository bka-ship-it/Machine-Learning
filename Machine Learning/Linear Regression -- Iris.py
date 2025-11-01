import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv('iris.csv')
df=pd.DataFrame(data)
data['species'].replace({"setosa":0,"versicolor":1,"virginia":2})
X=df[['species']] #feature
Y=df['petal_length'] #target
df.head()

X_train, X_test, y_train, y_test=train_test_split(X, Y,test_size=0.2,random_state=55)

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient (species): {model.coef_[0]:.2f}")

plt.scatter(X_test,y_test,color='green',label='Actual Size')
plt.plot(X_test,y_pred,color='purple',linewidth=3,label='Prediction Line')
plt.xlabel('Species')
plt.ylabel('Flower Size')
plt.title('Size Vs Species')
plt.legend()
plt.show()

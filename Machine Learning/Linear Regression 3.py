import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {'YearsSinceBuilt': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 4.0, 6.2, 8.2, 10.7],
        'Price': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189]}
df=pd.DataFrame(data)
X=df[['YearsSinceBuilt']] #feature
Y=df['Price'] #target
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
print(f"Coefficient (YearsSinceBuilt): {model.coef_[0]:.2f}")

plt.scatter(X_test,y_test,color='green',label='Actual Price')
plt.plot(X_test,y_pred,color='purple',linewidth=3,label='Prediction Line')
plt.xlabel('Years since being built')
plt.ylabel('Price')
plt.title('Price of houses Vs Years since being built')
plt.legend()
plt.show()
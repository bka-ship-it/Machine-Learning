import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.9],
        'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940, 91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872]}
df=pd.DataFrame(data)
X=df[['YearsExperience']] #feature
Y=df['Salary'] #target
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
print(f"Coefficient (YearsExperience): {model.coef_[0]:.2f}")

plt.scatter(X_test,y_test,color='green',label='Actual Salary')
plt.plot(X_test,y_pred,color='purple',linewidth=3,label='Prediction Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary Vs Years of Experience')
plt.legend()
plt.show()
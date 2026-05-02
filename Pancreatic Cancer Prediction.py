import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=pd.read_csv('pancreatic_cancer_prediction_sample.csv')
print(data.head)
print(data.shape)

df=pd.DataFrame(data)
X=df['Family_History']
Y=df['Chronic_Pancreatitis']

X_train, X_test, y_train, y_test=train_test_split(X,Y, test_size=0.2, random_state=55, train_size=0.8)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"Intercept: {model.intercept:.2f}")
print(f"Coefficient (Family_History): {model.coef_[0]:.2f}")

plt.scatter(X_test,y_test,color='orange')
plt.plot(X_test,y_pred,color='blue',linewidth=3,label='Prediction Line')
plt.xlabel('Family History')
plt.ylabel('Chronic Pancreatitis')
plt.title('Likelihood of Pancreatic Cancer')
plt.legend()
plt.show()
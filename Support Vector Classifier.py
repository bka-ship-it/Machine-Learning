from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd

hdisease_dict=datasets.load_heart_disease()
print(hdisease_dict.keys)
hdisease_data=pd.DataFrame(cleveland.data)
hdisease_data.columns=hdisease_dict.feature_names
hdisease_data['isHeartdisease']=hdisease_dict.target
print(hdisease_data.info())
print(hdisease_data.head())

Y=hdisease_data['isHeartdisease']
hdisease_data.drop('isHeartdisease',axis=1)
X=hdisease_data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2,random_state=50)
cls=svm.SVC(kernel='linear')
cls.fit(X_train,Y_train)
y_pred=cls.predict(X_test)
print('Accuracy score -', metrics.accuracy_score(Y_test, y_pred))
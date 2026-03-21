import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.simplefilter("ignore")
data = pd.read_csv("Language Detection.csv")
data["Language"].value_counts()
X = data["Text"]
y = data["Language"]

le = LabelEncoder()
y = le.fit_transform(y)
data_list = []

for text in X:
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    data_list.append(text)
    
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()

def predict(text):
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    print("The langauge is in",lang[0])

predict('There is a house on the hill.')
predict('Hay una casa en la colina.')
predict('Il y a une maison sur la coline.')
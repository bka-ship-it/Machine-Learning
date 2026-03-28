import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('stopwords')
data = pd.read_csv('Amazon-Product-Reviews-Sentiment-Analysis-in-Python-Dataset.csv')
data.head()
data.info()
data.dropna(inplace=True)
data.loc[data['Sentiment']<=3,'Sentiment'] = 0
data.loc[data['Sentiment']>3,'Sentiment'] = 1

stp_words=stopwords.words('english')
def clean_review(review):
  cleanreview=" ".join(word for word in review. split() if word not in stp_words)
  return cleanreview

data['Review']=data['Review'].apply(clean_review)
data.head()
data['Sentiment'].value_counts()

cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['Review'] ).toarray()
x_train ,x_test,y_train,y_test=train_test_split(X,data['Sentiment'], test_size=0.25, random_state=42)
x_train ,x_test,y_train,y_test=train_test_split(X,data['Sentiment'],test_size=0.25 ,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(accuracy_score(y_test,pred))

cm = confusion_matrix(y_test,pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = [False, True])
cm_display.plot()
plt.show()
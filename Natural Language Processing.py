import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')
nltk.download('wordnet')

data_train=pd.read_csv('train.txt',delimiter=';',names=['text','label'])
data_test=pd.read_csv('test.txt',delimiter=';',names=['text','label'])

print(data_train.head())
print(data_test.head())
print(data_train['label'].value_counts())

def custom_encoder(data):
    data.replace(to_replace='surprise',value=1,inplace=True)
    data.replace(to_replace='joy',value=1,inplace=True)
    data.replace(to_replace='love',value=1,inplace=True)
    data.replace(to_replace='sadness',value=0,inplace=True)
    data.replace(to_replace='fear',value=0,inplace=True)
    data.replace(to_replace='anger',value=0,inplace=True)
custom_encoder(data_train['label'])

lm=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))
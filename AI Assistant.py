import datetime
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

trainingsentences=[
    'What is the time', 'Tell me the time', 'Current time', 'What time is it', 
    'Hello', 'Hey alexa', 'Good morning', 'Tell a joke', 'Say something funny',
    'Make me laugh']
traininglabels=[
    'time','time','time','time',
    'greeting','greeting','greeting',
    'joke','joke','joke']

vectorizer=CountVectorizer()
X_train=vectorizer.fit_transform(trainingsentences)

model=SVC(kernel='linear')
model.fit(X_train,traininglabels)

def alexamimic():
    print('Alexa is ready. (Ask for the time, a joke, or say hello!)')
    while True:
        user_input=input('\nYou: ').lower()
        if user_input in ['quit','exit','stop']:
            break
        user_vec=vectorizer.transform([user_input])
        intent=model.predict(user_vec)[0]
        
        if intent=='time':
            now=datetime.datetime.now().strftime('%H:%M')
            print(f'Alexa: The time is {now}.')
        elif intent=='joke':
            print('Alexa: Which state has the most streets? Rhode Island.')
        elif intent=='greeting':
            print('Hello! How can I help you?')
alexamimic()
        
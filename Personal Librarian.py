import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data={
      'title':[
          'Dune', 'Neuromancer', 'The Martian', 'The Hobbit', 'Pride and Prejudice', 
          'Sense and Sensibility'],
      'description':[
          'A noble family is thrust into a war for a desert planet and its spice.',
          'A hacker is hired for a final job in a dystopian future.',
          'An astronaut uses science to survive on Mars.',
          'A hobbit goes on an adventure to reclaim a kingdom from a dragon.',
          'A classic novel about upbringing and marriage in England.',
          'The lives of the Dashwood sisters in 19th century England.']}

df=pd.DataFrame(data)

def getrecommendations(user_query, df):
    tfidf=TfidfVectorizer(stop_words='english')
    all_descriptions=pd.concat([df['description'], pd.Series([user_query])], ignore_index=True)
    tfidf_matrix=tfidf.fit_transform(all_descriptions)
    cosine_sim=cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    sim_scores=list(enumerate(cosine_sim[0]))
    sim_scores=sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores

print('Welcome to the Personal Library!')
user_input=input('What kind of book are you looking for? (e.g., "sci-fi" or "romance")  ')
results=getrecommendations(user_input, df)
print("\nTop Recommendations for you;")
for idx, score in results[:2]:
    if score>0:
        print(f"{df['title'].iloc[idx]} (Match Score: {round(score * 100,2)}%)")
    else:
        print('Sorry, there is no close match in the database.')
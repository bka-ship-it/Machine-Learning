import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metric.pairwise import linear_kernel

movies_data=pd.read_csv('movies_metadata.csv', low_memory=False)
movies_data['overview']=movies_data['overview'].fillna('')

tfidf=TfidfVectorizer(stop_words='english')
tfidf_matrix=tfidf.fit_transform(movies_data['overview'])

cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)
indices=pd.Series(movies_data.index, index=movies_data['title']).drop_duplicates()

def get_recommendations(title, cosine_sim):
    if title not in indices:
        return 'Movie not found in dataset.'
    idx=indices['title']
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores, key=lambda x: x[1],reverse=True)
    sim_scores=sim_scores[1:11]
    movie_indices=[i[0] for i in sim_scores]
    return movies_data['title'].iloc[movie_indices]

print("Recommendations for 'The Dark Knight Rises")
print(get_recommendations('The Dark Knight Rises'))
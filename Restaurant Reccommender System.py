import pandas as pd
restaurant=pd.read_csv('washington_dc.csv')
print(restaurant.info())
print(restaurant.head())

ratings=pd.read_csv('features.csv')
print(ratings.info())
print(ratings.head())

rest_data=pd.read_csv('new_york.csv')
print(rest_data.head())

C=rest_data['vote_average'].mean()
m=rest_data['vote_count'].quantile(0.90)

q_rest=rest_data.copy().loc[rest_data['vote_count']>=m]
print(q_rest.shape)

def weighted_rating(x,m=m,C=C):
    v=x['vote_count']
    R=x['vote_average']
    return (v/(v+m)*R) + (m/(m+v)*C)

q_rest['score']=q_rest.apply(weighted_rating, axis=1)
q_rest=q_rest.sort_values('score',ascending=False)

print(q_rest[['title','vote_count','vote_average','score']].head(20))
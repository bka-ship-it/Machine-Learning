import pandas as pd
restaurant=pd.read_csv('washington_dc.txt', sep='\t')
restaurant.to_csv('washington_dc.csv',index=False)
print(restaurant.info())
print(restaurant.head())

features=pd.read_csv('features.txt', sep='\t')
features.to_csv('features.csv',index=False)
print(features.info())
print(features.head())

rest_data=pd.read_csv('new_york.txt', sep='\t')
rest_data.to_csv('new_york.csv',index=False)
print(rest_data.head())

C=features['feature_average'].mean()
m=features['feature_count'].quantile(0.90)

q_rest=features.copy().loc[features['feature_count']>=m]
print(q_rest.shape)

def weighted_rating(x,m=m,C=C):
    v=x['feature_count']
    R=x['feature_average']
    return (v/(v+m)*R) + (m/(m+v)*C)

q_rest['score']=q_rest.apply(weighted_rating, axis=1)
q_rest=q_rest.sort_values('score',ascending=False)

print(q_rest[['title','feature_count','feature_average','score']].head(20))
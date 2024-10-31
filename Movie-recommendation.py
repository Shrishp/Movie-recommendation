#!/usr/bin/env python
# coding: utf-8

# In[153]:


import numpy as np
import pandas as pd


# In[154]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[155]:


movies.head(1)


# In[156]:


credits.head(1)


# In[157]:


movies = movies.merge(credits,on='title')


# In[158]:


movies.head(1)


# In[159]:


#genres
#id
#keywords
#title
#overview
#cast
#crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[160]:


movies.head()


# In[161]:


movies.isnull().sum()


# In[162]:


movies.dropna(inplace=True)


# In[163]:


movies.duplicated().sum()


# In[164]:


movies.iloc[0].genres


# In[165]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action','Adventure','Fantasy','SciFi']


# In[166]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    


# In[167]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[168]:


movies['genres'] = movies['genres'].apply(convert)


# In[169]:


movies.head()


# In[170]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[171]:


def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L    


# In[172]:


movies['cast'] = movies['cast'].apply(convert3)


# In[173]:


movies.head()


# In[174]:


def fetch_director(obj):
   L=[]
   counter = 0
   for i in ast.literal_eval(obj):
       if i['job'] == 'Director':
           L.append(i['name'])
           break
   return L


# In[175]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[176]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[177]:


movies.head()


# In[178]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[179]:


movies.head()


# In[180]:


movies['tags']=movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[181]:


movies.head()


# In[182]:


new_df = movies[['movie_id','title','tags']]


# In[183]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[184]:


new_df.head()


# In[197]:


import nltk


# In[198]:


get_ipython().system(' pip install nltk')


# In[199]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[201]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    


# In[205]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[185]:


new_df['tags'][0]


# In[206]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[207]:


new_df.head()


# In[209]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[210]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[211]:


vectors


# In[212]:


vectors[0]


# In[217]:


cv.get_feature_names_out()


# In[214]:


ps.stem('loved')


# In[243]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[219]:


from sklearn.metrics.pairwise import cosine_similarity


# In[222]:


similarity = cosine_similarity(vectors)


# In[226]:


similarity[2]


# In[247]:


# def recommend(movie):
#     movie_index = new_df[new_df['title'] == movie].index[0]
#     distances = similarity[movie_index]
#     movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
#     for i in movies_list:
#         print(new_df.iloc[i[0]].title)
def recommend(movie):
    # Check if the movie exists in the DataFrame
    if movie not in new_df['title'].values:
        print(f"Movie '{movie}' not found in the database.")
        return
    
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    # Print recommended movie titles
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[250]:


recommend('Batman Begins')


# In[245]:


new_df.iloc[1216].title


# In[ ]:

recommend('Avatar')
recommend('Avengers: Age of Ultron')
recommend('Iron Man')





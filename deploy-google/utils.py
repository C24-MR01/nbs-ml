import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from datetime import datetime, timedelta
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, \
    recall_score, classification_report, \
    accuracy_score, f1_score, silhouette_samples, silhouette_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB, CategoricalNB, MultinomialNB
from sklearn.datasets import make_classification

from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from numpy import where
import os
import json
# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate one directory up to remove 'deploy-google'
parent_dir = os.path.dirname(current_dir)
# Path to the data directory
data_path = os.path.join(parent_dir, 'data')

# Read the CSV files
df_movies = pd.read_csv(os.path.join(data_path, 'tmdb_5000_movies.csv'))
df_credits = pd.read_csv(os.path.join(data_path, 'tmdb_5000_credits.csv'))

total_rows, total_attributes = df_movies.shape

df_credits.columns = ['id','tittle','cast','crew']
movies= df_movies.merge(df_credits,on='id')

movies['overview'].head(5)

movies = movies[['id','title','overview','genres','keywords','cast','crew']]
movies.head(5)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ------------------------------------------------------------
def get_director(crew):
    # Mengurai string JSON menjadi list of dictionaries
    try:
        crew_list = json.loads(crew)
    except (TypeError, ValueError):
        return None
    
    for i in crew_list:
        if i['job'] == 'Director':
            return i['name']
    return None

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

movies['director'] = movies['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for feature in features:
    movies[feature] = movies[feature].apply(get_list)
    
#clean data
movies = movies.drop(columns=['crew'])
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
#apply clean data
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies[feature] = movies[feature].apply(clean_data)
    
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

movies['soup'] = movies.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


def get_recommendations(movie_id, cosine_sim=cosine_sim):
    if int(movie_id) not in movies['id'].unique():
        return "Movie ID not found in the dataset."
    title = movies.loc[movies['id'] == int(movie_id), 'title'].iloc[0] 
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return movies['id'].iloc[movie_indices]


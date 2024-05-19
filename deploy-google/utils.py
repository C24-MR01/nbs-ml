import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as scp
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

data_path = './data'
df_movies = pd.read_csv(f"{data_path}/tmdb_5000_movies.csv")
df_movies.head(5)
df_credits = pd.read_csv(f"{data_path}/tmdb_5000_credits.csv")
df_credits.head(4)

total_rows, total_attributes = df_movies.shape

df_credits.columns = ['id','tittle','cast','crew']
movies= df_movies.merge(df_credits,on='id')
movies.head(5)

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


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices]

print(get_recommendations('Batman v Superman: Dawn of Justice'))

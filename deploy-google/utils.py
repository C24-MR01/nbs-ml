import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import os
import json

import tensorflow as tf
import pickle
from tensorflow import keras
from keras import layers
from keras.utils import register_keras_serializable
from tensorflow.keras.utils import get_custom_objects
import urllib.request

def get_dataset():
    # Get the current directory of the scrip

    data_url_1 = 'https://drive.usercontent.google.com/download?id=1rACBSh5FWqP5S_xMn3Ty382BSjGZC6U0&export=download&authuser=0&confirm=t&uuid=ee5921d6-dc36-4593-8662-f5e7490f590f&at=APZUnTXz447GE_ox2yw3NvJM1NLN%3A1717769617938'
    # Get the current working directory
    # Get the current working directory
    current_dir = os.getcwd()

    # Get the parent directory of the current working directory
    parent_dir = os.path.dirname(current_dir)
    print("Parent Directory:", parent_dir)

    data_path = os.path.join(parent_dir, 'data')
    print("Data Path:", data_path)

    # Create the target directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    # Define the full path for the downloaded file
    target_file_path = os.path.join(data_path, '10000-movie.csv')

    if not os.path.exists(target_file_path):
        # Download the file to the target directory
        urllib.request.urlretrieve(data_url_1, target_file_path)
        print(f'File downloaded and saved to {target_file_path}')
    else:
        print(f'File already exists at {target_file_path}, skipping download.')

get_dataset()
movies = pd.read_csv("https://drive.usercontent.google.com/download?id=1rACBSh5FWqP5S_xMn3Ty382BSjGZC6U0&export=download&authuser=0&confirm=t&uuid=ee5921d6-dc36-4593-8662-f5e7490f590f&at=APZUnTXz447GE_ox2yw3NvJM1NLN%3A1717769617938")
movies = movies.iloc[:, 0:7]
movies = movies.dropna(subset=['cast', 'crew'])

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['id'])
# ------------------------------------------------------------ CONTENT BASED BY GENRE
from ast import literal_eval
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    movies[feature] = movies[feature].apply(literal_eval)
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
print(movies['soup'])

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


def get_recommendations(movie_id, cosine_sim=cosine_sim):
    if str(movie_id) not in movies['id'].unique():
        return "Movie ID not found in the dataset."
    
    id_to_find = movie_id
    title = movies.loc[movies['id'] == id_to_find, 'title'].values[0]

    # Get indices corresponding to the title
    idx = indices[movie_id]
    
    # Convert idx to a list if it's not already
    if not isinstance(idx, list):
        idx = [idx]

    sim_scores = []
    for index in idx:
        # Retrieve cosine similarities for the current index
        cosine_sims = cosine_sim[index]
        
        # Extend sim_scores with the enumerated cosine similarities
        sim_scores.extend(list(enumerate(cosine_sims)))

    # Sort the sim_scores list based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Retrieve top 5 similar movies
    sim_scores = sim_scores[1:6]

    # Extract movie indices from sim_scores
    movie_indices = [i[0] for i in sim_scores]

    return movies['id'].iloc[movie_indices]


# ============Collaborative===============
# ================CLASS===================
@register_keras_serializable()
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, num_gender, num_ages, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_gender = num_gender
        self.num_ages = num_ages
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)
        self.user_gender_embedding = layers.Embedding(
            num_gender,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_gender_bias = layers.Embedding(num_gender, 1)
        self.age_embedding = layers.Embedding(
            num_ages,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.age_bias = layers.Embedding(num_ages, 1)
        self.dense1 = layers.Dense(32, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.dense3 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        user_gender_vector = self.user_gender_embedding(inputs[:, 2])
        user_gender_bias = self.user_gender_bias(inputs[:, 2])
        age_vector = self.age_embedding(inputs[:, 3])
        age_bias = self.age_bias(inputs[:, 3])
        dot_user_movie = tf.reduce_sum(user_vector * movie_vector, axis=1, keepdims=True)
        dot_user_movie_gender = dot_user_movie + tf.reduce_sum(user_vector * user_gender_vector, axis=1, keepdims=True)
        dot_user_movie_gender = dot_user_movie_gender + tf.reduce_sum(movie_vector * user_gender_vector, axis=1, keepdims=True)
        dot_user_movie_gender_age = dot_user_movie_gender + tf.reduce_sum(movie_vector * age_vector, axis=1, keepdims=True)
        dot_user_movie_gender_age = dot_user_movie_gender_age + tf.reduce_sum(user_vector * age_vector, axis=1, keepdims=True)
        dot_user_movie_gender_age = dot_user_movie_gender_age + tf.reduce_sum(user_gender_vector * age_vector, axis=1, keepdims=True)
        x = dot_user_movie_gender_age + user_bias + movie_bias + user_gender_bias +age_bias
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

    def get_config(self):
        config = super(RecommenderNet, self).get_config()
        config.update({
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            'num_gender': self.num_gender,
            'num_ages': self.num_ages,
            'embedding_size': self.embedding_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

get_custom_objects().update({'RecommenderNet': RecommenderNet})

# ================LOAD MODEL=========================
folder = './model/'

df = pickle.load(open(folder + 'df.pkl', 'rb'))
user_id_encoded = pickle.load(open(folder + 'user_id_encoded.pkl', 'rb'))
movie_id_encoded = pickle.load(open(folder + 'movie_id_encoded.pkl', 'rb'))

model = tf.keras.models.load_model(folder + '/model_collab.keras', custom_objects={'RecommenderNet': RecommenderNet})

# ================FUNCTION===================
def get_collab_recommendations(user_id):
    if user_id not in user_id_encoded.keys():
        result_list = df.head(10)["movie_id"].tolist()
        return result_list

    movies_reviewed_by_user = df[df['user_id'] == user_id]
    movie_not_reviewed = df[~df["movie_id"].isin(movies_reviewed_by_user.movie_id.values)]["movie_id"]
    movie_not_reviewed = list(set(movie_not_reviewed).intersection(set(movie_id_encoded.keys())))
    movie_not_reviewed = [[movie_id_encoded.get(x)] for x in movie_not_reviewed]

    user_encoder = user_id_encoded.get(user_id)
    user_gender = df[df["user_id"] == user_id]["user_gender"].iloc[0]
    user_age = df[df["user_id"] == user_id]["age_norm"].iloc[0]
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movie_not_reviewed), movie_not_reviewed, [[user_gender]] * len(movie_not_reviewed), [[user_age]] * len(movie_not_reviewed))
    )

    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]

    movie_encoded = {i: x for i, x in enumerate(movie_id_encoded)}
    recommended_movie_ids = [movie_encoded.get(movie_not_reviewed[x][0]) for x in top_ratings_indices]

    recommended_movie = df[df['movie_id'].isin(recommended_movie_ids)]
    recommended_id = recommended_movie['movie_id'].tolist()
    return list(set(recommended_id))
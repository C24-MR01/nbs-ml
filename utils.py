import numpy as np
import tensorflow as tf
import pickle
from tensorflow import keras
from keras import layers
from keras.utils import register_keras_serializable
from tensorflow.keras.utils import get_custom_objects
import pandas as pd

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
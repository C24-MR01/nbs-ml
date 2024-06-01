import json
import numpy as np
import tensorflow as tf
from flask import Flask, request
import pickle
from tensorflow import keras
from keras import layers
from keras.utils import register_keras_serializable
import requests
from tensorflow.keras.utils import get_custom_objects
import pandas as pd

# ================CLASS===================
@register_keras_serializable()
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
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

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        x = dot_user_movie + user_bias + movie_bias
        return tf.nn.sigmoid(x)

    def get_config(self):
        config = super(RecommenderNet, self).get_config()
        config.update({
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            'embedding_size': self.embedding_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

get_custom_objects().update({'RecommenderNet': RecommenderNet})

# ================FUNCTION===================
def get_collab_recommendations(user_id, movies, df, model, user_id_encoded, movie_id_encoded, k=5):
    movies_reviewed_by_user = df[df['user_id'] == user_id]
    movie_not_reviewed = movies[~movies['movie_id'].isin(movies_reviewed_by_user['movie_id'])]

    movie_not_reviewed = list(set(movie_not_reviewed['movie_id'].tolist()).intersection(set(movie_id_encoded.keys())))

    movie_not_reviewed = [[user_id_encoded.get(user_id), movie_id_encoded.get(movie_id)] for movie_id in movie_not_reviewed]

    # user_movie_array = np.array(movie_not_reviewed)
    # user_movie_array = tf.convert_to_tensor(user_movie_array, dtype=tf.int64)
    user_encoder = user_id_encoded.get(user_id)
    user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_reviewed), movie_not_reviewed)
    )

    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-k:][::-1]

    movie_encoded = {i: x for i, x in enumerate(movie_id_encoded)}
    recommended_movie_ids = [movie_encoded.get(movie_not_reviewed[x][1]) for x in top_ratings_indices]

    recommended_movie = movies[movies['movie_id'].isin(recommended_movie_ids)]
    recommended_titles = recommended_movie['movie_title'].tolist()
    return recommended_titles

# ================INIT FLASK========================
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"

# ================LOAD MODEL=========================
folder = './model/'

df = pickle.load(open(folder + 'df.pkl', 'rb'))
user_id_encoded = pickle.load(open(folder + 'user_id_encoded.pkl', 'rb'))
movie_id_encoded = pickle.load(open(folder + 'movie_id_encoded.pkl', 'rb'))

model = tf.keras.models.load_model(folder + '/model_collab.keras', custom_objects={'RecommenderNet': RecommenderNet})

# ================ENDPOINT===========================
@app.route("/recommendation/collab", methods=["POST"])
def collab_recommendation():
    data = request.json
    user_id = data['user_id']
    k = data.get('k', None)

    if k is None or not (1 <= k <= len(movie_id_encoded)):
        k = len(movie_id_encoded)

    if user_id not in user_id_encoded.keys():
        result_json = df.head(k).to_json(orient="records")
        return json.dumps({'success': False, 'message': 'User not found!', 'data': json.loads(result_json)})

    result = get_collab_recommendations(user_id=user_id, movies=df, df=df,
                                        model=model, user_id_encoded=user_id_encoded, movie_id_encoded=movie_id_encoded, k=k)
    result = list(set(result))
    result_df = pd.DataFrame(result)

    result_json = result_df.to_json(orient="records")
    transformed_data = [{ "movie_title": item["0"] } for item in json.loads(result_json)]
    result_json = json.dumps(transformed_data)
    
    return json.dumps({'success': True, 'message': 'Success retrieve collaborative filtering data!', 'data': json.loads(result_json)})

# ================MAIN===============================
if __name__ == "__main__":
    app.run(debug=True)
# ===================================================

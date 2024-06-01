import json
import numpy as np
import tensorflow as tf
from flask import Flask, request
import pickle
from tensorflow import keras
from keras import layers
from keras.utils import register_keras_serializable
import requests

# ================FUNCTION===================


def get_collab_recommendations(user_id, movies, df, model, user_id_encoded, movie_id_encoded, k=5):
  movies_reviewed_by_user = df[df['user_id'] == user_id]
  movie_not_reviewed = movies[~movies['movie_id'].isin(
    movies_reviewed_by_user['movie_id'])]

  movie_not_reviewed = list(set(movie_not_reviewed['movie_id'].tolist(
  )).intersection(set(movie_id_encoded.keys())))

  movie_not_reviewed = [[user_id_encoded.get(user_id), movie_id_encoded.get(
    movie_id)] for movie_id in movie_not_reviewed]

  user_movie_array = np.array(movie_not_reviewed)
  user_movie_array = tf.convert_to_tensor(
    user_movie_array, dtype=tf.int64)  # Convert to tf.int64

  ratings = model.predict(user_movie_array).flatten()

  top_ratings_indices = ratings.argsort()[-k:][::-1]

  movie_encoded = {i: x for i, x in enumerate(movie_id_encoded)}
  recommended_movie_ids = [movie_encoded.get(
    movie_not_reviewed[x][1]) for x in top_ratings_indices]

  recommended_movie = movies[movies['movie_id'].isin(recommended_movie_ids)]

  return recommended_movie


# ================INIT FLASK========================
app = Flask(__name__)


@app.route("/")
def hello_world():
  return "Hello, World!"
# ===================================================


# ================LOAD MODEL=========================
folder = './model/'

df = pickle.load(open(folder + 'df.pkl', 'rb'))
user_id_encoded = pickle.load(open(folder + 'user_id_encoded.pkl', 'rb'))
movie_id_encoded = pickle.load(open(folder + 'movie_id_encoded.pkl', 'rb'))

model = tf.keras.models.load_model(folder + '/model_collab.keras')
# ===================================================

# ================ENDPOINT===========================


@app.route("/recommendation/collab", methods=["POST"])
def collab_recommendation():
  data = request.json
  user_id = data['user_id']
  # Use get method to get the value of 'k', default to None if not present
  k = data.get('k', None)

  if k is None or not (1 <= k <= len(movie_id_encoded)):
    k = len(movie_id_encoded)

  if user_id not in user_id_encoded.keys():
    result_json = df.head(k).to_json(orient="records")
    return json.dumps({'success': False, 'message': 'User not found!', 'data': json.loads(result_json)})

  result = get_collab_recommendations(user_id=user_id, movies=df, df=df,
                                      model=model, user_id_encoded=user_id_encoded, movie_id_encoded=movie_id_encoded, k=k)

  result_json = result.to_json(orient="records")

  return json.dumps({'success': True, 'message': 'Success retrieve collaborative filtering data!', 'data': json.loads(result_json)})
# ===================================================


# ================MAIN===============================
if __name__ == "__main__":
  app.run(debug=True)
# ===================================================
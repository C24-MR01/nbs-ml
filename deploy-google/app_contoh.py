import json
import numpy as np
import tensorflow as tf
from flask import Flask, request
import pickle


# ================FUNCTION===================


def get_collab_recommendations(user_id, items, collab_df, model, user_id_encoded, farmer_id_encoded, k=5):
  farmer_reviewed_by_user = collab_df[collab_df['user_id'] == user_id]
  farmer_not_reviewed = items[~items['farmer_id'].isin(
    farmer_reviewed_by_user['farmer_id'])]

  farmer_not_reviewed = list(set(farmer_not_reviewed['farmer_id'].tolist(
  )).intersection(set(farmer_id_encoded.keys())))

  farmer_not_reviewed = [[user_id_encoded.get(user_id), farmer_id_encoded.get(
    farmer_id)] for farmer_id in farmer_not_reviewed]

  user_farmer_array = np.array(farmer_not_reviewed)
  user_farmer_array = tf.convert_to_tensor(
    user_farmer_array, dtype=tf.int64)  # Convert to tf.int64

  ratings = model.predict(user_farmer_array).flatten()

  top_ratings_indices = ratings.argsort()[-k:][::-1]

  farmer_encoded = {i: x for i, x in enumerate(farmer_id_encoded)}
  recommended_farmer_ids = [farmer_encoded.get(
    farmer_not_reviewed[x][1]) for x in top_ratings_indices]

  recommended_farmer = items[items['farmer_id'].isin(recommended_farmer_ids)]

  return recommended_farmer


# ================INIT FLASK========================
app = Flask(__name__)


@app.route("/")
def hello_world():
  return "Hello, World!"
# ===================================================


# ================LOAD MODEL=========================
folder = './model/'

cb_df = pickle.load(open(folder + 'cb_df.pkl', 'rb'))
cosine_sim_df = pickle.load(open(folder + 'cosine_sim_df.pkl', 'rb'))

collab_df = pickle.load(open(folder + 'collab_df.pkl', 'rb'))
user_id_encoded = pickle.load(open(folder + 'user_id_encoded.pkl', 'rb'))
farmer_id_encoded = pickle.load(open(folder + 'farmer_id_encoded.pkl', 'rb'))
model = tf.keras.models.load_model(folder + '/model_collab')
# ===================================================

# ================ENDPOINT===========================


@app.route("/recommendation/cb", methods=["POST"])
def cb_recommendation():
  data = request.json
  farmer_id = data['farmer_id']
  k = data.get('k', None)

  if k is None or not (1 <= k <= len(cosine_sim_df.columns)):
    k = len(cosine_sim_df.columns)

  if farmer_id not in cosine_sim_df.index:
    result_json = cb_df.head(k).to_json(orient="records")
    return json.dumps({'success': False, 'message': 'Farmer not found!', 'data': json.loads(result_json)})

  result = get_cb_recommendations(farmer_id, cosine_sim_df, cb_df, k)

  result_json = result.to_json(orient="records")

  return json.dumps({'success': True, 'message': 'Success retrieve content based filtering data!', 'data': json.loads(result_json)})


@app.route("/recommendation/collab", methods=["POST"])
def collab_recommendation():
  data = request.json
  user_id = data['user_id']
  k = data.get('k', None)

  if k is None or not (1 <= k <= len(farmer_id_encoded)):
    k = len(farmer_id_encoded)

  if user_id not in user_id_encoded.keys():
    result_json = cb_df.head(k).to_json(orient="records")
    return json.dumps({'success': False, 'message': 'User not found!', 'data': json.loads(result_json)})

  result = get_collab_recommendations(user_id=user_id, items=cb_df, collab_df=collab_df,
                                      model=model, user_id_encoded=user_id_encoded, farmer_id_encoded=farmer_id_encoded, k=k)

  result_json = result.to_json(orient="records")

  return json.dumps({'success': True, 'message': 'Success retrieve collaborative filtering data!', 'data': json.loads(result_json)})
# ===================================================


# ================MAIN===============================
if __name__ == "__main__":
  app.run(debug=True)
# ===================================================

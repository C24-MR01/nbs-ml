# app.py

import json

from flask import Flask, request, jsonify
from utils import get_recommendations, cosine_sim2, get_collab_recommendations

app = Flask(__name__)

# Define a route for recommendation
@app.route('/recommend-synopsys', methods=['GET'])
def recommend():
    movie_id = request.args.get('id')

    if not id:
        return jsonify({'error': 'ID parameter is required'}), 400

    recommendations = get_recommendations((movie_id))
    if type(recommendations) == str:
        return recommendations
    return jsonify({'recommendations': recommendations.tolist()})


# recommendation based on genre,cast,crew

@app.route('/recommend-metadata', methods=['POST'])
def recommend_metadata():
    data = request.get_json()
    input = data.get("input")
    if not input:
        return {"error": "Input is required"}, 400
    rec_1, rec_2, rec_3, rec_4, rec_5 = get_recommendations(input, cosine_sim2)
    summary_data = {
        "rec_1": rec_1,
        "rec_2": rec_2,
        "rec_3": rec_3,
        "rec_4": rec_4,
        "rec_5": rec_5,
    }
    return summary_data, 201

@app.route("/recommendation-collab", methods=["GET"])
def recommend_collaboration():
    movie_id = request.args.get('id')

    if not id:
        return jsonify({'error': 'ID parameter is required'}), 400
    
    recommendations = get_collab_recommendations((movie_id))
    return json.dumps({'recommendations': recommendations})
    
if __name__ == '__main__':
    app.run(debug=True)

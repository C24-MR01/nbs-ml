# app.py

from flask import Flask, request, jsonify
from utils import get_recommendations

app = Flask(__name__)

# Define a route for recommendation
@app.route('/recommend-synopsys', methods=['GET'])
def recommend():
    movie_id = request.args.get('id')

    if not id:
        return jsonify({'error': 'ID parameter is required'}), 400

    recommendations = get_recommendations((movie_id))
    return jsonify({'recommendations': recommendations.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

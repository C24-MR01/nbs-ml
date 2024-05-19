# app.py

from flask import Flask, request, jsonify
from utils import get_recommendations

app = Flask(__name__)

# Define a route for recommendation
@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')

    if not title:
        return jsonify({'error': 'Title parameter is required'}), 400

    recommendations = get_recommendations(title)
    return jsonify({'recommendations': recommendations.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

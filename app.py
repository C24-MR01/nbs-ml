import json
from flask import Flask, request, jsonify
from utils import get_collab_recommendations

# ================INIT FLASK========================
app = Flask(__name__)

# ================ENDPOINT===========================
@app.route("/recommendation-collab", methods=["GET"])
def recommend_collaboration():
    movie_id = request.args.get('id')

    if not id:
        return jsonify({'error': 'ID parameter is required'}), 400
    
    recommendations = get_collab_recommendations((movie_id))
    return json.dumps({'recommendations': recommendations})


# ================MAIN===============================
if __name__ == "__main__":
    app.run(debug=True)
# ===================================================

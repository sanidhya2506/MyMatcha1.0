from flask import Flask, request, jsonify, render_template
from recommender import get_recommendations

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    data = request.json
    user_input = data.get("query", "").strip()
    if not user_input:
        return jsonify({"input": "", "recommendations": []})
    
    recommendations = get_recommendations(user_input)
    return jsonify({"input": user_input, "recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True, port=5000)

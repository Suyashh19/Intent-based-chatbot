from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# =========================
# Load trained model files
# =========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# =========================
# Routes
# =========================

@app.route("/")
def home():
    """Serve frontend UI"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict intent from user message"""
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"intent": "invalid_input"})

    user_input = data["message"].strip()

    if user_input == "":
        return jsonify({"intent": "empty_input"})

    # Vectorize input
    X = vectorizer.transform([user_input])

    # Predict intent
    intent = model.predict(X)[0]

    return jsonify({"intent": intent})


# =========================
# Run app
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

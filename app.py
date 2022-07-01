from flask import Flask, jsonify, request
from Classifier import get_pred

app = Flask(__name__)


@app.route("/predict_digit", methods=["POST"])
def predict_data():
    image = request.files.get("digit")

    prediction = get_pred(image)
    return jsonify({
        "Prediction": prediction
    }), 200


if __name__ == "__main__":
    app.run(debug=True)

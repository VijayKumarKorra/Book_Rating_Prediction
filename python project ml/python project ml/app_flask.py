
from flask import Flask, request, jsonify
from inference_utils import load_artifacts, build_features_from_payload, predict_rating

app = Flask(__name__)
pipeline, meta, pub_map = load_artifacts()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected JSON object."}), 400
        feat = build_features_from_payload(payload, meta, pub_map)
        pred = predict_rating(pipeline, feat)
        return jsonify({"predicted_average_rating": round(pred, 3)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Usage: python app_flask.py
    # Then POST to http://127.0.0.1:5000/predict with JSON body.
    app.run(host="0.0.0.0", port=5000, debug=False)

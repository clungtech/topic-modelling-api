from flask import Flask, request, jsonify
import pickle
import os

# Load model and vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
topics = pickle.load(open("model/topics.pkl", "rb"))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    X = vectorizer.transform([text])
    y_pred = model.predict(X)
    predicted_topics = [topics[i] for i, val in enumerate(y_pred[0]) if val == 1]
    return jsonify({'topics': predicted_topics})

# This is the entrypoint for Vercel
def handler(request):
    with app.test_request_context(
        path=request.path,
        base_url=request.base_url,
        method=request.method,
        headers=request.headers,
        data=request.body,
        query_string=request.query_string
    ):
        response = app.full_dispatch_request()
        return (response.data, response.status_code, response.headers.items())

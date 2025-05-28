from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load saved files
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/topics.pkl", "rb") as f:
    topic_labels = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data['title'] + " " + data['abstract']
    X = vectorizer.transform([text])
    y_pred = model.predict(X)[0]
    
    topics = [topic_labels[i] for i in range(len(topic_labels)) if y_pred[i] == 1]
    
    return jsonify({"predicted_topics": topics})

if __name__ == "__main__":
    app.run(debug=True)

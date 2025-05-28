import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Load CSV
df = pd.read_csv("train.csv")

# Combine title and abstract
df['text'] = df['TITLE'] + " " + df['ABSTRACT']

# Select topic columns
topic_columns = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
y = df[topic_columns].values

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Save all components
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/topics.pkl", "wb") as f:
    pickle.dump(topic_columns, f)

print("✅ Model trained and saved.")

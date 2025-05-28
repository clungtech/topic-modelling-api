import pandas as pd
import pickle
import re

# 1. Load test.csv
df = pd.read_csv('test.csv')

# 2. Combine TITLE and ABSTRACT
df['text'] = df['TITLE'].fillna('') + '. ' + df['ABSTRACT'].fillna('')

# 3. Clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.lower().strip()       # Lowercase and strip
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# 4. Load trained vectorizer and model
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# 5. Transform text to feature vectors
X_test = vectorizer.transform(df['cleaned_text'])

# 6. Predict topic distribution
predictions = model.predict(X_test)

# 7. Choose the most dominant topic
df['PREDICTED_TOPIC'] = predictions.argmax(axis=1)

# Map predicted topic indices to human-readable topic names
topic_labels = {
    0: "Machine Learning",
    1: "Astronomy",
    2: "Natural Language Processing",
    3: "Computer Vision",
    4: "Physics",
    5: "Healthcare AI"
}

df['TOPIC_NAME'] = df['PREDICTED_TOPIC'].map(topic_labels)

# 8. Save results to CSV
df[['ID', 'PREDICTED_TOPIC', 'TOPIC_NAME']].to_csv('predicted_test_output.csv', index=False)

print("✅ Prediction complete. Output saved to 'predicted_test_output.csv'")

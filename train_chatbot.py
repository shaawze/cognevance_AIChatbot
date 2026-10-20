

import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load training examples
with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

texts = []
labels = []

for intent, examples in intents.items():
    for example in examples:
        texts.append(example)
        labels.append(intent)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

# Build and train the model
chatbot_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

chatbot_model.fit(X_train, y_train)

# Evaluate the model
y_pred = chatbot_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=0))

# Save the trained model
joblib.dump(chatbot_model, "chatbot_model.pkl")

print("Model saved successfully!")


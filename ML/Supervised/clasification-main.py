#clasification

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Load dataset
file_path = "spam.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "uciml/sms-spam-collection-dataset",
    file_path,
    pandas_kwargs={"encoding": "latin-1"}
)

# The dataset has extra columns, we only keep 'v1' (label) and 'v2' (message)
df = df[['v1', 'v2']]
df.columns = ['label', 'message']  # rename columns

print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------
# Step 1: Prepare data
# ---------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Inputs (X) = text messages, Outputs (y) = labels
X = df['message']
y = df['label']

# Split dataset into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------
# Step 2: Convert text into numeric features
# ---------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------
# Step 3: Train model
# ---------------------------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ---------------------------------------------
# Step 4: Evaluate model
# ---------------------------------------------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------
# Step 5: Test with new messages
# ---------------------------------------------
new_messages = [
    "Congratulations! You won a free iPhone, click here to claim.",
    "Hey, are we still on for dinner tonight?"
]

new_messages_tfidf = vectorizer.transform(new_messages)
predictions = model.predict(new_messages_tfidf)

for msg, label in zip(new_messages, predictions):
    print(f"Message: {msg} --> Prediction: {label}")

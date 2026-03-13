import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/emails.csv")

# Split features and labels
X = data["email"]
y = data["label"]

# Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model
with open("model/spam_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Model trained and saved successfully!")
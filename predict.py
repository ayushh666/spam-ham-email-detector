import pickle

# Load model
with open("model/spam_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# Take user input
email = input("Enter email text: ")

# Transform input
email_vector = vectorizer.transform([email])

# Predict
prediction = model.predict(email_vector)

print("Prediction:", prediction[0])
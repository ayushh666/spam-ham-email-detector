import streamlit as st
import pickle

# Load model
with open("model/spam_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

st.title("📧 Spam Ham Email Detector")

email = st.text_area("Enter email text")

if st.button("Detect Spam"):
    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)

    if prediction[0] == "spam":
        st.error("This email is SPAM")
    else:
        st.success("This email is HAM (Safe)")
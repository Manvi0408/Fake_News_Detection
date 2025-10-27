import streamlit as st
from joblib import load

# Load model and vectorizer
model = load('final_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

st.title("Fake News Detector")

news = st.text_area("Enter news text here:")

if st.button("Predict"):
    vect_text = vectorizer.transform([news])
    prediction = model.predict(vect_text)[0]
    prob = model.predict_proba(vect_text)[0][1]
    st.write(f"Prediction: {prediction}")
    st.write(f"Truth Probability: {prob:.2f}")

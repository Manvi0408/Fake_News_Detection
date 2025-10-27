
# -*- coding: utf-8 -*-
"""
Updated prediction.py to work with current scikit-learn
"""

from joblib import load

# Load the trained model and vectorizer
model = load('final_model.sav')         # <- matches your folder
vectorizer = load('tfidf_vectorizer.joblib')

# Function to predict fake news
def detecting_fake_news(var):
    vect_text = vectorizer.transform([var])
    prediction = model.predict(vect_text)
    prob = model.predict_proba(vect_text)
    
    print("The given statement is:", prediction[0])
    print("The truth probability score is:", prob[0][1])

# Main block
if __name__ == '__main__':
    var = input("Please enter the news text you want to verify: ")
    print("You entered:", var)
    detecting_fake_news(var)

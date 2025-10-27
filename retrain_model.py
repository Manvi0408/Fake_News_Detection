import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

# Load dataset
data = pd.read_csv('train.csv')  # CSV with 'text'/'Statement' and 'label'/'Label'

# Adjust column names
if 'Statement' in data.columns:
    X = data['Statement']
    y = data['Label']
else:
    X = data['text']
    y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

# Accuracy
y_pred = model.predict(X_test_vect)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Save
dump(model, 'final_model.joblib')
dump(vectorizer, 'tfidf_vectorizer.joblib')

print("✅ Model and vectorizer retrained and saved successfully!")

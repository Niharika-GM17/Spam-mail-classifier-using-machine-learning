# Spam Mail Classifier using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
# Dataset can be downloaded from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']  # rename columns

# 2. Convert labels to binary (spam=1, ham=0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# 4. Convert text to numerical vectors (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Predictions
y_pred = model.predict(X_test_tfidf)

# 7. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import pandas as pd

# Load CSV
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Quick look
print(df.head())
print(df['label'].value_counts())
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)  # fit + transform on training
X_test_vec = vectorizer.transform(X_test)        # transform only on test
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
import pickle

# Save trained model
pickle.dump(model, open('model.pkl', 'wb'))

# Save vectorizer
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model and vectorizer saved!")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Load Dataset
# You can download dataset from Kaggle or UCI (SMS Spam Collection dataset)
# For now, assume your dataset is 'spam.csv' with columns ['label','message']
data = pd.read_csv("spam.csv", encoding="latin-1")[['v1','v2']]
data.columns = ['label', 'message']

# Convert labels to binary
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# 3. Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Save Model and Vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and Vectorizer saved!")

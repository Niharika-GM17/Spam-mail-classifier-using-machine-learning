import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------
# Load or Train Model
# -----------------------
# If you already saved your trained model, load it.
# For now, let's train quickly inside this file (basic demo).
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# -----------------------
# Streamlit UI
# -----------------------
st.title("📧 Spam Mail Classifier")
st.write("Enter a message and check if it is **Spam** or **Ham**")

# Input box
user_input = st.text_area("Type your message here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.error("🚨 This is a SPAM message!")
        else:
            st.success("✅ This is a HAM (not spam) message.")

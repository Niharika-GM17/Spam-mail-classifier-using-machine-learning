from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# --- Link Detection Helpers ---
def contains_link(message):
    url_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
    return bool(re.search(url_pattern, message))

def is_suspicious_link(message):
    url_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
    urls = re.findall(url_pattern, message)
    for url in urls:
        if len(url) > 60 or "bit.ly" in url or "tinyurl" in url:
            return True
    return False

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    # Check for suspicious links before using ML
    if contains_link(message):
        if is_suspicious_link(message):
            prediction = "Spam (Suspicious Link)"
        else:
            transformed_message = vectorizer.transform([message])
            prediction = model.predict(transformed_message)[0]
    else:
        transformed_message = vectorizer.transform([message])
        prediction = model.predict(transformed_message)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

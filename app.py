from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # simple HTML page

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['message']
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    result = 'Spam' if prediction == 1 else 'Ham'
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

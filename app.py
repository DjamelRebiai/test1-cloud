import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify

# Load data and train model
print("Loading data and training model...")
df = pd.read_csv("emails_dataset.csv")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["email"])
y = df["label"]
model = MultinomialNB()
model.fit(X, y)
print("Model trained successfully.")

# Create Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "email" not in data:
            return jsonify({"error": "Missing 'email' field in request json"}), 400
        
        text = data["email"]
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)
        return jsonify({"classe": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

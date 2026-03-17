import imaplib
import email
from email.header import decode_header
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify, render_template

# 1. Load data and train model
print("Loading data and training model...")
df = pd.read_csv("emails_dataset.csv")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["email"])
y = df["label"]
model = MultinomialNB()
model.fit(X, y)
print("Model trained successfully.")

app = Flask(__name__)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.replace("\r", " ").replace("\n", " ").strip()

def decode_mime_words(s):
    if not s:
        return ""
    return ''.join(
        word.decode(encoding or 'utf-8') if isinstance(word, bytes) else word
        for word, encoding in decode_header(s)
    )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "email" not in data:
            return jsonify({"error": "Missing 'email' field in request json"}), 400
        
        text = data["email"]
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)
        return jsonify({"classe": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/fetch-emails", methods=["POST"])
def fetch_emails():
    try:
        data = request.get_json()
        server = data.get("server", "imap.gmail.com")
        email_addr = data.get("email")
        password = data.get("password")
        limit = min(int(data.get("limit", 5)), 20)  # Max 20 emails
        
        if not email_addr or not password:
            return jsonify({"error": "Email and App Password required"}), 400

        # Connect to server
        mail = imaplib.IMAP4_SSL(server)
        mail.login(email_addr, password)
        mail.select("inbox")

        # Search for recent emails
        status, messages = mail.search(None, "ALL")
        if status != "OK":
            return jsonify({"error": "No messages found!"}), 404

        email_ids = messages[0].split()
        latest_email_ids = email_ids[-limit:]
        
        results = []
        for e_id in reversed(latest_email_ids):
            status, msg_data = mail.fetch(e_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_mime_words(msg["Subject"])
                    sender = decode_mime_words(msg.get("From", ""))
                    
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))
                            if content_type == "text/plain" and "attachment" not in content_disposition:
                                try:
                                    body = part.get_payload(decode=True).decode()
                                    break
                                except:
                                    pass
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode()
                        except:
                            pass
                            
                    body = clean_text(body)
                    if not body:
                        body = clean_text(subject)

                    # Predict
                    vector = vectorizer.transform([body])
                    prediction = model.predict(vector)[0]
                    
                    results.append({
                        "subject": subject,
                        "sender": sender,
                        "body": body,
                        "prediction": prediction
                    })

        mail.logout()
        return jsonify({"emails": results})

    except imaplib.IMAP4.error:
        return jsonify({"error": "Authentication failed. Make sure you use an App Password."}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

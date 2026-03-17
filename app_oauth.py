import os
import base64
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify, render_template, redirect, session, url_for

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "emails_dataset.csv")
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, "client_secret.json")

print("Loading data and training model...")
df = pd.read_csv(DATASET_PATH)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["email"])
y = df["label"]
model = MultinomialNB()
model.fit(X, y)
print("Model trained successfully.")

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Necessary for session cookies

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.replace("\r", " ").replace("\n", " ").strip()

@app.route("/")
def index():
    # Handle the OAuth2 callback from Google
    if 'code' in request.args:
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, scopes=SCOPES)
        
        # Determine the redirect URI based on the request host
        if "localhost" in request.url_root or "127.0.0.1" in request.url_root:
            flow.redirect_uri = request.base_url
        else:
            flow.redirect_uri = 'https://test1-cloud.onrender.com/'
            
        flow.fetch_token(authorization_response=request.url)
        creds = flow.credentials
        
        session['credentials'] = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        return redirect(url_for('index'))

    return render_template("oauth.html", logged_in='credentials' in session)

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

@app.route("/login")
def login():
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES)
    
    if "localhost" in request.url_root or "127.0.0.1" in request.url_root:
        flow.redirect_uri = request.base_url.replace('/login', '/')
    else:
        flow.redirect_uri = 'https://test1-cloud.onrender.com/'

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent')

    session['state'] = state
    return redirect(authorization_url)

@app.route("/logout")
def logout():
    session.pop('credentials', None)
    return redirect(url_for('index'))

@app.route("/fetch-emails-oauth", methods=["GET"])
def fetch_emails_oauth():
    if 'credentials' not in session:
        return jsonify({"error": "Not authenticated. Please login first."}), 401

    creds_data = session['credentials']
    creds = Credentials(
        token=creds_data['token'],
        refresh_token=creds_data['refresh_token'],
        token_uri=creds_data['token_uri'],
        client_id=creds_data['client_id'],
        client_secret=creds_data['client_secret'],
        scopes=creds_data['scopes']
    )

    try:
        service = build('gmail', 'v1', credentials=creds)
        # Fetch the latest 5 emails
        results = service.users().messages().list(userId='me', maxResults=5).execute()
        messages = results.get('messages', [])

        if not messages:
            return jsonify({"emails": []})

        emails_result = []
        for msg in messages:
            msg_id = msg['id']
            message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
            
            payload = message.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = "(No Subject)"
            sender = "(Unknown)"
            for header in headers:
                if header['name'] == 'Subject':
                    subject = header['value']
                elif header['name'] == 'From':
                    sender = header['value']
            
            # Extract plain text body
            body = ""
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        data = part['body'].get('data')
                        if data:
                            body = base64.urlsafe_b64decode(data).decode('utf-8')
                            break
            else:
                data = payload.get('body', {}).get('data')
                if data:
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
                    
            body = clean_text(body)
            if not body:
                body = clean_text(subject)

            # Predict the class
            vector = vectorizer.transform([body])
            prediction = model.predict(vector)[0]
            
            emails_result.append({
                "subject": subject,
                "sender": sender,
                "body": body,
                "prediction": prediction
            })

        return jsonify({"emails": emails_result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # ONLY for local testing. Remove this setting if running on Render.
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    app.run(host="0.0.0.0", port=5002, debug=True)

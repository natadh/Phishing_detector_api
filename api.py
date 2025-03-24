from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request body model
class EmailRequest(BaseModel):
    email_text: str

@app.post("/predict/")
async def predict_phishing(request: EmailRequest):
    # Transform input text using the trained vectorizer
    email_vector = vectorizer.transform([request.email_text])
    
    # Make prediction
    prediction = model.predict(email_vector)[0]  # 1 = Phishing, 0 = Legitimate
    probability = model.predict_proba(email_vector)[0][1]  # Probability of phishing
    
    return {"prediction": int(prediction), "probability": float(probability)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

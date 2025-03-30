from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading my trained model and vectorizer
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Initializing FastAPI app
app = FastAPI()

# Defining request body model
class EmailRequest(BaseModel):
    email_text: str

@app.post("/predict/")
def predict_phishing(request: EmailRequest):
    # Vectorizing the input email text
    email_vector = vectorizer.transform([request.email_text])
    
    # Getting the probability that the email is phishing (model outputs probability for phishing class)
    probability = model.predict_proba(email_vector)[0][1]
    
    # Setting a custom threshold for classification
    threshold = 0.8  # Only classify as phishing if probability is 80% or higher
    
    # Use the custom threshold to determine prediction
    prediction = 1 if probability >= threshold else 0
    
    # Determine risk level and recommendation based on the probability
    if probability >= 0.8:
        risk_level = "High"
        recommendation = (
            "Warning: This email is likely phishing. Do not click on any links or provide personal information."
        )
    elif probability >= 0.5:
        risk_level = "Medium"
        recommendation = (
            "Caution: This email may be suspicious. Verify the sender and content before taking any action."
        )
    else:
        risk_level = "Low"
        recommendation = (
            "This email appears legitimate. However, always double-check unexpected requests."
        )
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": risk_level,
        "recommendation": recommendation
    }

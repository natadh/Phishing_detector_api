print("Phishing Detector Project Initialized!")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# For initializing the FastAPI
app = FastAPI()

# Prints a message when my app starts
@app.on_event("startup")
def startup_event():
    print("Starting up FastAPI...")

# Enable CORS so that my Outlook add-in (or other frontends) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Home endpoint for basic testing
@app.get("/")
def home():
    return {"message": "Phishing Detector API is running!"}

# Loading my trained model & vectorizer 
print("Loading model and vectorizer...")
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
print("Model and vectorizer loaded successfully!")

# Defining a request body model for the /predict/ endpoint
class EmailRequest(BaseModel):
    email_text: str

# Prediction endpoint for phishing detection with custom threshold and recommendations
@app.post("/predict/")
def predict_phishing(request: EmailRequest):
    # Vectorize the email text
    email_vector = vectorizer.transform([request.email_text])
    
    # Get probability of phishing
    probability = model.predict_proba(email_vector)[0][1]
    
    # Use a custom threshold: classify as phishing only if probability >= 0.8
    threshold = 0.8
    prediction = 1 if probability >= threshold else 0
    
    # Determine risk level and recommendation based on probability
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

# Runing the app (only if this file is executed directly, not imported)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

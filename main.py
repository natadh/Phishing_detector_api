print("Phishing Detector Project Initialized!")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Initialize FastAPI
app = FastAPI()

# Print a message when the app starts
@app.on_event("startup")
def startup_event():
    print("Starting up FastAPI...")

# Enable CORS so your Outlook add-in can call this API
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

# Load your trained model & vectorizer (make sure these files exist)
print("Loading model and vectorizer...")
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
print("Model and vectorizer loaded successfully!")

# Define a request body model for the /predict/ endpoint
class EmailRequest(BaseModel):
    email_text: str

# Prediction endpoint for phishing detection
@app.post("/predict/")
def predict_phishing(request: EmailRequest):
    # Vectorize the email text
    email_vector = vectorizer.transform([request.email_text])
    # Make prediction
    prediction = model.predict(email_vector)[0]  # 1 = Phishing, 0 = Legitimate
    probability = model.predict_proba(email_vector)[0][1]  # Probability of phishing

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

# Run the app (only if this file is executed directly, not imported)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

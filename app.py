import streamlit as st
import joblib
from preprocess import clean_text
from database import init_db, save_prediction  # Importing the database functions

# === Step 3.3: Initialize the Database ===
# This call ensures the DB and table exist each time the app runs.
init_db()


# Load the trained model and TF-IDF vectorizer
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Define a function for recommendations based on prediction and model confidence
# def get_recommendation(prediction, prob):
#     if prediction == 1:
#         return "Warning: This email is likely phishing. Do not click on any links, report it, and delete the email."
#     else:
#         return "This email appears legitimate. However, always verify the sender's authenticity before taking any actions."

def get_recommendation(prediction, proba):
    risk = "Low"
    if proba >= 0.8:
        risk = "High"
    elif proba >= 0.4:
        risk = "Medium"

    if prediction == 1:
        return f"Warning: This email is likely phishing. Risk level: {risk}. Do not click on links."
    else:
        return f"This email appears legitimate. Risk level: {risk}. Always verify the sender's authenticity."


# Streamlit UI
st.title("Email Phishing Detector")
st.write("Paste the email text below and click 'Analyze Email' to check if it's phishing.")

# Text area for user to paste email text
email_text = st.text_area("Email Text", height=250)

if st.button("Analyze Email"):
    if email_text:
        cleaned = clean_text(email_text)
        features = vectorizer.transform([cleaned])
        
        # Get prediction and prediction probability
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]  # Probability of phishing

        # Use the enhanced recommendation function
        recommendation = get_recommendation(prediction, proba)
        
        # Determine risk level based on probability
        # if proba >= 0.8:
        #     risk_level = "High"
        # elif proba >= 0.4:
        #     risk_level = "Medium"
        # else:
        #     risk_level = "Low"
        
        # Get recommendation including risk level
        # recommendation = f"Risk Level: {risk_level}. " + (
        #     "Warning: This email is likely phishing. Do not click on any links."
        #     if prediction == 1 else
        #     "This email appears legitimate. Always verify the sender's authenticity."
        # )
        
        st.markdown(f"### **Prediction:** {'Phishing' if prediction == 1 else 'Legitimate'}")
        st.markdown(f"### **Recommendation:** {recommendation}")
        st.write("Prediction Probability:", proba)

        # Saving the Prediction to the Database 
        save_prediction(email_text, cleaned, int(prediction), recommendation)
        st.write("✅ Prediction saved to database!")
    else:
        st.warning("Please enter some email text.")

        




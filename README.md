# 🛡️ AI-Driven Phishing Email Detection System (AIDE)

<img width="714" height="375" alt="image" src="https://github.com/user-attachments/assets/fa011ade-9246-486a-8108-a9d6c6ce19b8" />
<img width="972" height="572" alt="image" src="https://github.com/user-attachments/assets/d269ff5f-0062-4642-894b-0982b6298e86" />


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Outlook](https://img.shields.io/badge/Outlook-Add--in-0078D4.svg)](https://docs.microsoft.com/en-us/office/dev/add-ins/)

An ML-powered email classification system that identifies phishing attempts with **96% accuracy**, trained on over **82,000 real emails**. Deployed as a Microsoft Outlook add-in for seamless, real-time threat detection directly in your inbox.

![Phishing Detection Demo](<img width="803" height="883" alt="image" src="https://github.com/user-attachments/assets/29aa9f52-6f52-43a0-bc84-78dce85e4507" />
)
<img width="972" height="589" alt="image" src="https://github.com/user-attachments/assets/a612171f-036f-47ed-91d9-923ecffd5ef2" />
<img width="972" height="586" alt="image" src="https://github.com/user-attachments/assets/6d50449b-9f86-40c9-8943-b6e9981afe44" />


*Real-time phishing detection integrated into Microsoft Outlook*

---

## 🎯 What It Does

AIDE classifies incoming emails as **phishing** or **legitimate** by analyzing email content using advanced Natural Language Processing (NLP) and machine learning techniques. The system provides:

- **Real-time threat detection** with confidence scores (0-100%)
- **Three-tier risk assessment** (High/Medium/Low)
- **Actionable recommendations** for each email
- **Native Outlook integration** - no need to leave your inbox
- **Audit trail** for all scanned emails

### Key Features

✅ **96% Accuracy** with 0.95 F1-score  
✅ **Sub-second predictions** (~300ms average response time)  
✅ **Cross-platform** - Works on Outlook Desktop, Web, and Mobile  
✅ **Privacy-focused** - Emails are processed and immediately discarded  
✅ **Customizable threshold** - Adjustable sensitivity for your security needs  

---

## 🔍 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│        MICROSOFT OUTLOOK                         │
│  ┌───────────────────────────────────────────┐  │
│  │  Outlook Add-in (Office.js)               │  │
│  │  • Extracts email content                 │  │
│  │  • Displays phishing alerts               │  │
│  └───────────────┬───────────────────────────┘  │
└──────────────────┼──────────────────────────────┘
                   │ HTTPS POST
                   ▼
┌─────────────────────────────────────────────────┐
│        FastAPI Backend (Python)                  │
│  ┌───────────────────────────────────────────┐  │
│  │  1. Preprocess text (NLTK)                │  │
│  │  2. Vectorize with TF-IDF (5000 features) │  │
│  │  3. Classify with Logistic Regression     │  │
│  │  4. Calculate risk level & recommendation │  │
│  └───────────────┬───────────────────────────┘  │
└──────────────────┼──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│        SQLite Database                           │
│  • Logs predictions for audit                    │
│  • Stores email hashes (not content)             │
│  • Enables model retraining                      │
└─────────────────────────────────────────────────┘
```

### Detection Pipeline

1. **Text Preprocessing** (preprocess.py)
   - Convert to lowercase
   - Remove URLs, numbers, and punctuation
   - Tokenize into words
   - Lemmatize (reduce words to root form)
   - Remove stopwords (common words like "the", "is")

2. **Feature Extraction** (train_model.py)
   - TF-IDF vectorization (Term Frequency-Inverse Document Frequency)
   - Converts text to 5,000 numerical features
   - Highlights distinctive words (e.g., "urgent", "verify", "account")

3. **Classification** (main.py)
   - Tuned Logistic Regression model
   - Custom threshold: 0.8 (reduces false positives)
   - Outputs probability score (0.0 to 1.0)

4. **Risk Assessment**
   - **High Risk** (≥80%): Likely phishing - Do not interact
   - **Medium Risk** (50-79%): Suspicious - Verify before acting
   - **Low Risk** (<50%): Appears legitimate

5. **Response**
   ```json
   {
     "prediction": 1,
     "probability": 0.94,
     "risk_level": "High",
     "recommendation": "Warning: This email is likely phishing. Do not click on any links or provide personal information."
   }
   ```

---

## 🧠 Machine Learning Details

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 96% |
| **Precision** | 94% |
| **Recall** | 96% |
| **F1-Score** | 0.95 |

### Training Data

- **Dataset Size:** 82,486 emails
- **Phishing Emails:** 42,891 (52%)
- **Legitimate Emails:** 39,595 (48%)
- **Sources:** Enron, SpamAssassin, Nazario, Nigerian Fraud, CEAS datasets
- **Split:** 80% training (65,989) / 20% testing (16,497)

### Model Comparison

We evaluated multiple algorithms before selecting Logistic Regression:

| Model | Accuracy | F1-Score | Speed | Model Size |
|-------|----------|----------|-------|------------|
| **Logistic Regression ✅** | 96% | 0.95 | Fast | 2 MB |
| Random Forest | 97% | 0.96 | Slow | 50 MB |
| Naive Bayes | 94% | 0.93 | Very Fast | 1 MB |
| SVM | 95% | 0.94 | Slow | 10 MB |

**Why Logistic Regression?**
- Nearly identical accuracy to Random Forest
- 25x smaller model size (faster loading)
- 10x faster predictions (better for real-time API)
- Easier to interpret and debug

### Hyperparameter Tuning

Used GridSearchCV with 5-fold cross-validation:

```python
Best Parameters:
- C: 0.1 (regularization strength)
- solver: 'saga' (optimization algorithm)
- max_iter: 1000

Cross-validation score: 96%
```

---

## 🛠️ Technical Stack

### Backend
- **Framework:** FastAPI (async Python web framework)
- **ML Library:** scikit-learn (Logistic Regression, TF-IDF)
- **NLP:** NLTK (tokenization, lemmatization, stopwords)
- **Data Processing:** pandas, numpy
- **Model Serialization:** joblib
- **Database:** SQLite (production: PostgreSQL-ready)
- **Server:** Uvicorn (ASGI server)

### Frontend (Outlook Add-in)
- **Office.js:** Microsoft's add-in framework
- **JavaScript:** ES6+ with async/await
- **UI Framework:** Office UI Fabric (Fluent Design)
- **Build Tool:** Webpack 5
- **Dev Server:** webpack-dev-server with hot reload
- **Package Manager:** npm

### Deployment
- **API Hosting:** Render.com (with auto-deploy from GitHub)
- **Add-in Hosting:** Render.com / Azure Static Web Apps
- **SSL:** Let's Encrypt (automatic HTTPS)
- **CI/CD:** GitHub Actions (planned)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 14+ and npm (for Outlook add-in)
- Microsoft Outlook (Desktop or Web)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/natadh/Phishing_detector_api.git
cd Phishing_detector_api
```

#### 2. Set Up Backend (FastAPI)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from database import init_db; init_db()"
```

#### 3. Train the Model (Optional - Pre-trained model included)

```bash
# Download dataset (if not included)
# Place phishing_email.csv in data/ folder

# Train model
python train_model.py

# Output:
# ✅ Model trained successfully!
# ✅ Model saved to models/phishing_model.pkl
# ✅ Vectorizer saved to models/tfidf_vectorizer.pkl
```

#### 4. Run the API

```bash
# Development mode (with auto-reload)
python main.py

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

API will be available at: `http://localhost:8000`

**Test the API:**

```bash
# Using curl
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Urgent! Your account will be suspended. Click here to verify: http://phishing-site.com"}'

# Response:
{
  "prediction": 1,
  "probability": 0.94,
  "risk_level": "High",
  "recommendation": "Warning: This email is likely phishing. Do not click on any links or provide personal information."
}
```

**API Documentation:**

Visit `http://localhost:8000/docs` for interactive Swagger documentation.

#### 5. Set Up Outlook Add-in

```bash
cd outlook-addin

# Install dependencies
npm install

# Generate SSL certificates (required for local development)
npx office-addin-dev-certs install

# Update API URL in taskpane.html or taskpane.js
# Change: http://127.0.0.1:8000/predict/
# To your API endpoint

# Start development server and sideload add-in
npm start
```

This will:
- Start webpack dev server at `https://localhost:3000`
- Validate `manifest.xml`
- Sideload add-in into Outlook
- Open Outlook automatically

---

## 📖 Usage

### Using the Outlook Add-in

1. **Open an email** in Microsoft Outlook
2. **Click the add-in icon** in the ribbon (or task pane)
3. **Click "Analyze Email"** button
4. **View results** instantly:
   ```
   Prediction: Phishing
   Probability: 0.94
   Risk Level: High
   Recommendation: Warning: This email is likely phishing.
   Do not click on any links or provide personal information.
   ```

### Using the API Directly

#### Python Example

```python
import requests

api_url = "http://localhost:8000/predict/"
email_text = """
Dear Customer,

Your PayPal account has been limited. Please verify your information immediately
by clicking the link below, or your account will be permanently suspended.

http://paypa1-verify.com/login

Thank you,
PayPal Security Team
"""

response = requests.post(
    api_url,
    json={"email_text": email_text}
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommendation: {result['recommendation']}")
```

#### JavaScript Example

```javascript
const analyzeEmail = async (emailText) => {
  const response = await fetch('http://localhost:8000/predict/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email_text: emailText })
  });
  
  const result = await response.json();
  console.log(`Risk Level: ${result.risk_level}`);
  console.log(`Probability: ${result.probability}`);
};

analyzeEmail("Congratulations! You've won $1,000,000...");
```

---

## 📁 Project Structure

```
Phishing_detector_api/
├── main.py                      # FastAPI application & endpoints
├── train_model.py               # ML model training script
├── preprocess.py                # Text cleaning & NLP functions
├── database.py                  # SQLite database operations
├── requirements.txt             # Python dependencies
│
├── models/
│   ├── phishing_model.pkl       # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl     # TF-IDF vectorizer
│
├── data/
│   └── phishing_email.csv       # Training dataset (82K+ emails)
│
├── phishing_detector.db         # SQLite database (auto-created)
│
└── outlook-addin/
    ├── manifest.xml             # Add-in configuration
    ├── taskpane.html            # Add-in UI
    ├── taskpane.js              # Add-in logic (Office.js)
    ├── taskpane.css             # Add-in styling
    ├── package.json             # Node.js dependencies
    └── webpack.config.js        # Build configuration
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Model Configuration
MODEL_PATH=models/phishing_model.pkl
VECTORIZER_PATH=models/tfidf_vectorizer.pkl
PREDICTION_THRESHOLD=0.8

# Database
DATABASE_URL=sqlite:///phishing_detector.db
# For production: postgresql://user:pass@host:port/dbname

# CORS (comma-separated origins)
ALLOWED_ORIGINS=http://localhost:3000,https://localhost:3000
# For production: https://your-addin-domain.com
```

### Adjusting Sensitivity

Modify the threshold in `main.py`:

```python
# Higher threshold = fewer false positives (more cautious)
threshold = 0.8  # Default: 80% confidence required

# Lower threshold = catches more phishing (more false positives)
threshold = 0.5  # 50% confidence required
```

---

## 🧪 Testing

### Run Unit Tests

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html
```

### Test API Endpoints

```bash
# Test health check
curl http://localhost:8000/

# Test prediction endpoint
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Test phishing email with urgent action required!"}'
```

### Manual Testing Checklist

- [ ] API returns predictions for legitimate emails
- [ ] API returns predictions for phishing emails
- [ ] Outlook add-in loads correctly
- [ ] Email extraction works in Outlook
- [ ] Results display properly in task pane
- [ ] Error handling works (API down, network error)
- [ ] SSL certificate is valid
- [ ] CORS is configured correctly

---

## 📊 Performance Metrics

### Response Times

| Operation | Average Time |
|-----------|--------------|
| Email extraction (Office.js) | ~100ms |
| Text preprocessing | ~20ms |
| TF-IDF vectorization | ~10ms |
| Model prediction | ~5ms |
| Database logging | ~15ms |
| Total API response | ~50ms |
| **End-to-end (user click → result)** | **~300ms** |

### Scalability

- **Current capacity:** ~500 requests/second (single instance)
- **Database:** SQLite handles ~10,000 reads/second
- **Memory usage:** ~200MB (model + vectorizer loaded)
- **Model size:** 2MB (fast loading)

---

## 🚢 Deployment

### Deploy Backend to Render.com

1. **Create account** at [render.com](https://render.com)

2. **Create new Web Service**
   - Connect GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables**
   ```
   DEBUG=False
   ALLOWED_ORIGINS=https://your-addin-domain.com
   ```

4. **Deploy** - Render auto-deploys on git push

5. **Get API URL**: `https://your-api.onrender.com`

### Deploy Outlook Add-in

#### Option 1: Render.com (Static Site)

```bash
cd outlook-addin

# Update API URL in taskpane.js
# Change localhost:8000 to your deployed API URL

# Build for production
npm run build

# Deploy dist/ folder to Render as static site
```

#### Option 2: Azure Static Web Apps

```bash
# Install Azure CLI
az login

# Create static web app
az staticwebapp create \
  --name phishing-detector-addin \
  --resource-group myResourceGroup \
  --source https://github.com/natadh/Phishing_detector_api \
  --location "East US" \
  --branch main \
  --app-location "outlook-addin" \
  --output-location "dist"
```

#### Option 3: GitHub Pages

```bash
# Build
npm run build

# Deploy dist/ to gh-pages branch
npm run deploy
# Or manually:
git subtree push --prefix outlook-addin/dist origin gh-pages
```

### Update Manifest for Production

```xml
<!-- manifest.xml -->
<SourceLocation DefaultValue="https://your-addin-domain.com/taskpane.html" />
```

### Distribute Add-in

**Option 1: Microsoft AppSource** (Public)
- Submit to Partner Center
- Review process (~5-7 days)
- Available to all Office 365 users

**Option 2: Centralized Deployment** (Enterprise)
- IT admin deploys organization-wide
- No user action required

**Option 3: Network Share** (Internal)
- Host manifest on company server
- Users add via "My Add-ins" → "Add from file"

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **URL Analysis**
  - Domain reputation checking
  - Shortened URL expansion
  - SSL certificate validation
  - Detect lookalike domains (paypa1.com vs paypal.com)

- [ ] **Sender Verification**
  - SPF/DKIM/DMARC validation
  - Sender reputation scoring
  - Domain age checking

- [ ] **Deep Learning Models**
  - Fine-tune BERT for better context understanding
  - Capture semantic meaning and intent
  - Handle multi-language emails

- [ ] **User Feedback Loop**
  - "Report Incorrect" button
  - Active learning for continuous improvement
  - Reduce false positives over time

- [ ] **Multi-Platform Support**
  - Gmail Chrome extension
  - Apple Mail plugin (MailKit)
  - Thunderbird add-on

- [ ] **Advanced Analytics**
  - Admin dashboard (Streamlit)
  - Threat trends over time
  - Top phishing domains
  - User statistics

- [ ] **Enterprise Features**
  - Batch email scanning
  - Role-based access control
  - Compliance reporting (GDPR, HIPAA)
  - Integration with SIEM systems

---

## 📚 Documentation

### API Reference

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Phishing Detector API is running!"
}
```

#### `POST /predict/`
Analyze email for phishing.

**Request Body:**
```json
{
  "email_text": "string"
}
```

**Response:**
```json
{
  "prediction": 0 | 1,  // 0 = legitimate, 1 = phishing
  "probability": 0.0-1.0,  // confidence score
  "risk_level": "High" | "Medium" | "Low",
  "recommendation": "string"  // actionable advice
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Urgent! Verify your account now!"}'
```

### Office.js Integration

The add-in uses Office.js to extract email content:

```javascript
// Extract email body
Office.context.mailbox.item.body.getAsync(
  Office.CoercionType.Text,
  (result) => {
    if (result.status === Office.AsyncResultStatus.Succeeded) {
      const emailBody = result.value;
      // Send to API
    }
  }
);
```

### Model Retraining

To retrain the model with new data:

```bash
# Add new emails to data/phishing_email.csv
# Format: email_text,label (0 or 1)

# Run training script
python train_model.py

# Models will be saved to models/ directory
# Restart API to load new models
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "Office is not defined" error in add-in**
- Ensure Office.js script is loaded before your code
- Wait for `Office.onReady()` before using Office APIs
- Check browser console for errors (F12)

**2. "CORS error" when calling API**
- Verify `ALLOWED_ORIGINS` includes your add-in domain
- Check API is running on correct port
- Ensure HTTPS is used in production

**3. "Model file not found"**
- Run `python train_model.py` to generate models
- Check `models/` directory exists
- Verify file paths in `main.py`

**4. "SSL certificate error" in Outlook**
- Reinstall dev certificates: `npx office-addin-dev-certs install`
- Check certificate is in Windows Trusted Root store
- Restart Outlook after installing certificates

**5. Add-in doesn't appear in Outlook**
- Validate manifest: `npm run validate`
- Check manifest is sideloaded: Outlook → Get Add-ins → My Add-ins
- Restart Outlook
- Clear Office cache (Windows: `%LOCALAPPDATA%\Microsoft\Office\16.0\Wef`)

**6. "Model accuracy is low"**
- Ensure dataset is balanced (roughly 50/50 phishing/legitimate)
- Check for data quality issues (duplicates, mislabeled)
- Try different algorithms (Random Forest, SVM)
- Increase features: `TfidfVectorizer(max_features=10000)`

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Write clear, commented code
- Add unit tests for new features
- Update documentation (README, docstrings)
- Follow PEP 8 style guide for Python
- Use ES6+ for JavaScript

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Prof. Fredrick Ogore** - Project supervisor and mentor
- **USIU-A** - Providing resources and support
- **Dataset Sources:**
  - [Kaggle Phishing Email Dataset](https://www.kaggle.com/datasets/subhajournal/phishingemails)
  - Enron Email Dataset
  - SpamAssassin Public Corpus
  - CEAS 2008 Dataset
- **Inspiration:**
  - Microsoft Defender for Office 365
  - Google Workspace Security
  - Academic research on phishing detection

---

## 📧 Contact

**Natalie Adhiambo Odhiambo**  
Applied Computer Technology Student  
United States International University - Africa

- GitHub: [@natadh](https://github.com/natadh)
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📈 Project Stats

![GitHub stars](https://img.shields.io/github/stars/natadh/Phishing_detector_api?style=social)
![GitHub forks](https://img.shields.io/github/forks/natadh/Phishing_detector_api?style=social)
![GitHub issues](https://img.shields.io/github/issues/natadh/Phishing_detector_api)
![GitHub license](https://img.shields.io/github/license/natadh/Phishing_detector_api)

---

## 🎓 Academic Context

This project was developed as part of the **APT4900: Final Project** course for the Bachelor of Science degree in Applied Computer Technology at United States International University - Africa (USIU-A), Spring Semester 2025.

**Project Objectives:**
- Demonstrate practical application of machine learning in cybersecurity
- Address real-world phishing threats with AI-powered solutions
- Integrate ML models with production-ready software architecture
- Deploy functional software accessible to end-users

**Key Learnings:**
- End-to-end ML pipeline (data collection → training → deployment)
- RESTful API design and implementation
- Microsoft Office extensibility platform (Office.js)
- Production deployment and DevOps practices
- User-centered design for cybersecurity tools

---

## ⭐ If you found this project helpful, please consider giving it a star!

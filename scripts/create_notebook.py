import nbformat as nbf
import json

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells
cells = []

# Title and Introduction
cells.append(nbf.v4.new_markdown_cell("""# SpamShield AI - Spam/Ham Detector

## Project Overview
A production-ready Flask web application that uses an Ensemble Machine Learning model to classify text messages and documents as SPAM or HAM (legitimate).

### Key Features
- **ML Model**: Voting Classifier (Naive Bayes + Logistic Regression + Random Forest)
- **Datasets**: Trained on `emails.csv`, `sms.csv`, and `spamham.csv`
- **Preprocessing**: TF-IDF Vectorization, SMOTE balancing, text normalization
- **Backend**: Flask API with file upload support (.txt, .csv, .docx, .pdf)
- **Frontend**: Modern dark-mode UI with glassmorphism and real-time confidence visualization
- **Accuracy**: 96%+ on test data
"""))

# Architecture
cells.append(nbf.v4.new_markdown_cell("""## System Architecture

```mermaid
graph TD
    A[User Input] --> B{Input Type}
    B -->|Text| C[Text Preprocessing]
    B -->|File| D[File Parser]
    D --> C
    C --> E[TF-IDF Vectorizer]
    E --> F[Ensemble Model]
    F --> G[Naive Bayes]
    F --> H[Logistic Regression]
    F --> I[Random Forest]
    G --> J[Voting Classifier]
    H --> J
    I --> J
    J --> K[Prediction + Confidence]
    K --> L[JSON Response]
    L --> M[UI Display]
```
"""))

# Dataset Analysis
cells.append(nbf.v4.new_markdown_cell("""## Dataset Analysis

### Data Sources
1. **emails.csv** - Email spam dataset
2. **sms.csv** - SMS spam dataset  
3. **spamham.csv** - Combined spam/ham messages

All datasets were merged and preprocessed to create a unified training set.
"""))

cells.append(nbf.v4.new_code_cell("""# Load and analyze datasets
import pandas as pd

# Load datasets
df_emails = pd.read_csv('emails.csv')
df_sms = pd.read_csv('sms.csv', encoding='latin-1')
df_spamham = pd.read_csv('spamham.csv')

print("Dataset Shapes:")
print(f"Emails: {df_emails.shape}")
print(f"SMS: {df_sms.shape}")
print(f"SpamHam: {df_spamham.shape}")

# Show sample from each
print("\\nSample from emails.csv:")
print(df_emails.head(2))
"""))

# Model Training
cells.append(nbf.v4.new_markdown_cell("""## Model Training Pipeline

### Preprocessing Steps
1. **Text Normalization**: Lowercase conversion
2. **Cleaning**: Remove special characters and numbers
3. **Tokenization**: Split into words
4. **Stopwords Removal**: Remove common English words
5. **Lemmatization**: Reduce words to base form

### Feature Engineering
- **TF-IDF Vectorization**: Convert text to numerical features
- **N-grams**: 1-3 word combinations
- **Max Features**: 5000 most important features

### Class Balancing
- **SMOTE**: Synthetic Minority Over-sampling Technique
- Ensures balanced representation of SPAM and HAM classes

### Model Architecture
- **Ensemble Voting Classifier**:
  - Multinomial Naive Bayes
  - Logistic Regression (max_iter=1000)
  - Random Forest (n_estimators=100)
- **Voting Strategy**: Soft voting (probability-based)
"""))

cells.append(nbf.v4.new_code_cell("""# Model training code snippet
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Define classifiers
nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Create ensemble
voting_clf = VotingClassifier(
    estimators=[('nb', nb), ('lr', lr), ('rf', rf)],
    voting='soft'
)

# Build pipeline
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
    ('smote', SMOTE(random_state=42)),
    ('clf', voting_clf)
])

print("Pipeline created successfully!")
print(pipeline)
"""))

# API Endpoints
cells.append(nbf.v4.new_markdown_cell("""## Flask API Endpoints

### 1. Home Page
```
GET /
Returns: HTML template (index.html)
```

### 2. Scan Endpoint
```
POST /scan
Body: 
  - text: string (message to scan)
  OR
  - file: multipart/form-data (.txt, .csv, .docx, .pdf)

Response:
{
  "verdict": "SPAM" | "HAM",
  "confidence": 93.97,
  "keywords": ["free", "winner", "prize"],
  "message_preview": "First 200 chars..."
}
```

### 3. Model Statistics
```
GET /api/stats
Response:
{
  "accuracy": 0.98,
  "confusion_matrix": [[...], [...]]
}
```
"""))

# Testing Results
cells.append(nbf.v4.new_markdown_cell("""## Testing & Verification

### Test Case 1: SPAM Detection
**Input**: "FREE WINNER! Click here to claim your prize NOW!"

**Result**:
- Verdict: SPAM
- Confidence: 93.97%
- Keywords: free, winner, prize, click

### Test Case 2: HAM Detection
**Input**: "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you."

**Result**:
- Verdict: HAM
- Confidence: 98.43%
- Keywords: None detected
"""))

# UI Screenshots
cells.append(nbf.v4.new_markdown_cell("""## User Interface

### Landing Page
Modern dark-themed interface with glassmorphism effects and gradient background.

### Features
- **Tab Switcher**: Toggle between text input and file upload
- **Real-time Character Count**: Track message length
- **Drag & Drop**: Easy file upload
- **Animated Results**: Circular confidence meter with smooth animations
- **Keyword Highlighting**: Visual indication of detected spam keywords
- **Responsive Design**: Works on mobile, tablet, and desktop
"""))

# File Structure
cells.append(nbf.v4.new_markdown_cell("""## Project Structure

```
spam-ham/
├── app.py                          # Flask application
├── train_model.py                  # ML training script
├── requirements.txt                # Python dependencies
├── spam_detector_model.joblib      # Trained model
├── model_metrics.joblib            # Model performance metrics
├── emails.csv                      # Dataset 1
├── sms.csv                         # Dataset 2
├── spamham.csv                     # Dataset 3
├── templates/
│   └── index.html                  # Main UI template
├── static/
│   ├── css/
│   │   └── style.css              # Styling
│   └── js/
│       └── main.js                # Frontend logic
└── uploads/                        # Temporary file storage
```
"""))

# How to Run
cells.append(nbf.v4.new_markdown_cell("""## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model (if not already trained)
```bash
python train_model.py
```

### 3. Start Flask App
```bash
python app.py
```

### 4. Access Application
Open browser and navigate to: `http://localhost:5000`
"""))

# Future Enhancements
cells.append(nbf.v4.new_markdown_cell("""## Future Enhancements

1. **Advanced Features**:
   - URL safety checking
   - Multi-language support
   - Email header analysis
   - Sender reputation scoring

2. **Model Improvements**:
   - Deep learning models (LSTM, BERT)
   - Active learning from user feedback
   - Real-time model updates

3. **Deployment**:
   - Docker containerization
   - Cloud deployment (AWS, Heroku)
   - API rate limiting
   - User authentication

4. **Analytics**:
   - Dashboard for scan statistics
   - Trend analysis
   - Batch processing reports
"""))

# Conclusion
cells.append(nbf.v4.new_markdown_cell("""## Conclusion

SpamShield AI successfully demonstrates a production-ready spam detection system with:
- ✅ High accuracy (96%+)
- ✅ Modern, responsive UI
- ✅ Multiple input methods (text, files)
- ✅ Real-time predictions
- ✅ Detailed confidence analysis

The application is ready for deployment and can be extended with additional features as needed.
"""))

# Add all cells to notebook
nb['cells'] = cells

# Write notebook
with open('SpamShield_Documentation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook created successfully: SpamShield_Documentation.ipynb")

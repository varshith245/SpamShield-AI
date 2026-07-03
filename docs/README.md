# SpamShield AI - Spam/Ham Detector 🛡️

A production-ready Flask web application that uses an Ensemble Machine Learning model to instantly classify text messages and documents as SPAM or HAM (legitimate).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25+-success.svg)

## ✨ Features

- **🤖 Advanced ML Model**: Ensemble Voting Classifier (Naive Bayes + Logistic Regression + Random Forest)
- **📊 High Accuracy**: 96%+ accuracy on test data
- **📁 Multiple Input Methods**: Text input or file upload (.txt, .csv, .docx, .pdf)
- **🎨 Modern UI**: Dark mode with glassmorphism effects and smooth animations
- **⚡ Real-time Analysis**: Instant predictions with confidence scores
- **🔍 Keyword Detection**: Highlights spam indicators in messages
- **📱 Responsive Design**: Works seamlessly on mobile, tablet, and desktop

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**
```bash
cd d:/PROJECTS/spam-ham
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model** (if not already trained)
```bash
python train_model.py
```
This will:
- Load and merge datasets (emails.csv, sms.csv, spamham.csv)
- Preprocess text data
- Train ensemble model with SMOTE balancing
- Save model artifacts (spam_detector_model.joblib, model_metrics.joblib)

4. **Start the Flask application**
```bash
python app.py
```

5. **Open your browser**
Navigate to: `http://localhost:5000`

## 📖 Usage

### Text Input
1. Click on the "Text Input" tab
2. Paste or type your message
3. Click "Scan Now"
4. View results with confidence score and detected keywords

### File Upload
1. Click on the "Upload File" tab
2. Drag & drop or browse for a file (.txt, .csv, .docx, .pdf)
3. Click "Scan Now"
4. View analysis results

## 🏗️ Architecture

```
User Input → Text Preprocessing → TF-IDF Vectorization → Ensemble Model → Prediction
                                                          ├─ Naive Bayes
                                                          ├─ Logistic Regression
                                                          └─ Random Forest
```

## 📂 Project Structure

```
spam-ham/
├── app.py                          # Flask application
├── train_model.py                  # ML training script
├── create_notebook.py              # Jupyter notebook generator
├── requirements.txt                # Dependencies
├── spam_detector_model.joblib      # Trained model
├── model_metrics.joblib            # Performance metrics
├── SpamShield_Documentation.ipynb  # Project documentation
├── emails.csv                      # Dataset 1
├── sms.csv                         # Dataset 2
├── spamham.csv                     # Dataset 3
├── templates/
│   └── index.html                  # Main UI
├── static/
│   ├── css/
│   │   └── style.css              # Styling
│   └── js/
│       └── main.js                # Frontend logic
└── uploads/                        # Temp file storage
```

## 🔧 API Endpoints

### `GET /`
Returns the main HTML interface

### `POST /scan`
Analyzes text or file for spam

**Request Body:**
- `text`: String (message to analyze)
- OR `file`: File upload (.txt, .csv, .docx, .pdf)

**Response:**
```json
{
  "verdict": "SPAM" | "HAM",
  "confidence": 93.97,
  "keywords": ["free", "winner", "prize"],
  "message_preview": "First 200 characters..."
}
```

### `GET /api/stats`
Returns model performance metrics

**Response:**
```json
{
  "accuracy": 0.98,
  "confusion_matrix": [[...], [...]]
}
```

## 🧪 Testing

### Example SPAM Messages
- "FREE WINNER! Click here to claim your prize NOW!"
- "Congratulations! You've won $1000. Call now!"
- "URGENT: Your account will be closed. Click this link immediately!"

### Example HAM Messages
- "Hey, are we still meeting for lunch tomorrow?"
- "Can you send me the project report by Friday?"
- "Thanks for your help with the presentation!"

## 🎯 Model Performance

- **Accuracy**: 96%+
- **Preprocessing**: Lowercase, remove special chars, stopwords removal, lemmatization
- **Feature Extraction**: TF-IDF with 1-3 n-grams, max 5000 features
- **Class Balancing**: SMOTE oversampling
- **Cross-validation**: 5-fold CV for robust evaluation

## 🛠️ Technologies Used

- **Backend**: Flask, Python
- **ML Libraries**: scikit-learn, imbalanced-learn, NLTK
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **File Parsing**: python-docx, PyPDF2
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Fonts**: Google Fonts (Inter)
- **Icons**: Font Awesome

## 📝 Documentation

For detailed documentation, open `SpamShield_Documentation.ipynb` in Jupyter Notebook:

```bash
jupyter notebook SpamShield_Documentation.ipynb
```

## 🔒 Privacy

- **No Data Storage**: Messages are processed in real-time and not stored
- **Local Processing**: All analysis happens on your server
- **Secure**: No external API calls for spam detection

## 🚀 Deployment

### Production Deployment
For production use with Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Optional)
Create a `Dockerfile` for containerized deployment:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Deep learning models (LSTM, BERT)
- Multi-language support
- URL safety checking
- Batch processing
- User feedback loop

## 📄 License

This project is open source and available for educational and commercial use.

## 👨‍💻 Author

Built with ❤️ using Flask and Machine Learning

---

**Note**: Make sure to train the model before running the application. The training process may take a few minutes depending on your hardware.

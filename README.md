# SpamShield AI - Advanced Spam Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A production-ready web application powered by Machine Learning for real-time spam message detection**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API Documentation](#-api-documentation) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

SpamShield AI is an intelligent spam detection system that leverages advanced machine learning algorithms to classify messages as spam or legitimate (ham) with high accuracy. Built with Flask and featuring a modern, responsive web interface, this application provides real-time spam detection capabilities for text messages.

### Key Highlights

- **High Accuracy**: 98.4% accuracy using ensemble machine learning models
- **Real-time Processing**: Instant spam detection with sub-second response times
- **Modern UI**: Beautiful, responsive dark-themed interface with smooth animations
- **Production Ready**: Fully functional web application with RESTful API endpoints
- **Privacy First**: All processing happens locally - no data is stored or transmitted

## ✨ Features

### Core Functionality
- **Text Analysis**: Paste or type messages directly for instant spam detection
- **ML-Powered Detection**: Ensemble classifier combining multiple algorithms
- **Confidence Scoring**: Detailed confidence percentages for each prediction
- **Keyword Detection**: Identifies common spam keywords in messages
- **Message Preview**: Shows analyzed message content with results

### Technical Features
- **Ensemble Model**: Voting classifier combining Naive Bayes, Logistic Regression, Decision Tree, and Random Forest
- **Advanced Preprocessing**: NLTK-based text preprocessing with lemmatization
- **Real-time API**: RESTful endpoints for programmatic access
- **Model Metrics**: Built-in performance statistics and accuracy tracking
- **Responsive Design**: Mobile-friendly interface that works on all devices

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://https://github.com/varshith245/SpamShield-AI.git
cd  D:\spam\SPAM_HAM
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data

The application requires NLTK data for text preprocessing. Run:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 4: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## 💻 Usage

### Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Paste or type the message you want to analyze in the text area
3. Click "Scan Now" to analyze the message
4. View the results including:
   - Spam/Ham verdict
   - Confidence percentage
   - Detected keywords
   - Message preview

### Example Messages

**Spam Example:**
```
FREE WINNER! Click here to claim your prize NOW! Limited time offer!
```

**Ham Example:**
```
Hey, are we still meeting for lunch tomorrow? Let me know the time.
```

## 🔌 API Documentation

### Endpoints

#### 1. Scan Text
**POST** `/scan`

Analyze text for spam detection.

**Request:**
```json
{
  "text": "Your message here"
}
```

**Response:**
```json
{
  "verdict": "SPAM",
  "confidence": 98.5,
  "keywords": ["free", "winner", "click"],
  "message_preview": "FREE WINNER! Click here..."
}
```

#### 2. Get Model Statistics
**GET** `/api/stats`

Retrieve model performance metrics.

**Response:**
```json
{
  "accuracy": 0.984,
  "confusion_matrix": [[...], [...]]
}
```

#### 3. Get Model Information
**GET** `/api/model-info`

Get detailed information about the loaded model.

**Response:**
```json
{
  "model_path": "models/artifact",
  "model_format": "pickle",
  "model_loaded": true,
  "metrics_loaded": true,
  "accuracy": 0.984
}
```

### Python API Usage

```python
from model_loader import get_model, predict_spam, detect_keywords

# Get model instance
model = get_model()

# Predict spam
result = predict_spam("Your message here")
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']}%")

# Detect keywords
keywords = detect_keywords("Your message here")
print(f"Keywords: {keywords}")
```

## 📁 Project Structure

```
SPAM_HAM/
├── app.py                  # Main Flask application
├── model_loader.py         # Model loading and prediction utilities
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
│
├── models/                # Trained ML models
│   ├── artifact           # Main model (pickle format)
│   ├── artifact_metrics   # Model performance metrics
│   └── artifacts/         # Additional model files
│
├── data/                  # Training datasets
│   ├── emails.csv         # Email dataset
│   ├── sms.csv            # SMS dataset
│   └── spamham.csv        # Combined dataset
│
├── notebooks/             # Jupyter notebooks
│   ├── Model_Training.ipynb
│   └── SpamShield_Documentation.ipynb
│
├── scripts/               # Utility scripts
│   ├── train_model.py     # Model training script
│   ├── compare_algorithms.py # Compare individual algorithm performance
│   ├── save_artifact.py   # Save model as artifact
│   └── save_best_model.py # Save best performing model
│
├── docs/                  # Additional documentation
│   ├── README.md
│   ├── MODEL_USAGE.md
│   └── ENSEMBLE_MODEL.md  # Ensemble model architecture details
│
├── static/                # Frontend static files
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── main.js        # Frontend logic
│
├── templates/             # HTML templates
│   └── index.html         # Main page
│
└── uploads/               # Temporary file uploads (auto-cleaned)
```

## 🔧 Development

### Training a New Model

To train a new model with your own dataset:

```bash
python scripts/train_model.py
```

The script will:
1. Load training data from `data/` directory
2. Preprocess and vectorize text
3. Train ensemble voting classifier
4. Evaluate performance
5. Save the best model

### Comparing Algorithm Performance

To compare individual algorithm accuracies:

```bash
python scripts/compare_algorithms.py
```

This script evaluates each algorithm separately and shows:
- Individual algorithm performance metrics
- Accuracy comparison ranking
- Ensemble improvement over best individual model

### Model Architecture

The system uses an **Ensemble Voting Classifier** combining three complementary algorithms:

- **Multinomial Naive Bayes**: Fast probabilistic classifier optimized for text
- **Logistic Regression**: Linear classification with regularization
- **Random Forest**: Non-linear ensemble classifier capturing complex patterns

**Performance**: 98.4% accuracy using soft voting mechanism

📖 **Detailed Documentation**: See [`docs/ENSEMBLE_MODEL.md`](docs/ENSEMBLE_MODEL.md) for comprehensive architecture details, algorithm comparison, and performance analysis.

### Running Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

## 📊 Model Performance

- **Accuracy**: 98.4%
- **Precision**: High precision for both spam and ham classification
- **Recall**: Excellent recall for spam detection
- **F1-Score**: Balanced performance metrics

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[ENSEMBLE_MODEL.md](docs/ENSEMBLE_MODEL.md)** - **Complete guide to the Ensemble Voting Classifier**
  - Architecture overview and pipeline structure
  - Detailed explanation of each algorithm (Naive Bayes, Logistic Regression, Random Forest)
  - Soft voting mechanism and how it works
  - Performance comparison and metrics
  - Feature engineering (TF-IDF, SMOTE)
  - Model training process and configuration
  - Algorithm selection rationale

- **[MODEL_USAGE.md](docs/MODEL_USAGE.md)** - Model usage examples and API reference
- **[README.md](docs/README.md)** - Complete setup and configuration guide

### Quick Links

- 🏗️ **Model Architecture**: [Ensemble Voting Classifier Details](docs/ENSEMBLE_MODEL.md)
- 🔧 **Usage Guide**: [How to Use the Model](docs/MODEL_USAGE.md)
- 📊 **Compare Algorithms**: Run `python scripts/compare_algorithms.py` to see individual algorithm performance

## 🛠️ Technologies Used

### Backend
- **Flask 3.0.0**: Web framework
- **Scikit-learn 1.7.2**: Machine learning library
- **NLTK 3.8.1**: Natural language processing
- **Pandas 2.3.3**: Data manipulation
- **NumPy 2.2.6**: Numerical computing

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript (ES6+)**: Interactive functionality
- **Font Awesome**: Icons
- **Google Fonts**: Typography (Inter, Space Grotesk)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Poornachandra-dh**

- GitHub: [@Poornachandra-dh](https://github.com/Poornachandra-dh)

## 🙏 Acknowledgments

- Scikit-learn community for excellent ML tools
- NLTK team for NLP resources
- Flask community for the web framework
- All contributors and users of this project

## 🚀 Deployment

### Deploy to Vercel

See [Deployment Guide](docs/DEPLOYMENT_VERCEL.md) for detailed instructions on deploying to Vercel.

**Quick Start:**
```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel
```

**Alternative Platforms:**
- **Railway** (Recommended for Flask) - Easy deployment
- **Render** - Free tier available
- **Fly.io** - Good for Python apps
- **Heroku** - Classic option

## 📧 Support

For issues, questions, or contributions, please open an issue on the [GitHub repository](https://github.com/Poornachandra-dh/SPAM_HAM/issues).

---

<div align="center">

**Made with ❤️ using Python, Flask, and Machine Learning**

⭐ Star this repo if you find it helpful!

</div>

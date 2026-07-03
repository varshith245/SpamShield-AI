import nbformat as nbf

# Create notebook
nb = nbf.v4.new_notebook()

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# SpamShield AI - Model Training Notebook

This notebook walks through the complete process of training the spam detection model, from data loading to model evaluation and deployment.

## Table of Contents
1. [Setup & Imports](#setup)
2. [Data Loading & Exploration](#data)
3. [Data Preprocessing](#preprocessing)
4. [Feature Engineering](#features)
5. [Model Training](#training)
6. [Model Evaluation](#evaluation)
7. [Model Persistence](#persistence)
8. [Testing Predictions](#testing)
"""))

# Setup
cells.append(nbf.v4.new_markdown_cell("""## 1. Setup & Imports

First, let's import all necessary libraries and set up our environment.
"""))

cells.append(nbf.v4.new_code_cell("""# Data manipulation
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, precision_recall_fscore_support)

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model persistence
import joblib

# Progress bar
from tqdm import tqdm

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ All libraries imported successfully!")
"""))

# Download NLTK data
cells.append(nbf.v4.new_markdown_cell("""### Download NLTK Data

We need to download stopwords and lemmatization data.
"""))

cells.append(nbf.v4.new_code_cell("""# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("✅ NLTK data downloaded!")
"""))

# Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 2. Data Loading & Exploration

We'll load three datasets:
- `emails.csv` - Email spam dataset
- `sms.csv` - SMS spam dataset
- `spamham.csv` - Combined spam/ham messages

All datasets will be merged into a single training set.
"""))

cells.append(nbf.v4.new_code_cell("""def load_datasets():
    \"\"\"Load and combine all datasets\"\"\"
    print("Loading datasets...")
    
    # Load datasets
    df_emails = pd.read_csv('emails.csv')
    df_sms = pd.read_csv('sms.csv', encoding='latin-1')
    df_spamham = pd.read_csv('spamham.csv')
    
    print(f"Emails dataset: {df_emails.shape}")
    print(f"SMS dataset: {df_sms.shape}")
    print(f"SpamHam dataset: {df_spamham.shape}")
    
    return df_emails, df_sms, df_spamham

df_emails, df_sms, df_spamham = load_datasets()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Explore Dataset Structure
"""))

cells.append(nbf.v4.new_code_cell("""# Check emails.csv structure
print("=" * 60)
print("EMAILS.CSV STRUCTURE")
print("=" * 60)
print(df_emails.head())
print(f"\\nColumns: {df_emails.columns.tolist()}")
print(f"Shape: {df_emails.shape}")
print(f"\\nMissing values:\\n{df_emails.isnull().sum()}")
"""))

cells.append(nbf.v4.new_code_cell("""# Check sms.csv structure
print("=" * 60)
print("SMS.CSV STRUCTURE")
print("=" * 60)
print(df_sms.head())
print(f"\\nColumns: {df_sms.columns.tolist()}")
print(f"Shape: {df_sms.shape}")
print(f"\\nMissing values:\\n{df_sms.isnull().sum()}")
"""))

cells.append(nbf.v4.new_code_cell("""# Check spamham.csv structure
print("=" * 60)
print("SPAMHAM.CSV STRUCTURE")
print("=" * 60)
print(df_spamham.head())
print(f"\\nColumns: {df_spamham.columns.tolist()}")
print(f"Shape: {df_spamham.shape}")
print(f"\\nMissing values:\\n{df_spamham.isnull().sum()}")
"""))

# Data Merging
cells.append(nbf.v4.new_markdown_cell("""### Merge Datasets

Standardize column names and combine all datasets.
"""))

cells.append(nbf.v4.new_code_cell("""def merge_datasets(df_emails, df_sms, df_spamham):
    \"\"\"Merge all datasets into a single DataFrame\"\"\"
    dfs = []
    
    # Process emails.csv
    if 'Message' in df_emails.columns and 'Label' in df_emails.columns:
        dfs.append(df_emails[['Message', 'Label']])
        print(f"✅ Added {len(df_emails)} samples from emails.csv")
    
    # Process sms.csv
    if 'Message' in df_sms.columns and 'Label' in df_sms.columns:
        dfs.append(df_sms[['Message', 'Label']])
        print(f"✅ Added {len(df_sms)} samples from sms.csv")
    
    # Process spamham.csv
    if 'Message' in df_spamham.columns and 'Label' in df_spamham.columns:
        dfs.append(df_spamham[['Message', 'Label']])
        print(f"✅ Added {len(df_spamham)} samples from spamham.csv")
    
    # Combine
    df = pd.concat(dfs, ignore_index=True)
    
    # Standardize labels
    df['Label'] = df['Label'].astype(str).str.lower().map({
        'spam': 1, 
        'ham': 0, 
        '1': 1, 
        '0': 0
    })
    
    # Drop missing values
    df.dropna(subset=['Message', 'Label'], inplace=True)
    
    print(f"\\n📊 Total samples: {len(df)}")
    print(f"📊 Class distribution:\\n{df['Label'].value_counts()}")
    
    return df

df = merge_datasets(df_emails, df_sms, df_spamham)
"""))

# EDA
cells.append(nbf.v4.new_markdown_cell("""### Exploratory Data Analysis
"""))

cells.append(nbf.v4.new_code_cell("""# Class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
df['Label'].value_counts().plot(kind='bar', ax=axes[0], color=['#00e676', '#ff1744'])
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class (0=HAM, 1=SPAM)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['HAM', 'SPAM'], rotation=0)

# Pie chart
df['Label'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                 colors=['#00e676', '#ff1744'], labels=['HAM', 'SPAM'])
axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

print(f"\\nClass Balance:")
print(f"HAM: {(df['Label']==0).sum()} ({(df['Label']==0).sum()/len(df)*100:.2f}%)")
print(f"SPAM: {(df['Label']==1).sum()} ({(df['Label']==1).sum()/len(df)*100:.2f}%)")
"""))

cells.append(nbf.v4.new_code_cell("""# Message length analysis
df['message_length'] = df['Message'].str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution by class
df[df['Label']==0]['message_length'].hist(bins=50, alpha=0.7, label='HAM', 
                                           color='#00e676', ax=axes[0])
df[df['Label']==1]['message_length'].hist(bins=50, alpha=0.7, label='SPAM', 
                                           color='#ff1744', ax=axes[0])
axes[0].set_title('Message Length Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Message Length (characters)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Box plot
df.boxplot(column='message_length', by='Label', ax=axes[1])
axes[1].set_title('Message Length by Class', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Class (0=HAM, 1=SPAM)')
axes[1].set_ylabel('Message Length')

plt.tight_layout()
plt.show()

print(f"\\nMessage Length Statistics:")
print(df.groupby('Label')['message_length'].describe())
"""))

# Preprocessing
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Preprocessing

We'll apply the following preprocessing steps:
1. Convert to lowercase
2. Remove special characters and numbers
3. Tokenize
4. Remove stopwords
5. Lemmatize
"""))

cells.append(nbf.v4.new_code_cell("""def preprocess_text(text):
    \"\"\"
    Preprocess text for ML model
    
    Steps:
    1. Lowercase
    2. Remove special chars
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    \"\"\"
    # Lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Test preprocessing
sample_text = "FREE WINNER! Click here to claim your $1000 prize NOW!!!"
print(f"Original: {sample_text}")
print(f"Processed: {preprocess_text(sample_text)}")
"""))

cells.append(nbf.v4.new_code_cell("""# Apply preprocessing to all messages
print("Preprocessing messages...")
tqdm.pandas(desc="Processing")

df['clean_message'] = df['Message'].progress_apply(preprocess_text)

print("\\n✅ Preprocessing complete!")
print(f"\\nSample processed messages:")
print(df[['Message', 'clean_message']].head(3))
"""))

# Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 4. Feature Engineering

We'll use TF-IDF (Term Frequency-Inverse Document Frequency) to convert text to numerical features.

**Parameters:**
- `max_features=5000`: Keep top 5000 most important features
- `ngram_range=(1, 3)`: Use 1-grams, 2-grams, and 3-grams
"""))

cells.append(nbf.v4.new_code_cell("""# Prepare data
X = df['clean_message']
y = df['Label']

print(f"Features (X): {X.shape}")
print(f"Labels (y): {y.shape}")
print(f"\\nLabel distribution:\\n{y.value_counts()}")
"""))

cells.append(nbf.v4.new_code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\\nTraining set distribution:\\n{y_train.value_counts()}")
print(f"\\nTest set distribution:\\n{y_test.value_counts()}")
"""))

# Model Training
cells.append(nbf.v4.new_markdown_cell("""## 5. Model Training

We'll train an **Ensemble Voting Classifier** combining:
1. **Multinomial Naive Bayes** - Fast, works well with text
2. **Logistic Regression** - Linear classifier
3. **Random Forest** - Non-linear ensemble

The pipeline includes:
- **TF-IDF Vectorization**: Convert text to features
- **SMOTE**: Balance classes by oversampling minority class
- **Voting Classifier**: Combine predictions from all models
"""))

cells.append(nbf.v4.new_code_cell("""# Define individual classifiers
nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Create ensemble
voting_clf = VotingClassifier(
    estimators=[
        ('nb', nb),
        ('lr', lr),
        ('rf', rf)
    ],
    voting='soft'  # Use probability-based voting
)

print("✅ Classifiers defined:")
print(f"  1. Multinomial Naive Bayes")
print(f"  2. Logistic Regression (max_iter=1000)")
print(f"  3. Random Forest (n_estimators=100)")
print(f"\\n✅ Ensemble: Voting Classifier (soft voting)")
"""))

cells.append(nbf.v4.new_code_cell("""# Build complete pipeline
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
    ('smote', SMOTE(random_state=42)),
    ('clf', voting_clf)
])

print("✅ Pipeline created:")
print(pipeline)
"""))

cells.append(nbf.v4.new_code_cell("""# Train model
print("🚀 Training model... (this may take a few minutes)")
print("-" * 60)

import time
start_time = time.time()

pipeline.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"\\n✅ Training complete!")
print(f"⏱️  Training time: {training_time:.2f} seconds")
"""))

# Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 6. Model Evaluation

Let's evaluate our model on the test set.
"""))

cells.append(nbf.v4.new_code_cell("""# Make predictions
print("Making predictions on test set...")
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

print("✅ Predictions complete!")
"""))

cells.append(nbf.v4.new_code_cell("""# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print("=" * 60)
print("MODEL PERFORMANCE METRICS")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("=" * 60)
"""))

cells.append(nbf.v4.new_code_cell("""# Detailed classification report
print("\\nDetailed Classification Report:")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM']))
"""))

cells.append(nbf.v4.new_code_cell("""# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=['HAM', 'SPAM'], 
            yticklabels=['HAM', 'SPAM'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.show()

print(f"\\nConfusion Matrix:")
print(f"True Negatives (HAM predicted as HAM): {cm[0][0]}")
print(f"False Positives (HAM predicted as SPAM): {cm[0][1]}")
print(f"False Negatives (SPAM predicted as HAM): {cm[1][0]}")
print(f"True Positives (SPAM predicted as SPAM): {cm[1][1]}")
"""))

# Model Persistence
cells.append(nbf.v4.new_markdown_cell("""## 7. Model Persistence

Save the trained model and metrics for deployment.
"""))

cells.append(nbf.v4.new_code_cell("""# Save model
joblib.dump(pipeline, 'spam_detector_model.joblib')
print("✅ Model saved: spam_detector_model.joblib")

# Save metrics
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': cm.tolist()
}
joblib.dump(metrics, 'model_metrics.joblib')
print("✅ Metrics saved: model_metrics.joblib")

# Check file sizes
import os
model_size = os.path.getsize('spam_detector_model.joblib') / (1024 * 1024)
print(f"\\nModel file size: {model_size:.2f} MB")
"""))

# Testing
cells.append(nbf.v4.new_markdown_cell("""## 8. Testing Predictions

Let's test our model with some sample messages.
"""))

cells.append(nbf.v4.new_code_cell("""def predict_spam(message, model=pipeline):
    \"\"\"Predict if a message is spam or ham\"\"\"
    # Preprocess
    clean_msg = preprocess_text(message)
    
    # Predict
    prediction = model.predict([clean_msg])[0]
    proba = model.predict_proba([clean_msg])[0]
    
    label = "SPAM" if prediction == 1 else "HAM"
    confidence = proba[1] if prediction == 1 else proba[0]
    
    return label, confidence * 100

# Test function
test_message = "FREE WINNER! Claim your prize now!"
label, conf = predict_spam(test_message)
print(f"Message: {test_message}")
print(f"Prediction: {label} ({conf:.2f}% confidence)")
"""))

cells.append(nbf.v4.new_code_cell("""# Test with multiple examples
test_messages = [
    "FREE WINNER! Click here to claim your $1000 prize NOW!!!",
    "Congratulations! You've won a free iPhone. Call now!",
    "URGENT: Your account will be closed. Click this link immediately!",
    "Hey, are we still meeting for lunch tomorrow?",
    "Can you send me the project report by Friday?",
    "Thanks for your help with the presentation!",
    "Meeting at 3pm in conference room B",
    "Win big money fast! No risk! Click here!"
]

print("=" * 80)
print("TESTING PREDICTIONS")
print("=" * 80)

for msg in test_messages:
    label, conf = predict_spam(msg)
    emoji = "🚨" if label == "SPAM" else "✅"
    print(f"\\n{emoji} {label} ({conf:.2f}%)")
    print(f"   Message: {msg[:60]}{'...' if len(msg) > 60 else ''}")
"""))

# Summary
cells.append(nbf.v4.new_markdown_cell("""## Summary

### Model Architecture
- **Pipeline**: TF-IDF → SMOTE → Voting Classifier
- **Classifiers**: Naive Bayes + Logistic Regression + Random Forest
- **Features**: 5000 TF-IDF features with 1-3 n-grams

### Performance
Model performance metrics will be displayed after training completion.

### Deployment
The model has been saved and is ready for deployment in the Flask application.

Files created:
- `spam_detector_model.joblib` - Trained model
- `model_metrics.joblib` - Performance metrics

### Next Steps
1. Integrate model with Flask app (`app.py`)
2. Test with real-world data
3. Monitor performance and retrain as needed
4. Consider advanced features (URL detection, sender analysis)
"""))

# Add all cells
nb['cells'] = cells

# Save notebook
with open('Model_Training.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("✅ Training notebook created: Model_Training.ipynb")

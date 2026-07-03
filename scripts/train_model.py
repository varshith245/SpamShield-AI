import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data():
    print("Loading datasets...")
    try:
        # Load datasets
        df_emails = pd.read_csv('emails.csv')
        df_sms = pd.read_csv('sms.csv', encoding='latin-1') # sms often has encoding issues
        df_spamham = pd.read_csv('spamham.csv')
        
        # Standardize columns
        # emails.csv: likely has 'text' and 'spam' (1/0) or similar. Let's inspect or assume based on head.
        # Based on previous `head` output:
        # emails.csv: ,Label,Message,label_num -> Label (spam/ham), Message
        # sms.csv: Label,Message
        # spamham.csv: Label,Message
        
        # We need 'message' and 'label' (0 for ham, 1 for spam)
        
        dfs = []
        
        # Process emails.csv
        if 'Message' in df_emails.columns and 'Label' in df_emails.columns:
            dfs.append(df_emails[['Message', 'Label']])
        
        # Process sms.csv
        if 'Message' in df_sms.columns and 'Label' in df_sms.columns:
            dfs.append(df_sms[['Message', 'Label']])
            
        # Process spamham.csv
        if 'Message' in df_spamham.columns and 'Label' in df_spamham.columns:
            dfs.append(df_spamham[['Message', 'Label']])
            
        if not dfs:
            raise ValueError("No valid datasets found.")
            
        df = pd.concat(dfs, ignore_index=True)
        
        # Clean labels
        # Map 'spam' -> 1, 'ham' -> 0
        df['Label'] = df['Label'].astype(str).str.lower().map({'spam': 1, 'ham': 0, '1': 1, '0': 0})
        
        # Drop rows with NaN labels or messages
        df.dropna(subset=['Message', 'Label'], inplace=True)
        
        print(f"Total samples loaded: {len(df)}")
        print(f"Class distribution:\n{df['Label'].value_counts()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_text(text):
    # Lowercase
    text = str(text).lower()
    # Remove special characters and numbers (keep only letters)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize (split by space)
    words = text.split()
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def train_model():
    df = load_data()
    if df is None:
        return

    print("Preprocessing text...")
    df['clean_message'] = df['Message'].apply(preprocess_text)
    
    X = df['clean_message']
    y = df['Label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Building pipeline...")
    
    # Define classifiers
    nb = MultinomialNB()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Ensemble
    voting_clf = VotingClassifier(estimators=[
        ('nb', nb),
        ('lr', lr),
        ('rf', rf)
    ], voting='soft')
    
    # Pipeline with SMOTE
    # Note: SMOTE should be applied only on training data. 
    # Imbalanced-learn pipeline handles this automatically (sampling only during fit)
    pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
        ('smote', SMOTE(random_state=42)),
        ('clf', voting_clf)
    ])
    
    print("Training model (this may take a while)...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    print("Saving artifacts...")
    joblib.dump(pipeline, 'spam_detector_model.joblib')
    
    # Save metadata/metrics
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    joblib.dump(metrics, 'model_metrics.joblib')
    
    print("Done!")

if __name__ == "__main__":
    train_model()

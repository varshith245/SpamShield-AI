"""
Compare individual algorithm performance for spam detection.
This script evaluates each algorithm separately to identify which performs best.
"""

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def load_data():
    """Load and combine all datasets"""
    print("Loading datasets...")
    try:
        # Change to data directory
        os.chdir('data') if os.path.exists('data') else None
        
        df_emails = pd.read_csv('emails.csv')
        df_sms = pd.read_csv('sms.csv', encoding='latin-1')
        df_spamham = pd.read_csv('spamham.csv')
        
        # Change back to parent directory
        os.chdir('..') if os.path.exists('data') else None
        
        dfs = []
        
        if 'Message' in df_emails.columns and 'Label' in df_emails.columns:
            dfs.append(df_emails[['Message', 'Label']])
        if 'Message' in df_sms.columns and 'Label' in df_sms.columns:
            dfs.append(df_sms[['Message', 'Label']])
        if 'Message' in df_spamham.columns and 'Label' in df_spamham.columns:
            dfs.append(df_spamham[['Message', 'Label']])
            
        if not dfs:
            raise ValueError("No valid datasets found.")
            
        df = pd.concat(dfs, ignore_index=True)
        df['Label'] = df['Label'].astype(str).str.lower().map({'spam': 1, 'ham': 0, '1': 1, '0': 0})
        df.dropna(subset=['Message', 'Label'], inplace=True)
        
        print(f"Total samples: {len(df)}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_text(text):
    """Preprocess text for ML model"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def evaluate_algorithm(name, pipeline, X_train, X_test, y_train, y_test):
    """Train and evaluate a single algorithm"""
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print('='*60)
    
    import time
    start_time = time.time()
    
    # Train
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predict
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n✅ Training Time: {training_time:.2f} seconds")
    print(f"\n📊 Performance Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Negatives (HAM→HAM):  {cm[0][0]}")
    print(f"   False Positives (HAM→SPAM): {cm[0][1]}")
    print(f"   False Negatives (SPAM→HAM): {cm[1][0]}")
    print(f"   True Positives (SPAM→SPAM): {cm[1][1]}")
    
    return {
        'name': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time,
        'confusion_matrix': cm
    }

def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess
    print("\nPreprocessing text...")
    df['clean_message'] = df['Message'].apply(preprocess_text)
    
    X = df['clean_message']
    y = df['Label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # TF-IDF Vectorizer (shared)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)
    
    results = []
    
    # 1. Multinomial Naive Bayes
    nb_pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
        ('smote', SMOTE(random_state=42)),
        ('clf', MultinomialNB())
    ])
    results.append(evaluate_algorithm(
        "Multinomial Naive Bayes", 
        nb_pipeline, 
        X_train, X_test, y_train, y_test
    ))
    
    # 2. Logistic Regression
    lr_pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    results.append(evaluate_algorithm(
        "Logistic Regression", 
        lr_pipeline, 
        X_train, X_test, y_train, y_test
    ))
    
    # 3. Decision Tree
    dt_pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
        ('smote', SMOTE(random_state=42)),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])
    results.append(evaluate_algorithm(
        "Decision Tree", 
        dt_pipeline, 
        X_train, X_test, y_train, y_test
    ))
    
    # 4. Random Forest
    rf_pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    results.append(evaluate_algorithm(
        "Random Forest", 
        rf_pipeline, 
        X_train, X_test, y_train, y_test
    ))
    
    # 5. Ensemble Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ],
        voting='soft'
    )
    ensemble_pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
        ('smote', SMOTE(random_state=42)),
        ('clf', voting_clf)
    ])
    results.append(evaluate_algorithm(
        "Ensemble Voting Classifier", 
        ensemble_pipeline, 
        X_train, X_test, y_train, y_test
    ))
    
    # Summary
    print(f"\n\n{'='*80}")
    print("ALGORITHM COMPARISON SUMMARY")
    print('='*80)
    
    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Algorithm':<35} {'Accuracy':<12} {'F1-Score':<12} {'Time (s)':<10}")
    print('-'*80)
    
    for i, result in enumerate(results_sorted, 1):
        print(f"{i:<5} {result['name']:<35} {result['accuracy']*100:>6.2f}%     "
              f"{result['f1_score']:>6.4f}     {result['training_time']:>6.2f}")
    
    print(f"\n{'='*80}")
    print("🏆 BEST PERFORMING ALGORITHM:")
    print(f"   {results_sorted[0]['name']}")
    print(f"   Accuracy: {results_sorted[0]['accuracy']*100:.2f}%")
    print(f"   F1-Score: {results_sorted[0]['f1_score']:.4f}")
    print('='*80)
    
    # Improvement over individual algorithms
    if len(results_sorted) > 1:
        best_individual = max([r for r in results if r['name'] != 'Ensemble Voting Classifier'], 
                             key=lambda x: x['accuracy'])
        ensemble = next(r for r in results if r['name'] == 'Ensemble Voting Classifier')
        
        improvement = ensemble['accuracy'] - best_individual['accuracy']
        print(f"\n📈 Ensemble Improvement:")
        print(f"   Best Individual: {best_individual['name']} ({best_individual['accuracy']*100:.2f}%)")
        print(f"   Ensemble: {ensemble['accuracy']*100:.2f}%")
        print(f"   Improvement: +{improvement*100:.2f}%")

if __name__ == "__main__":
    main()


"""
SpamShield AI - Model Loader Utility (Using Pickle Artifact)

This module handles loading the trained spam detection model from 'artifact' file
and provides a clean interface for making predictions.
"""

import os
import pickle
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class SpamDetectorModel:
    """
    Wrapper class for the spam detection model.
    Handles model loading, preprocessing, and predictions.
    """
    
    def __init__(self, model_path='models/spam_detector_model.joblib', metrics_path='models/model_metrics.joblib'):
        """
        Initialize the spam detector model.
        
        Args:
            model_path (str): Path to the trained model file.
            metrics_path (str): Path to the metrics file.
        """
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.model = None
        self.metrics = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = self._load_stopwords()
        
        # Load model
        self._load_model()
    
    def _load_stopwords(self):
        """Load NLTK stopwords, downloading resources if necessary."""
        try:
            return set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            return set(stopwords.words('english'))

    def _resolve_path(self, path):
        """Resolve a local path relative to this module."""
        if os.path.isabs(path):
            return path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, path)

    def _load_model(self):
        """Load the trained model and metrics from disk."""
        try:
            model_path = self._resolve_path(self.model_path)
            metrics_path = self._resolve_path(self.metrics_path)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Prefer joblib format for the current model artifact.
            try:
                self.model = joblib.load(model_path)
            except Exception as joblib_error:
                print(f"joblib.load failed for {model_path}: {joblib_error}")
                print("Attempting pickle fallback for legacy artifact format...")
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)

            if os.path.exists(metrics_path):
                try:
                    self.metrics = joblib.load(metrics_path)
                except Exception:
                    with open(metrics_path, 'rb') as f:
                        self.metrics = pickle.load(f)
                print(f"Metrics loaded successfully from {metrics_path}")
            else:
                print(f"Warning: Metrics file not found: {metrics_path}")
                self.metrics = {'accuracy': 0, 'confusion_matrix': []}

        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_text(self, text):
        """
        Preprocess text for model prediction.
        
        Steps:
        1. Convert to lowercase
        2. Remove special characters and numbers
        3. Tokenize
        4. Remove stopwords
        5. Lemmatize
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords and lemmatize
        words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words
        ]
        
        return ' '.join(words)
    
    def predict(self, text):
        """
        Predict if text is spam or ham.
        
        Args:
            text (str): Text to classify
            
        Returns:
            dict: Prediction results containing:
                - verdict (str): "SPAM" or "HAM"
                - confidence (float): Confidence percentage (0-100)
                - prediction (int): Raw prediction (0 or 1)
                - probabilities (list): [prob_ham, prob_spam]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot make predictions.")
        
        # Preprocess
        clean_text = self.preprocess_text(text)
        
        # Predict
        prediction = self.model.predict([clean_text])[0]
        probabilities = self.model.predict_proba([clean_text])[0]
        
        # Format results
        verdict = "SPAM" if prediction == 1 else "HAM"
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        return {
            'verdict': verdict,
            'confidence': round(confidence * 100, 2),
            'prediction': int(prediction),
            'probabilities': probabilities.tolist()
        }
    
    def detect_spam_keywords(self, text):
        """
        Detect common spam keywords in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: List of detected spam keywords
        """
        spam_keywords = [
            'free', 'winner', 'prize', 'money', 'cash', 'urgent', 
            'click', 'link', 'offer', 'buy', 'congratulations', 
            'claim', 'won', 'lottery', 'limited', 'act now', 
            'expire', 'bonus', 'guarantee', 'risk free'
        ]
        
        text_lower = text.lower()
        detected = [word for word in spam_keywords if word in text_lower]
        
        return detected
    
    def get_metrics(self):
        """
        Get model performance metrics.
        
        Returns:
            dict: Model metrics including accuracy, confusion matrix, etc.
        """
        return self.metrics
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            'model_path': self.model_path,
            'model_format': 'pickle',
            'model_loaded': self.model is not None,
            'metrics_loaded': self.metrics is not None,
            'accuracy': self.metrics.get('accuracy', 0) if self.metrics else 0
        }


# Singleton instance
_model_instance = None

def get_model():
    """
    Get the singleton model instance.
    
    Returns:
        SpamDetectorModel: The loaded model instance
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = SpamDetectorModel()
    
    return _model_instance


# Convenience functions
def predict_spam(text):
    """
    Convenience function to predict spam.
    
    Args:
        text (str): Text to classify
        
    Returns:
        dict: Prediction results
    """
    model = get_model()
    return model.predict(text)


def detect_keywords(text):
    """
    Convenience function to detect spam keywords.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        list: Detected spam keywords
    """
    model = get_model()
    return model.detect_spam_keywords(text)


if __name__ == "__main__":
    # Test the model loader
    print("=" * 60)
    print("TESTING SPAM DETECTOR MODEL (PICKLE ARTIFACT)")
    print("=" * 60)
    
    # Initialize model
    model = get_model()
    print(f"\nModel Info: {model.get_model_info()}")
    
    # Test predictions
    test_messages = [
        "FREE WINNER! Click here to claim your prize NOW!",
        "Hey, are we still meeting for lunch tomorrow?"
    ]
    
    print("\n" + "=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)
    
    for msg in test_messages:
        result = predict_spam(msg)
        keywords = detect_keywords(msg)
        
        print(f"\nMessage: {msg}")
        print(f"Verdict: {result['verdict']} ({result['confidence']}% confidence)")
        print(f"Keywords: {keywords if keywords else 'None detected'}")

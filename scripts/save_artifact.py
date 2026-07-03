"""
Save the trained SpamShield AI model as 'artifact' file in pickle format.
This script loads the existing model and saves it with the new filename.
"""

import pickle
import joblib
import os

def save_model_as_artifact():
    """
    Load the existing model and save it as 'artifact' file using pickle.
    """
    print("=" * 60)
    print("SAVING MODEL AS ARTIFACT (PICKLE FORMAT)")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('spam_detector_model.joblib'):
        print("❌ Error: spam_detector_model.joblib not found!")
        print("Please train the model first using: python train_model.py")
        return False
    
    try:
        # Load the existing model
        print("\n📂 Loading existing model...")
        model = joblib.load('spam_detector_model.joblib')
        print("✅ Model loaded successfully")
        
        # Save as 'artifact' using pickle
        print("\n💾 Saving model as 'artifact' file (pickle format)...")
        with open('artifact', 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Get file size
        file_size = os.path.getsize('artifact') / (1024 * 1024)
        print(f"✅ Model saved successfully as 'artifact'")
        print(f"📊 File size: {file_size:.2f} MB")
        
        # Also save metrics if they exist
        if os.path.exists('model_metrics.joblib'):
            print("\n💾 Saving metrics as 'artifact_metrics' (pickle format)...")
            metrics = joblib.load('model_metrics.joblib')
            with open('artifact_metrics', 'wb') as f:
                pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("✅ Metrics saved successfully as 'artifact_metrics'")
        
        # Verify the saved model
        print("\n🔍 Verifying saved model...")
        with open('artifact', 'rb') as f:
            loaded_model = pickle.load(f)
        print("✅ Model verification successful!")
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print("Model saved as: artifact")
        print("Metrics saved as: artifact_metrics")
        print("\nYou can now use these files in your Flask application.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = save_model_as_artifact()
    exit(0 if success else 1)

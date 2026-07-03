"""
Save the best performing model as 'best_model.pkl' in models/artifacts/ folder.
"""

import pickle
import joblib
import os

def save_best_model():
    """
    Load the existing model and save it as 'best_model.pkl' in models/artifacts/.
    """
    print("=" * 60)
    print("SAVING BEST MODEL AS best_model.pkl")
    print("=" * 60)
    
    # Create artifacts directory
    os.makedirs('models/artifacts', exist_ok=True)
    print("\n✅ Created models/artifacts/ directory")
    
    # Check if model exists
    model_path = 'models/artifact'
    if not os.path.exists(model_path):
        # Try joblib version
        model_path = 'models/spam_detector_model.joblib'
        if not os.path.exists(model_path):
            print("❌ Error: No model found!")
            return False
    
    try:
        # Load the model
        print(f"\n📂 Loading model from {model_path}...")
        if model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        print("✅ Model loaded successfully")
        
        # Save as best_model.pkl
        output_path = 'models/artifacts/best_model.pkl'
        print(f"\n💾 Saving as {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Best model saved successfully!")
        print(f"📊 File size: {file_size:.2f} MB")
        
        # Also save metrics if available
        metrics_path = 'models/artifact_metrics'
        if not os.path.exists(metrics_path):
            metrics_path = 'models/model_metrics.joblib'
        
        if os.path.exists(metrics_path):
            print(f"\n💾 Saving metrics as best_model_metrics.pkl...")
            if metrics_path.endswith('.joblib'):
                metrics = joblib.load(metrics_path)
            else:
                with open(metrics_path, 'rb') as f:
                    metrics = pickle.load(f)
            
            with open('models/artifacts/best_model_metrics.pkl', 'wb') as f:
                pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("✅ Metrics saved successfully!")
        
        # Verify
        print("\n🔍 Verifying saved model...")
        with open(output_path, 'rb') as f:
            loaded_model = pickle.load(f)
        print("✅ Model verification successful!")
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Best model saved to: {output_path}")
        print("Ready to use in Flask application!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = save_best_model()
    exit(0 if success else 1)

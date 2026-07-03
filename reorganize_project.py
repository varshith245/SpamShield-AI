"""
Project Reorganization Script
Organizes SpamShield AI into a clean, professional structure.
"""

import os
import shutil

def create_directory_structure():
    """Create the organized directory structure."""
    
    directories = [
        'models',           # Trained models and artifacts
        'notebooks',        # Jupyter notebooks
        'scripts',          # Utility scripts
        'docs',            # Documentation
        'data',            # Dataset files
        'static/css',      # Already exists
        'static/js',       # Already exists
        'templates',       # Already exists
        'uploads',         # Already exists
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}/")
    
    return True

def organize_files():
    """Move files to their appropriate directories."""
    
    file_moves = {
        # Models
        'artifact': 'models/artifact',
        'artifact_metrics': 'models/artifact_metrics',
        'spam_detector_model.joblib': 'models/spam_detector_model.joblib',
        'model_metrics.joblib': 'models/model_metrics.joblib',
        
        # Notebooks
        'Model_Training.ipynb': 'notebooks/Model_Training.ipynb',
        'SpamShield_Documentation.ipynb': 'notebooks/SpamShield_Documentation.ipynb',
        
        # Scripts
        'train_model.py': 'scripts/train_model.py',
        'save_artifact.py': 'scripts/save_artifact.py',
        'create_notebook.py': 'scripts/create_notebook.py',
        'create_training_notebook.py': 'scripts/create_training_notebook.py',
        
        # Documentation
        'README.md': 'docs/README.md',
        'MODEL_USAGE.md': 'docs/MODEL_USAGE.md',
        
        # Data
        'emails.csv': 'data/emails.csv',
        'sms.csv': 'data/sms.csv',
        'spamham.csv': 'data/spamham.csv',
    }
    
    print("\nOrganizing files...")
    for source, destination in file_moves.items():
        if os.path.exists(source):
            try:
                # Create parent directory if needed
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                
                # Move file
                shutil.move(source, destination)
                print(f"  ✅ {source} → {destination}")
            except Exception as e:
                print(f"  ⚠️  Could not move {source}: {e}")
        else:
            print(f"  ⚠️  File not found: {source}")
    
    return True

def create_readme():
    """Create a main README.md in the root."""
    
    readme_content = """# SpamShield AI - Spam/Ham Detector

A production-ready Flask web application that uses Machine Learning to detect spam messages.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open browser:**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
spam-ham/
├── app.py                  # Main Flask application
├── model_loader.py         # Model loading utility
├── requirements.txt        # Python dependencies
├── models/                 # Trained models
│   ├── artifact           # Main model (pickle)
│   └── artifact_metrics   # Model metrics
├── data/                  # Datasets
│   ├── emails.csv
│   ├── sms.csv
│   └── spamham.csv
├── notebooks/             # Jupyter notebooks
│   ├── Model_Training.ipynb
│   └── SpamShield_Documentation.ipynb
├── scripts/               # Utility scripts
│   ├── train_model.py
│   └── save_artifact.py
├── docs/                  # Documentation
│   ├── README.md
│   └── MODEL_USAGE.md
├── static/                # Frontend assets
│   ├── css/
│   └── js/
└── templates/             # HTML templates
    └── index.html
```

## 📊 Features

- **ML Model**: Ensemble classifier (96%+ accuracy)
- **File Upload**: Supports .txt, .csv, .docx, .pdf
- **Real-time Analysis**: Instant spam detection
- **Modern UI**: Dark mode with animations
- **API Endpoints**: RESTful API for integration

## 📚 Documentation

See `docs/` folder for detailed documentation:
- `README.md` - Complete setup guide
- `MODEL_USAGE.md` - Model usage examples

## 🔧 Development

**Train new model:**
```bash
python scripts/train_model.py
```

**Run notebooks:**
```bash
jupyter notebook notebooks/
```

## 📝 License

Open source - Educational and commercial use permitted.
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("\n✅ Created new README.md in root")
    return True

def update_paths_in_files():
    """Update file paths in Python files to reflect new structure."""
    
    updates = {
        'app.py': [
            ("from model_loader import", "from model_loader import"),
        ],
        'model_loader.py': [
            ("model_path='artifact'", "model_path='models/artifact'"),
            ("metrics_path='artifact_metrics'", "metrics_path='models/artifact_metrics'"),
        ],
    }
    
    print("\nUpdating file paths...")
    for filename, replacements in updates.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for old, new in replacements:
                    content = content.replace(old, new)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"  ✅ Updated {filename}")
            except Exception as e:
                print(f"  ⚠️  Error updating {filename}: {e}")
    
    return True

def cleanup():
    """Remove unnecessary files and directories."""
    
    cleanup_items = [
        '__pycache__',
        '.ipynb_checkpoints',
    ]
    
    print("\nCleaning up...")
    for item in cleanup_items:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                print(f"  ✅ Removed {item}")
            except Exception as e:
                print(f"  ⚠️  Could not remove {item}: {e}")
    
    return True

def main():
    """Main reorganization function."""
    
    print("=" * 60)
    print("SPAMSHIELD AI - PROJECT REORGANIZATION")
    print("=" * 60)
    
    try:
        create_directory_structure()
        organize_files()
        create_readme()
        update_paths_in_files()
        cleanup()
        
        print("\n" + "=" * 60)
        print("✅ PROJECT REORGANIZATION COMPLETE!")
        print("=" * 60)
        print("\nNew structure:")
        print("  📁 models/     - Trained models")
        print("  📁 data/       - Datasets")
        print("  📁 notebooks/  - Jupyter notebooks")
        print("  📁 scripts/    - Utility scripts")
        print("  📁 docs/       - Documentation")
        print("  📁 static/     - Frontend assets")
        print("  📁 templates/  - HTML templates")
        print("\n✅ Ready to use! Run: python app.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during reorganization: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

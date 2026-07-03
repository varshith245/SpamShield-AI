# SpamShield AI - Project Structure

## 📁 Directory Structure

```
spam-ham/
├── app.py                      # Main Flask application
├── model_loader.py             # Model loading utility
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── reorganize_project.py       # Project reorganization script
│
├── models/                     # Trained models and artifacts
│   ├── artifact                # Main model (pickle format, 30MB)
│   ├── artifact_metrics        # Model performance metrics
│   ├── spam_detector_model.joblib  # Model (joblib format)
│   └── model_metrics.joblib    # Metrics (joblib format)
│
├── data/                       # Training datasets
│   ├── emails.csv              # Email spam dataset (5.4MB)
│   ├── sms.csv                 # SMS spam dataset (499KB)
│   └── spamham.csv             # Combined dataset (5.7MB)
│
├── notebooks/                  # Jupyter notebooks
│   ├── Model_Training.ipynb    # Complete training workflow
│   └── SpamShield_Documentation.ipynb  # Project documentation
│
├── scripts/                    # Utility scripts
│   ├── train_model.py          # Train the ML model
│   ├── save_artifact.py        # Save model as pickle artifact
│   ├── create_notebook.py      # Generate documentation notebook
│   └── create_training_notebook.py  # Generate training notebook
│
├── docs/                       # Documentation
│   ├── README.md               # Complete setup guide
│   └── MODEL_USAGE.md          # Model usage examples
│
├── static/                     # Frontend assets
│   ├── css/
│   │   └── style.css           # Application styles
│   └── js/
│       └── main.js             # Frontend JavaScript
│
├── templates/                  # HTML templates
│   └── index.html              # Main application page
│
└── uploads/                    # Temporary file uploads (auto-created)

```

## 🎯 Core Files

### Application Files
- **app.py** - Main Flask application with routes
- **model_loader.py** - Model loading and prediction utility
- **requirements.txt** - All Python dependencies with versions

### Model Files (models/)
- **artifact** - Production model in pickle format (default)
- **artifact_metrics** - Model performance metrics
- Alternative: joblib format files also available

### Data Files (data/)
- **emails.csv** - Email spam/ham dataset
- **sms.csv** - SMS spam/ham dataset
- **spamham.csv** - Combined spam/ham dataset

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run application:**
   ```bash
   python app.py
   ```

3. **Access at:** `http://localhost:5000`

## 🔧 Development

### Train New Model
```bash
python scripts/train_model.py
```

### Save as Pickle Artifact
```bash
python scripts/save_artifact.py
```

### Run Notebooks
```bash
jupyter notebook notebooks/
```

## 📊 File Sizes

| Directory | Size | Description |
|-----------|------|-------------|
| models/ | ~60 MB | Trained models |
| data/ | ~11 MB | Training datasets |
| notebooks/ | ~35 KB | Jupyter notebooks |
| static/ | ~15 KB | Frontend assets |

## 🗂️ File Organization

### By Purpose
- **Production**: app.py, model_loader.py, models/, static/, templates/
- **Development**: scripts/, notebooks/, data/
- **Documentation**: docs/, README.md

### By Type
- **Python**: .py files in root and scripts/
- **Data**: .csv files in data/
- **Models**: .joblib and pickle files in models/
- **Notebooks**: .ipynb files in notebooks/
- **Web**: HTML/CSS/JS in templates/ and static/

## ✅ Clean Structure Benefits

1. **Organized** - Clear separation of concerns
2. **Scalable** - Easy to add new features
3. **Maintainable** - Files grouped logically
4. **Professional** - Industry-standard structure
5. **Documented** - Clear purpose for each directory

## 🔄 Migration Notes

All files have been reorganized from the root directory into appropriate subdirectories. The application automatically uses the new paths:
- Models loaded from `models/artifact`
- Data accessed from `data/`
- Scripts in `scripts/`
- Documentation in `docs/`

No code changes required - paths are already updated!

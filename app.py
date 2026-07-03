import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from model_loader import get_model, detect_keywords

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv', 'docx', 'pdf'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Model using model_loader utility
try:
    spam_model = get_model()
    print("SpamShield AI model loaded successfully!")
    print(f"Model Info: {spam_model.get_model_info()}")
except Exception as e:
    print(f"Error loading model: {e}")
    spam_model = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def read_file_content(filepath, filename):
    """
    Read content from uploaded file.
    
    Supports: .txt, .csv, .docx, .pdf
    """
    ext = filename.rsplit('.', 1)[1].lower()
    content = ""
    
    try:
        if ext == 'txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
        elif ext == 'csv':
            import pandas as pd
            df = pd.read_csv(filepath)
            # Try to find a text column
            text_col = next((col for col in df.columns if 'message' in col.lower() or 'text' in col.lower()), None)
            if text_col:
                content = " ".join(df[text_col].astype(str).tolist())
            else:
                content = df.to_string()
                
        elif ext == 'docx':
            import docx
            doc = docx.Document(filepath)
            content = " ".join([para.text for para in doc.paragraphs])
            
        elif ext == 'pdf':
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            content = " ".join([page.extract_text() for page in reader.pages])
            
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
        
    return content

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    """
    Scan text or file for spam.
    
    Accepts:
    - text: String (via form data)
    - file: File upload (.txt, .csv, .docx, .pdf)
    
    Returns:
    - JSON with verdict, confidence, keywords, and preview
    """
    if not spam_model:
        return jsonify({'error': 'Model not loaded'}), 500

    text_to_scan = ""
    
    # Handle Text Input
    if 'text' in request.form and request.form['text'].strip():
        text_to_scan = request.form['text']
    
    # Handle File Upload
    elif 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            text_to_scan = read_file_content(filepath, filename)
            os.remove(filepath)  # Clean up
            
            if text_to_scan is None:
                return jsonify({'error': 'Could not read file content'}), 400
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    else:
        return jsonify({'error': 'No input provided'}), 400

    if not text_to_scan.strip():
        return jsonify({'error': 'Empty content'}), 400

    # Make prediction using model_loader
    try:
        result = spam_model.predict(text_to_scan)
        keywords = detect_keywords(text_to_scan)
        
        return jsonify({
            'verdict': result['verdict'],
            'confidence': result['confidence'],
            'keywords': keywords,
            'message_preview': text_to_scan[:200] + "..." if len(text_to_scan) > 200 else text_to_scan
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/api/stats')
def stats():
    """
    Get model performance statistics.
    
    Returns:
    - JSON with accuracy and confusion matrix
    """
    if spam_model:
        metrics = spam_model.get_metrics()
        return jsonify(metrics)
    else:
        return jsonify({'accuracy': 0, 'confusion_matrix': []}), 500

@app.route('/api/model-info')
def model_info():
    """
    Get detailed model information.
    
    Returns:
    - JSON with model details
    """
    if spam_model:
        info = spam_model.get_model_info()
        return jsonify(info)
    else:
        return jsonify({'error': 'Model not loaded'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

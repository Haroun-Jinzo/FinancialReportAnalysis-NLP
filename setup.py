"""
Financial NLP Agent - Setup Script
Automates environment setup and model downloads
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directory_structure():
    """Create project directory structure"""
    directories = [
        'config',
        'data/raw',
        'data/processed',
        'data/models',
        'data/outputs',
        'src/preprocessing',
        'src/models',
        'src/extraction',
        'src/analysis',
        'src/agents',
        'src/utils',
        'dashboard/components',
        'dashboard/styles',
        'notebooks',
        'tests',
        'scripts',
        'logs'
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py for Python packages
        if directory.startswith('src/') or directory.startswith('dashboard/'):
            init_file = Path(directory) / '__init__.py'
            init_file.touch(exist_ok=True)
    
    print("✓ Directory structure created")

def download_spacy_models():
    """Download required spaCy models"""
    print("\nDownloading spaCy models...")
    models = ['en_core_web_sm', 'en_core_web_md']
    
    for model in models:
        try:
            print(f"  Downloading {model}...")
            subprocess.run([sys.executable, '-m', 'spacy', 'download', model], 
                         check=True, capture_output=True)
            print(f"  ✓ {model} downloaded")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to download {model}: {e}")

def download_nltk_data():
    """Download required NLTK data"""
    print("\nDownloading NLTK data...")
    import nltk
    
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    for resource in resources:
        try:
            print(f"  Downloading {resource}...")
            nltk.download(resource, quiet=True)
            print(f"  ✓ {resource} downloaded")
        except Exception as e:
            print(f"  ✗ Failed to download {resource}: {e}")

def create_config_files():
    """Create default configuration files"""
    print("\nCreating configuration files...")
    
    # model_config.yaml
    model_config = """# Model Configuration
models:
  ner:
    name: "dslim/bert-base-NER"
    cache_dir: "data/models/ner"
    device: "cpu"  # Change to "cuda" if GPU available
  
  sentiment:
    name: "ProsusAI/finbert"
    cache_dir: "data/models/sentiment"
    device: "cpu"
  
  qa:
    name: "deepset/roberta-base-squad2"
    cache_dir: "data/models/qa"
    device: "cpu"
  
  summarization:
    name: "facebook/bart-large-cnn"
    cache_dir: "data/models/summarization"
    device: "cpu"
  
  embeddings:
    name: "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: "data/models/embeddings"
    device: "cpu"

# Model parameters
parameters:
  max_length: 512
  batch_size: 8
  num_beams: 4
  temperature: 0.7
"""
    
    # pipeline_config.yaml
    pipeline_config = """# Pipeline Configuration
preprocessing:
  remove_stopwords: false
  lemmatize: true
  lowercase: true
  remove_numbers: false
  min_word_length: 2

extraction:
  confidence_threshold: 0.7
  max_entities: 100
  extract_tables: true
  extract_figures: true

analysis:
  sentiment_threshold: 0.6
  trend_window: 4  # quarters
  risk_keywords_weight: 1.5

output:
  format: "json"
  include_raw_text: false
  include_embeddings: false
"""
    
    # extraction_rules.json
    extraction_rules = """{
  "financial_entities": {
    "MONEY": ["revenue", "profit", "loss", "earnings", "income"],
    "METRIC": ["EBITDA", "EPS", "ROE", "ROI", "margin", "ratio"],
    "PERIOD": ["Q1", "Q2", "Q3", "Q4", "FY", "fiscal year", "quarter"],
    "PERCENTAGE": ["growth", "increase", "decrease", "change"]
  },
  
  "patterns": {
    "revenue": "revenue\\s*(?:of|was|is)?\\s*\\$?[\\d,]+\\.?\\d*[MBK]?",
    "profit": "(?:net\\s+)?profit\\s*(?:of|was|is)?\\s*\\$?[\\d,]+\\.?\\d*[MBK]?",
    "eps": "EPS\\s*(?:of|was|is)?\\s*\\$?[\\d.]+",
    "percentage_change": "(increased|decreased|rose|fell|grew)\\s*(?:by\\s*)?[\\d.]+%"
  },
  
  "risk_keywords": [
    "risk", "uncertainty", "challenge", "decline", "loss", 
    "volatility", "adverse", "litigation", "regulatory"
  ],
  
  "positive_keywords": [
    "growth", "profit", "increase", "strong", "exceed", 
    "outperform", "success", "innovation", "expansion"
  ]
}
"""
    
    # Write files
    with open('config/model_config.yaml', 'w') as f:
        f.write(model_config)
    
    with open('config/pipeline_config.yaml', 'w') as f:
        f.write(pipeline_config)
    
    with open('config/extraction_rules.json', 'w') as f:
        f.write(extraction_rules)
    
    print("✓ Configuration files created")

def create_env_file():
    """Create .env.example file"""
    env_content = """# Environment Variables

# Database
DATABASE_URL=sqlite:///data/financial_nlp.db

# Model Cache Directory
TRANSFORMERS_CACHE=./data/models

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# OCR (if using Tesseract)
TESSERACT_PATH=/usr/bin/tesseract

# Optional: GPU Settings
CUDA_VISIBLE_DEVICES=0
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    print("✓ .env.example created")

def main():
    """Main setup function"""
    print("=" * 60)
    print("Financial NLP Agent - Setup")
    print("=" * 60)
    
    # Create directories
    create_directory_structure()
    
    # Create config files
    create_config_files()
    create_env_file()
    
    # Download models
    try:
        download_nltk_data()
        download_spacy_models()
    except Exception as e:
        print(f"\n⚠ Warning: Some models failed to download: {e}")
        print("You can download them manually later.")
    
    print("\n" + "=" * 60)
    print("Setup Complete! ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Copy .env.example to .env and configure")
    print("2. Run: python scripts/download_models.py")
    print("3. Start with document parsing: see src/preprocessing/")
    print("\n")

if __name__ == "__main__":
    main()
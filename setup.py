#!/usr/bin/env python3
"""
Setup script for IndicaAI Pharmaceutical Marketing Agent
Downloads required models and initializes the project
"""

import os
import subprocess
import sys
import nltk
import spacy
from pathlib import Path

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('wordnet')
    print("NLTK data downloaded successfully!")

def download_spacy_model():
    """Download spaCy model for biomedical NLP"""
    print("Downloading spaCy model...")
    try:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("spaCy model downloaded successfully!")
    except subprocess.CalledProcessError:
        print("Failed to download spaCy model. Please run: python -m spacy download en_core_web_sm")

def create_directories():
    """Create necessary project directories"""
    directories = [
        "data/raw",
        "data/processed",
        "data/outputs",
        "models",
        "notebooks",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_env_template():
    """Create .env template file"""
    env_template = """# API Keys for IndicaAI
# News API (get free key from https://newsapi.org/)
NEWS_API_KEY=your_news_api_key_here

# OpenAI API (optional, for advanced LLM features)
OPENAI_API_KEY=your_openai_api_key_here

# Configuration
LOG_LEVEL=INFO
MAX_REQUESTS_PER_MINUTE=60
"""

    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_template)
        print("Created .env template file. Please add your API keys.")
    else:
        print(".env file already exists.")

def main():
    """Main setup function"""
    print("Setting up IndicaAI Pharmaceutical Marketing Agent...")

    # Create directories
    create_directories()

    # Download models and data
    download_nltk_data()
    download_spacy_model()

    # Create environment template
    create_env_template()

    print("\nSetup completed successfully!")
    print("Next steps:")
    print("1. Add your API keys to the .env file")
    print("2. Run: python scripts/make_dataset.py to collect initial data")
    print("3. Run: python main.py to start the application")

if __name__ == "__main__":
    main()

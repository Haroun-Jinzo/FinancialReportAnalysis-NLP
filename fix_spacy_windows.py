"""
Fix spaCy model installation for Windows
"""

import subprocess
import sys

print("="*60)
print("SPACY MODEL FIX FOR WINDOWS")
print("="*60)

# Method 1: Direct pip install (most reliable for Windows)
print("\n[Method 1] Installing via pip...")
print("Running: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl")

try:
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
    ], check=True, capture_output=True, text=True)
    
    print("✓ Model downloaded successfully!")
    print(result.stdout)
    
except subprocess.CalledProcessError as e:
    print("✗ Method 1 failed")
    print(e.stderr)
    
    # Method 2: Alternative approach
    print("\n[Method 2] Trying alternative installation...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "en-core-web-sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
        ], check=True)
        print("✓ Alternative method succeeded!")
    except:
        print("✗ Method 2 also failed")
        print("\n📝 Manual installation steps:")
        print("1. Download: https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl")
        print("2. Run: pip install path\\to\\downloaded\\file.whl")
        sys.exit(1)

# Test if it works
print("\n[Testing] Loading spaCy model...")
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    print("✓ Model loaded successfully!")
    
    # Quick test
    doc = nlp("Apple Inc. reported revenue of $90 billion.")
    print(f"✓ Test successful! Found {len(doc.ents)} entities")
    
    print("\n" + "="*60)
    print("✅ SPACY FIXED! All systems ready!")
    print("="*60)
    print("\n✅ You can now run: python scripts/download_models.py")
    
except Exception as e:
    print(f"✗ Model still not working: {e}")
    print("\n🔧 Try these manual steps:")
    print("1. pip uninstall spacy")
    print("2. pip install spacy==3.7.2")
    print("3. pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl")
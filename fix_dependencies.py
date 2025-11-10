"""
Dependency Fix Script
Resolves spaCy and sentence-transformers installation issues
"""

import subprocess
import sys

def run_command(command, description):
    """Run a command and report status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            shell=True
        )
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("="*60)
    print("DEPENDENCY FIX SCRIPT")
    print("="*60)
    
    fixes = []
    
    # Fix 1: Upgrade pip, setuptools, wheel
    print("\n[1/5] Upgrading pip, setuptools, and wheel...")
    fixes.append(run_command(
        f"{sys.executable} -m pip install --upgrade pip setuptools wheel",
        "Upgrade pip and tools"
    ))
    
    # Fix 2: Uninstall problematic packages
    print("\n[2/5] Removing conflicting packages...")
    subprocess.run(
        f"{sys.executable} -m pip uninstall -y sentence-transformers huggingface_hub",
        shell=True,
        capture_output=True
    )
    print("✓ Cleaned up conflicting packages")
    
    # Fix 3: Install compatible versions
    print("\n[3/5] Installing compatible versions...")
    fixes.append(run_command(
        f"{sys.executable} -m pip install huggingface_hub==0.20.3 sentence-transformers==2.2.2",
        "Install compatible versions"
    ))
    
    # Fix 4: Install spaCy properly
    print("\n[4/5] Installing spaCy and models...")
    fixes.append(run_command(
        f"{sys.executable} -m pip install spacy==3.7.2",
        "Install spaCy"
    ))
    
    # Download spaCy models
    print("\n  Downloading spaCy English model...")
    fixes.append(run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Download en_core_web_sm"
    ))
    
    # Fix 5: Verify installations
    print("\n[5/5] Verifying installations...")
    print("\nChecking imports...")
    
    verification = {
        'transformers': False,
        'sentence_transformers': False,
        'spacy': False,
        'torch': False
    }
    
    try:
        import transformers
        verification['transformers'] = True
        print(f"  ✓ transformers v{transformers.__version__}")
    except Exception as e:
        print(f"  ✗ transformers: {e}")
    
    try:
        import sentence_transformers
        verification['sentence_transformers'] = True
        print(f"  ✓ sentence_transformers v{sentence_transformers.__version__}")
    except Exception as e:
        print(f"  ✗ sentence_transformers: {e}")
    
    try:
        import spacy
        verification['spacy'] = True
        print(f"  ✓ spacy v{spacy.__version__}")
        
        # Try loading model
        try:
            nlp = spacy.load('en_core_web_sm')
            print(f"  ✓ spacy model 'en_core_web_sm' loaded")
        except:
            print(f"  ⚠ spacy model 'en_core_web_sm' not found")
    except Exception as e:
        print(f"  ✗ spacy: {e}")
    
    try:
        import torch
        verification['torch'] = True
        print(f"  ✓ torch v{torch.__version__}")
        print(f"  ℹ CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"  ✗ torch: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("FIX SUMMARY")
    print("="*60)
    
    success = all(verification.values())
    
    if success:
        print("✓ All packages installed correctly!")
        print("\n✅ You can now run: python scripts/download_models.py")
    else:
        print("⚠ Some packages have issues. See details above.")
        print("\nMissing packages:")
        for pkg, status in verification.items():
            if not status:
                print(f"  - {pkg}")
        
        print("\n📝 Manual fix commands:")
        print(f"  pip install --upgrade transformers torch")
        print(f"  pip install huggingface_hub==0.20.3 sentence-transformers==2.2.2")
        print(f"  pip install spacy")
        print(f"  python -m spacy download en_core_web_sm")
    
    print("\n")

if __name__ == "__main__":
    main()
"""
Quick test to verify all dependencies are working
"""

print("Testing dependencies...\n")

tests_passed = 0
tests_total = 6

# Test 1: Transformers
try:
    import transformers
    print(f"✓ transformers v{transformers.__version__}")
    tests_passed += 1
except Exception as e:
    print(f"✗ transformers: {e}")

# Test 2: PyTorch
try:
    import torch
    print(f"✓ torch v{torch.__version__}")
    print(f"  ℹ CUDA available: {torch.cuda.is_available()}")
    tests_passed += 1
except Exception as e:
    print(f"✗ torch: {e}")

# Test 3: Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    print(f"✓ sentence-transformers imported successfully")
    tests_passed += 1
except Exception as e:
    print(f"✗ sentence-transformers: {e}")

# Test 4: spaCy
try:
    import spacy
    print(f"✓ spacy v{spacy.__version__}")
    tests_passed += 1
except Exception as e:
    print(f"✗ spacy: {e}")

# Test 5: spaCy Model
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    print(f"✓ spacy model 'en_core_web_sm' loaded")
    tests_passed += 1
except Exception as e:
    print(f"✗ spacy model: {e}")

# Test 6: HuggingFace Hub
try:
    from huggingface_hub import hf_hub_download
    print(f"✓ huggingface_hub working correctly")
    tests_passed += 1
except Exception as e:
    print(f"✗ huggingface_hub: {e}")

# Summary
print("\n" + "="*50)
print(f"Result: {tests_passed}/{tests_total} tests passed")
print("="*50)

if tests_passed == tests_total:
    print("\n🎉 All dependencies working!")
    print("✅ You can now run: python scripts/download_models.py")
else:
    print("\n⚠ Some dependencies failed")
    print("Run: python fix_dependencies.py")
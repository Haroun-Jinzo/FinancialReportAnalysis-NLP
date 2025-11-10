import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    BartTokenizer,
    BartForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer
import torch

CACHE_DIR = Path("data/models")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# set environment variable for transformers cache
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)

def download_model(model_name, model_class, tokenizer_class, category):
    """Download and cache a model and its tokenizer."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Category: {category}")
    print(f"{'='*60}")

    cache_path = CACHE_DIR / category
    cache_path.mkdir(exist_ok=True)

    try:
        #Download Tokenizer
        print(" downloading Tokenizer...")
        tokenzier = tokenizer_class.from_pretrained(model_name, cache_dir=str(cache_path))

        print(" ✓ Tokenizer downloaded")

        # Download Model
        print(" downloading Model...")
        model = model_class.from_pretrained(model_name, cache_dir=str(cache_path))
        
        print(" ✓ Model downloaded")

        # get models Size
        param_count = sum(p.numel() for p in model.parameters())
        size_mb = param_count * 4 / (1024 ** 2)  # assuming float32 (4 bytes)
        print(f" Parameters: {param_count:,} | Approx. Size: {size_mb:.2f} MB")

        #Test the model
        print(" Testing model...")
        test_text = "Apple Inc. is looking at buying U.K. startup for $1 billion"
        inputs = tokenzier(test_text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)

        print(" ✓ Model test successful")

        return True
    except Exception as e:
        print(f" ✗ Failed to download or test {model_name}: {e}")
        return False
    
def download_sentence_transformer(model_name):
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Category: Embeddings")
    print(f"{'='*60}")

    try:
        cache_path = CACHE_DIR / "embeddings"
        cache_path.mkdir(exist_ok=True)

        print(" downloading Sentence Model...")
        model = SentenceTransformer(model_name, cache_folder=str(cache_path))
        print(" Model downloaded")

        # test the model
        print(" Testing model...")
        test_embeddings = model.encode(["This is a test sentence."])
        print(f" Model test successful (embeding dim: {len(test_embeddings)})")

        return True
    except Exception as e:
        print(f" ✗ Failed to download or test {model_name}: {e}")
        return False
    
def main():
    print("\n" + "="*60)
    print("Financial NLP Agent - Model Downloader")
    print("="*60)
    print("\nThis will download ~2-3 GB of models. Continue? (y/n)")
    
    response = input().strip().lower()
    if response != 'y':
        print("Download cancelled.")
        return
    
    results = {}
    
    # 1. Named Entity Recognition (NER)
    results['NER'] = download_model(
        model_name="dslim/bert-base-NER",
        model_class=AutoModelForTokenClassification,
        tokenizer_class=AutoTokenizer,
        category="ner"
    )
    
    # 2. Financial Sentiment Analysis
    results['Sentiment'] = download_model(
        model_name="ProsusAI/finbert",
        model_class=AutoModelForSequenceClassification,
        tokenizer_class=AutoTokenizer,
        category="sentiment"
    )
    
    # 3. Question Answering
    results['QA'] = download_model(
        model_name="deepset/roberta-base-squad2",
        model_class=AutoModelForQuestionAnswering,
        tokenizer_class=AutoTokenizer,
        category="qa"
    )
    
    # 4. Summarization
    results['Summarization'] = download_model(
        model_name="facebook/bart-large-cnn",
        model_class=BartForConditionalGeneration,
        tokenizer_class=BartTokenizer,
        category="summarization"
    )
    
    # 5. Sentence Embeddings
    results['Embeddings'] = download_sentence_transformer(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    for model_type, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{model_type:20s}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nTotal: {success_count}/{total_count} models downloaded successfully")
    
    if success_count == total_count:
        print("\n🎉 All models ready! You can now start processing documents.")
    else:
        print("\n⚠ Some models failed. Check error messages above.")
        print("You can retry by running this script again.")
    
    print("\n")

if __name__ == "__main__":
    main()
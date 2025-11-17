"""
Model Loader - Base class for managing transformer models
Handles loading, caching, and device management
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    BartForConditionalGeneration,
    BartTokenizer,
    pipeline
)


class ModelLoader:
    """
    Base class for loading and managing transformer models
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize model loader
        
        Args:
            config_path: Path to model configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.device = self._get_device()
        self.cache_dir = Path("data/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store loaded models to avoid reloading
        self._loaded_models = {}
        
        print(f"✓ Model Loader initialized")
        print(f"  Device: {self.device}")
        print(f"  Cache: {self.cache_dir}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"⚠ Config file not found: {self.config_path}")
            print("  Using default configuration")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration if file not found"""
        return {
            'models': {
                'ner': {
                    'name': 'dslim/bert-base-NER',
                    'cache_dir': 'data/models/ner',
                    'device': 'cpu'
                },
                'sentiment': {
                    'name': 'ProsusAI/finbert',
                    'cache_dir': 'data/models/sentiment',
                    'device': 'cpu'
                },
                'qa': {
                    'name': 'deepset/roberta-base-squad2',
                    'cache_dir': 'data/models/qa',
                    'device': 'cpu'
                },
                'summarization': {
                    'name': 'facebook/bart-large-cnn',
                    'cache_dir': 'data/models/summarization',
                    'device': 'cpu'
                },
                'embeddings': {
                    'name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'cache_dir': 'data/models/embeddings',
                    'device': 'cpu'
                }
            },
            'parameters': {
                'max_length': 512,
                'batch_size': 8,
                'num_beams': 4,
                'temperature': 0.7
            }
        }
    
    def _get_device(self) -> str:
        """Determine the best available device (cuda/cpu)"""
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"  ℹ GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print(f"  ℹ Using CPU (GPU not available)")
        
        return device
    
    def load_model(self, model_type: str, force_reload: bool = False) -> Dict:
        """
        Load a specific model type
        
        Args:
            model_type: Type of model (ner, sentiment, qa, summarization, embeddings)
            force_reload: Force reload even if cached
            
        Returns:
            Dictionary with model, tokenizer, and pipeline
        """
        # Check cache first
        if model_type in self._loaded_models and not force_reload:
            print(f"✓ Using cached {model_type} model")
            return self._loaded_models[model_type]
        
        print(f"\n{'='*60}")
        print(f"Loading {model_type.upper()} Model")
        print(f"{'='*60}")
        
        model_config = self.config['models'].get(model_type)
        if not model_config:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_name = model_config['name']
        cache_dir = Path(model_config['cache_dir'])
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Model: {model_name}")
        print(f"  Cache: {cache_dir}")
        
        try:
            result = self._load_specific_model(model_type, model_name, cache_dir)
            self._loaded_models[model_type] = result
            print(f"✓ {model_type.upper()} model loaded successfully")
            return result
            
        except Exception as e:
            print(f"✗ Error loading {model_type} model: {e}")
            raise
    
    def _load_specific_model(self, model_type: str, model_name: str, 
                            cache_dir: Path) -> Dict:
        """Load specific model based on type"""
        
        if model_type == 'ner':
            return self._load_ner_model(model_name, cache_dir)
        elif model_type == 'sentiment':
            return self._load_sentiment_model(model_name, cache_dir)
        elif model_type == 'qa':
            return self._load_qa_model(model_name, cache_dir)
        elif model_type == 'summarization':
            return self._load_summarization_model(model_name, cache_dir)
        elif model_type == 'embeddings':
            return self._load_embeddings_model(model_name, cache_dir)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_ner_model(self, model_name: str, cache_dir: Path) -> Dict:
        """Load NER model"""
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        
        print("  Loading model...")
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        model.to(self.device)
        model.eval()
        
        print("  Creating pipeline...")
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device == 'cuda' else -1,
            aggregation_strategy="simple"
        )
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'pipeline': ner_pipeline,
            'type': 'ner'
        }
    
    def _load_sentiment_model(self, model_name: str, cache_dir: Path) -> Dict:
        """Load sentiment analysis model"""
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        
        print("  Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        model.to(self.device)
        model.eval()
        
        print("  Creating pipeline...")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device == 'cuda' else -1
        )
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'pipeline': sentiment_pipeline,
            'type': 'sentiment'
        }
    
    def _load_qa_model(self, model_name: str, cache_dir: Path) -> Dict:
        """Load question answering model"""
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        
        print("  Loading model...")
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        model.to(self.device)
        model.eval()
        
        print("  Creating pipeline...")
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device == 'cuda' else -1
        )
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'pipeline': qa_pipeline,
            'type': 'qa'
        }
    
    def _load_summarization_model(self, model_name: str, cache_dir: Path) -> Dict:
        """Load summarization model"""
        print("  Loading tokenizer...")
        tokenizer = BartTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        
        print("  Loading model...")
        model = BartForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        model.to(self.device)
        model.eval()
        
        print("  Creating pipeline...")
        summarization_pipeline = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device == 'cuda' else -1
        )
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'pipeline': summarization_pipeline,
            'type': 'summarization'
        }
    
    def _load_embeddings_model(self, model_name: str, cache_dir: Path) -> Dict:
        """Load sentence embeddings model"""
        from sentence_transformers import SentenceTransformer
        
        print("  Loading sentence transformer...")
        model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_dir)
        )
        
        if self.device == 'cuda':
            model = model.to(self.device)
        
        return {
            'model': model,
            'tokenizer': None,  # SentenceTransformer handles its own tokenization
            'pipeline': None,
            'type': 'embeddings'
        }
    
    def get_model_info(self, model_type: str) -> Dict:
        """Get information about a loaded model"""
        if model_type not in self._loaded_models:
            return {'status': 'not_loaded'}
        
        model_data = self._loaded_models[model_type]
        model = model_data['model']
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        return {
            'status': 'loaded',
            'type': model_type,
            'parameters': f"{param_count:,}",
            'device': self.device,
            'model_name': self.config['models'][model_type]['name']
        }
    
    def clear_cache(self, model_type: Optional[str] = None):
        """Clear model cache"""
        if model_type:
            if model_type in self._loaded_models:
                del self._loaded_models[model_type]
                print(f"✓ Cleared {model_type} model from cache")
        else:
            self._loaded_models.clear()
            print("✓ Cleared all models from cache")
        
        # Force garbage collection
        import gc
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = ModelLoader()
    
    # Load a model
    ner_model = loader.load_model('ner')
    
    # Get model info
    info = loader.get_model_info('ner')
    print(f"\nModel Info: {info}")
    
    print("\n✓ Model Loader Module Ready!")
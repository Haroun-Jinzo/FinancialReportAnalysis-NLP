import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

# Download nltk data
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from data_preprocessing import NERDataConverter
from data_preprocessing import FinancialNERDataset
from compute_metrics import compute_metrics_wrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== MAIN TRAINER CLASS ====================

class FinancialNERFineTuner:
    
    def __init__(self, 
                 base_model: str = 'dslim/bert-base-NER',
                 output_dir: str = 'models/financial_ner'):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize data converter
        self.converter = NERDataConverter()
        self.label2id = self.converter.label2id
        self.id2label = self.converter.id2label
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        logger.info(f"Initialized Financial NER Fine-tuner")
        logger.info(f"Base model: {self.base_model}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Number of labels: {len(self.label2id)}")
    
    def load_and_prepare_data(self,
                             data_file: str,
                             format: str = 'json') -> Dict:
        logger.info(f"Loading data from {data_file}")
        
        if format == 'json':
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Convert to NER format
            examples = self.converter.convert_from_json(raw_data)
        
        elif format == 'jsonl':
            # Handle JSONL format (one JSON object per line)
            raw_data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            raw_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON line: {e}")
                            continue
            
            # Convert to NER format
            examples = self.converter.convert_from_json(raw_data)
        
        else:
            raise ValueError(f"Format {format} not supported yet")
        
        logger.info(f"Loaded {len(examples)} examples")
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        train_val, test = train_test_split(examples, test_size=0.15, random_state=42)
        train, val = train_test_split(train_val, test_size=0.15, random_state=42)
        
        logger.info(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        return {
            'train': train,
            'val': val,
            'test': test
        }
    
    def fine_tune(self,
                 data_splits: Dict,
                 num_epochs: int = 5,
                 batch_size: int = 16,
                 learning_rate: float = 5e-5,
                 max_length: int = 128) -> Dict:
        logger.info("="*70)
        logger.info("STARTING FINANCIAL NER FINE-TUNING")
        logger.info("="*70)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Load model with correct number of labels
        num_labels = len(self.label2id)
        logger.info(f"Loading model from {self.base_model}...")
        logger.info(f"Configuring for {num_labels} labels")
        logger.info(f"Labels: {list(self.label2id.keys())[:10]}...")
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.base_model,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True  # Important: allows loading with different output size
        )
        
        # Verify the model configuration
        logger.info(f"Model output layer size: {self.model.config.num_labels}")
        logger.info(f"Expected label range: 0 to {num_labels - 1}")
        
        self.model.to(self.device)
        
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset = FinancialNERDataset(
            data_splits['train'],
            self.tokenizer,
            self.label2id,
            max_length
        )
        
        val_dataset = FinancialNERDataset(
            data_splits['val'],
            self.tokenizer,
            self.label2id,
            max_length
        )
        
        # Data collator for token classification
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=100,
            
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            
            fp16=torch.cuda.is_available(),
            seed=42,
            report_to=None
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_wrapper(self.id2label),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        logger.info("Starting training...")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples: {len(val_dataset)}")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        train_result = self.trainer.train()
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        logger.info("Evaluating on validation set...")
        val_metrics = self.trainer.evaluate()
        
        # Save
        logger.info(f"Saving model to {self.output_dir}")
        self.trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        # Save label mappings
        with open(self.output_dir / 'label_mappings.json', 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f, indent=2)
        
        # Save config
        config = {
            'base_model': self.base_model,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_length': max_length,
            'training_time': training_time,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'val_metrics': val_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Training time: {training_time/60:.1f} minutes")
        logger.info(f"Val F1: {val_metrics['eval_f1']:.4f}")
        logger.info(f"Val Precision: {val_metrics['eval_precision']:.4f}")
        logger.info(f"Val Recall: {val_metrics['eval_recall']:.4f}")
        logger.info("="*70)
        
        return config
    
    def evaluate_on_val_set(self, data_splits: Dict, max_length: int = 128) -> Dict:
        logger.info("Evaluating on test set...")
        
        test_dataset = FinancialNERDataset(
            data_splits['test'],
            self.tokenizer,
            self.label2id,
            max_length
        )
        
        # Get predictions
        predictions_output = self.trainer.predict(test_dataset)
        predictions = np.argmax(predictions_output.predictions, axis=2)
        labels = predictions_output.label_ids
        
        # Convert to seqeval format
        pred_labels = []
        true_labels = []
        
        for pred_seq, true_seq in zip(predictions, labels):
            pred_seq_labels = []
            true_seq_labels = []
            
            for pred_id, true_id in zip(pred_seq, true_seq):
                if true_id != -100:  # Ignore padding
                    pred_seq_labels.append(self.id2label[pred_id])
                    true_seq_labels.append(self.id2label[true_id])
            
            pred_labels.append(pred_seq_labels)
            true_labels.append(true_seq_labels)
        
        # Compute metrics
        metrics = {
            'precision': precision_score(true_labels, pred_labels),
            'recall': recall_score(true_labels, pred_labels),
            'f1': f1_score(true_labels, pred_labels)
        }
        
        # Classification report
        report = classification_report(true_labels, pred_labels)
        
        logger.info("\n" + "="*70)
        logger.info("TEST SET RESULTS")
        logger.info("="*70)
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info("\nDetailed Report:")
        logger.info("\n" + report)
        logger.info("="*70)
        
        # Save results
        results = {
            'metrics': metrics,
            'classification_report': report
        }
        
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump({'metrics': metrics}, f, indent=2)
        
        with open(self.output_dir / 'test_report.txt', 'w') as f:
            f.write(report)
        
        return results
    
    def predict(self, text: str) -> List[Dict]:
        from nltk.tokenize import word_tokenize
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        self.model.eval()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Get predictions
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Convert to labels
        word_ids = inputs.word_ids(batch_index=0)
        pred_labels = []
        
        for word_idx, pred_id in zip(word_ids, predictions[0]):
            if word_idx is not None:
                pred_labels.append((word_idx, self.id2label[pred_id.item()]))
        
        # Extract entities
        entities = self._extract_entities_from_labels(tokens, pred_labels)
        
        return entities
    
    def _extract_entities_from_labels(self, tokens, pred_labels):
        """Extract entities from BIO tags"""
        entities = []
        current_entity = None
        current_tokens = []
        seen_word_indices = set()
        
        for word_idx, label in pred_labels:
            # Skip if we've already processed this word index
            if word_idx in seen_word_indices:
                continue
            seen_word_indices.add(word_idx)
            
            # Make sure word_idx is valid
            if word_idx >= len(tokens):
                continue
            
            if label.startswith('B-'):
                # Start new entity - save previous entity if exists
                if current_entity and current_tokens:
                    entities.append({
                        'text': ' '.join(current_tokens),
                        'type': current_entity
                    })
                current_entity = label[2:]
                current_tokens = [tokens[word_idx]]
            
            elif label.startswith('I-'):
                # Continue entity only if type matches
                if current_entity == label[2:]:
                    current_tokens.append(tokens[word_idx])
                else:
                    # Type mismatch - end current entity and start new one
                    if current_entity and current_tokens:
                        entities.append({
                            'text': ' '.join(current_tokens),
                            'type': current_entity
                        })
                    current_entity = label[2:]
                    current_tokens = [tokens[word_idx]]
            
            else:  # 'O'
                # End current entity
                if current_entity and current_tokens:
                    entities.append({
                        'text': ' '.join(current_tokens),
                        'type': current_entity
                    })
                    current_entity = None
                    current_tokens = []
        
        # Add last entity if exists
        if current_entity and current_tokens:
            entities.append({
                'text': ' '.join(current_tokens),
                'type': current_entity
            })
        
        return entities

# ==================== EXAMPLE USAGE ====================

def main():
    """Complete NER fine-tuning workflow"""
    import sys
    import os
    
    print("\n" + "="*70)
    print("FINANCIAL NER FINE-TUNING")
    print("="*70)
    
    # Determine data file
    if len(sys.argv) < 2:
        # Use default data file
        data_file = 'data/training/xbrl_financial_ner_augmented.json'
        print(f"\n⚠️  No command-line argument provided. Using default: {data_file}")
        
        # Check if default file exists
        if not os.path.exists(data_file):
            print(f"\n❌ Error: Default file '{data_file}' not found!")
            print("\n📋 Usage:")
            print(f"  python {sys.argv[0]} <path_to_annotated_data.json>")
            print("\n💡 Example data format:")
            example_data = [
                {
                    "text": "Apple Inc. reported Q3 revenue of $90.1 billion",
                    "entities": [
                        {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10},
                        {"text": "Q3", "label": "DATE", "start": 20, "end": 22},
                        {"text": "$90.1 billion", "label": "MONEY", "start": 34, "end": 47}
                    ]
                }
            ]
            print(json.dumps(example_data, indent=2))
            print("\n📝 Steps to prepare data:")
            print("1. Collect financial texts")
            print("2. Use annotation tool (Label Studio, Prodigy, or Doccano)")
            print("3. Mark entities with start/end character positions")
            print("4. Save as JSON with format above")
            print("5. Run: python Ner_FineTuning.py your_data.json")
            return  # ONLY RETURN IF FILE NOT FOUND
    else:
        data_file = sys.argv[1]
        print(f"\n✅ Using data file: {data_file}")
    
    # At this point, data_file is set and exists
    # Continue with training...
    
    # Initialize
    print("\n🚀 Step 1: Initializing fine-tuner...")
    finetuner = FinancialNERFineTuner(
        base_model='dslim/bert-base-NER',
        output_dir='models/financial_ner'
    )
    
    # Load and prepare data
    print("\n📊 Step 2: Loading and preparing data...")
    try:
        data_splits = finetuner.load_and_prepare_data(
            data_file=data_file,
            format='json'  # Change to 'jsonl' if your file is JSONL
        )
    except FileNotFoundError:
        print(f"\n❌ Error: File '{data_file}' not found!")
        return
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check if we have enough data
    if len(data_splits['train']) < 50:
        print(f"\n⚠️  WARNING: Only {len(data_splits['train'])} training examples!")
        print("This is too few for effective training.")
        print("\nRecommendations:")
        print("1. You need at least 500-1000 annotated examples")
        print("2. Use the sample data I provided earlier")
        print("3. Or use a proper NER dataset")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Fine-tune
    print("\n🔥 Step 3: Fine-tuning model...")
    training_results = finetuner.fine_tune(
        data_splits,
        num_epochs=5,
        batch_size=16,
        learning_rate=5e-5,
        max_length=128
    )
    
    # Evaluate on val set
    print("\n📈 Step 4: Evaluating on val set...")
    test_results = finetuner.evaluate_on_val_set(data_splits)
    
    # Test on example sentences
    print("\n🧪 Step 5: Testing on example sentences...")
    examples = [
        "Apple Inc. reported Q3 revenue of $90.1 billion, up 5% from last year",
        "Tesla's stock (TSLA) rose 5.2% to $245.30 after the earnings call",
        "The Federal Reserve announced a 0.25% interest rate hike on March 15th",
        "Microsoft acquired Activision Blizzard for $68.7 billion",
        "Amazon's EPS increased to $1.29, beating analyst expectations"
    ]
    
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    for text in examples:
        print(f"\n📝 Text: {text}")
        try:
            entities = finetuner.predict(text)
            
            if entities:
                print("   Entities found:")
                for entity in entities:
                    print(f"     • {entity['text']:<20} → {entity['type']}")
            else:
                print("   No entities found")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "="*70)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"📁 Model saved to: {finetuner.output_dir}")
    print(f"📊 Test F1 Score: {test_results['metrics']['f1']:.4f}")
    print(f"📊 Test Precision: {test_results['metrics']['precision']:.4f}")
    print(f"📊 Test Recall: {test_results['metrics']['recall']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.model_loader import ModelLoader
from data_preprocessing import FinancialSentimentDataset
from compute_metrics import LoggingCallback, compute_metrics

# Configure logger
logger = logging.getLogger(__name__)

class FinbertFineTuner:

    def __init__(self, output_dir: str = 'models/finbert_finetuned'):
        self.output_dir = Path(output_dir)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # model and tokenzier
        self.model = None
        self.tokenizer = None
        self.trainer = None

        # Training History
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_f1': [],
            'eval_accuracy': []
        }

    def load_and_prepare_data(self, 
                             data_file: str,
                             text_column: str = 'text',
                             label_column: str = 'label',
                             test_size: float = 0.15,
                             val_size: float = 0.15,
                             balance_classes: bool = True) -> Dict:
        
        logger.info(f"Loading data from {data_file}")

        # Load data
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.endswith('.json'):
            df = pd.read_json(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")
        
        # Extract texts and labels
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].astype(str).str.lower().tolist()
        
        # CRITICAL: Clean data - remove rows with missing values
        logger.info(f"Initial data: {len(texts)} texts, {len(labels)} labels")
        
        # Create pairs and filter out invalid ones
        valid_pairs = []
        for text, label in zip(texts, labels):
            # Remove if text or label is missing/empty/nan
            if (text and label and 
                text.strip() and label.strip() and
                text.lower() not in ['nan', 'none', ''] and 
                label.lower() not in ['nan', 'none', '']):
                valid_pairs.append((text.strip(), label.strip()))
        
        if not valid_pairs:
            raise ValueError("No valid data after cleaning! Check your CSV file.")
        
        # Unzip back to separate lists
        texts, labels = zip(*valid_pairs)
        texts = list(texts)
        labels = list(labels)
        
        logger.info(f"After cleaning: {len(texts)} valid examples")
        if len(valid_pairs) < len(df):
            removed = len(df) - len(valid_pairs)
            logger.warning(f"⚠️  Removed {removed} rows with missing/invalid data")
        
        # Validate labels
        valid_labels = {'positive', 'negative', 'neutral'}
        unique_labels = set(labels)
        invalid = unique_labels - valid_labels
        if invalid:
            raise ValueError(f"Invalid labels found: {invalid}. Must be: {valid_labels}")
        
        logger.info(f"Final: {len(texts)} examples")
        logger.info(f"Label distribution:\n{pd.Series(labels).value_counts()}")
        
        # Balance classes if requested
        #if balance_classes:
         #   texts, labels = self._balance_classes(texts, labels)
          #  logger.info(f"After balancing: {len(texts)} examples")
           # logger.info(f"Balanced distribution:\n{pd.Series(labels).value_counts()}")
        
        # Split data
        # First: train vs (val+test)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels,
            test_size=(val_size + test_size),
            random_state=42,
            stratify=labels
        )
        
        # Second: val vs test
        val_size_adjusted = val_size / (val_size + test_size)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=(1 - val_size_adjusted),
            random_state=42,
            stratify=temp_labels
        )
        
        logger.info(f"Split sizes - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Validate splits
        if len(train_texts) == 0:
            raise ValueError("Training set is empty after splitting!")
        if len(val_texts) == 0:
            logger.warning("⚠️  Validation set is empty - consider increasing val_size")
        if len(test_texts) == 0:
            logger.warning("⚠️  Test set is empty - consider increasing test_size")
        
        return {
            'train': {'texts': train_texts, 'labels': train_labels},
            'val': {'texts': val_texts, 'labels': val_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }

    def Fine_Tune(self, data_splits: Dict,
                  num_epochs: int = 4,
                  batch_size: int = 16,
                  learning_rate: float = 2e-5,
                  warmup_steps: int = 500,
                  weight_decay: float = 0.01,
                  max_length: int = 128,
                  gradient_accumulation_steps: int = 1,
                  fp16: bool = True):
        

        logger.info("="*70)
        logger.info("STARTING FINBERT FINE-TUNING")
        logger.info("="*70)

        # Load tokenizer and model
        
        logger.info("Loading FinBERT model and tokenizer...")
        # Load FinBERT model
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3,
                                                                         problem_type="single_label_classification")

        self.model.to(self.device)


        logger.info("load datasets...")
        train_dataset = FinancialSentimentDataset(
            data_splits['train']['texts'],
            data_splits['train']['labels'],
            self.tokenizer,
            max_length=128
        )

        val_dataset = FinancialSentimentDataset(
            data_splits['val']['texts'],
            data_splits['val']['labels'],
            self.tokenizer,
            max_length=128
        )

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,

            # Evaluation strategy
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,

            # Logging
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=50,
            logging_first_step=True,

            # Optimization
            fp16=fp16 and torch.cuda.is_available(),
            dataloader_num_workers=4,

            # Best model
            load_best_model_at_end=True,
            metric_for_best_model='f1_weighted',
            greater_is_better=True,

            seed=42
        )

        # create Trainer
        log_file = self.output_dir / 'training.log'
        has_val_data = len(val_dataset) > 0

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if has_val_data else None,
            compute_metrics=compute_metrics if has_val_data else None,
            #callbacks=[
             #   EarlyStoppingCallback(early_stopping_patience=3),
              #  LoggingCallback(str(log_file))
               #]
        )

        #train
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

        logger.info("Evaluating on validation set...")

        val_metrics = self.trainer.evaluate()

        logger.info(f"Saving model to {self.output_dir}")
        self.trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))

        # save configuration
        config = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_length': max_length,
            'training_time': training_time,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'final_train_loss': train_result.training_loss,
            'val_metrics': val_metrics,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.output_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        logger.info("="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Training time: {training_time/60:.1f} minutes")
        logger.info(f"Final train loss: {train_result.training_loss:.4f}")
        logger.info(f"Val F1 (weighted): {val_metrics['eval_f1_weighted']:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['eval_accuracy']:.4f}")
        logger.info("="*70)

        return config
    
    def evaluate_on_val_set(self, data_splits: Dict) -> Dict:
        logger.info("Evaluating on val set...")

        test_dataset = FinancialSentimentDataset(
            data_splits['test']['texts'],
            data_splits['test']['labels'],
            self.tokenizer,
            max_length=128
        )

        test_metrics = self.trainer.evaluate(eval_dataset=test_dataset)

        logger.info(f"Test Metrics: {test_metrics}")
        predictions_output = self.trainer.predict(test_dataset)

        predictions = np.argmax(predictions_output.predictions, axis = 1)
        true_labels = predictions_output.label_ids

        #Convert to label names
        id_to_label = {0: 'positive', 1:'negative', 2:'neutral'}
        pred_labels = [id_to_label[p] for p in predictions]
        true_labels_text = [id_to_label[t] for t in true_labels]

        #Calculate metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1_macro': f1_score(true_labels, predictions, average='macro'),
            'f1_weighted': f1_score(true_labels, predictions, average='weighted'),
            'precision': precision_score(true_labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(true_labels, predictions, average='weighted', zero_division=0),
        }

        # pre_class Metrics
        class_report = classification_report(true_labels_text, pred_labels,target_names=['positive', 'negative', 'neutral'],
                                             output_dict=True)
        
        #confusion matrix
        cm = confusion_matrix(true_labels_text, pred_labels, labels=['positive', 'negative', 'neutral'])

        #Log results
        logger.info(f"\nTest Set Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")

        logger.info(f"\nPer-class F1 scores:")

        for label in ['positive', 'negative', 'neutral']:
            f1 = class_report[label]['f1-score']
            logger.info(f"  {label:8s}: {f1:.4f}")

        results = {
            'metrics': metrics,
            'class_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': pred_labels,
            'true_labels': true_labels_text
        }
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump({k: v for k, v in results.items() if k not in ['predictions', 'true_labels']}, f, indent=2)

        #Plot Confusion Matrix
        self._plot_confusion_matrix(cm, ['positive', 'negative', 'neutral'])
        
        logger.info(f"\nResults saved to {self.output_dir}")
        logger.info("="*70)
        
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, labels: List[str]):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved to {self.output_dir / 'confusion_matrix.png'}")


    def test_on_examples(self, examples: List[str]) -> List[Dict]:
        if self.model is None or self.tokenizer is None:
            logger.error("Model and tokenizer not loaded. Please load a trained model first.")
            return []
        
        self.model.eval()
        results = []

        with torch.no_grad():
            for text in examples:
                #tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=128,
                    truncation=True,
                    padding=True
                ).to(self.device)

                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                pred_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_id].item()

                # Map to level
                id_to_label = {0: 'positive', 1:'negative', 2:'neutral'}

                label = id_to_label[pred_id]

                results.append({
                    'text': text,
                    'predicted_label': label,
                    'confidence': confidence,
                    'probabilities': {
                        'positive': probs[0][0].item(),
                        'negative': probs[0][1].item(),
                        'neutral': probs[0][2].item()}
                })

        return results
    
def main():
    """Complete example workflow"""
    
    print("\n" + "="*70)
    print("FINBERT FINE-TUNING - COMPLETE EXAMPLE")
    print("="*70)
    
    # Initialize
    finetuner = FinbertFineTuner(output_dir='models/finbert_custom')
    
    # Step 1: Load and prepare data
    print("\n📊 Step 1: Loading and preparing data...")
    data_splits = finetuner.load_and_prepare_data(
        data_file='data/training/Financial_Sentiment_data_cleaned.csv',
        text_column='sentence',
        label_column='sentiment',
        balance_classes=True
    )
    
    # Step 2: Fine-tune
    print("\n🚀 Step 2: Fine-tuning FinBERT...")
    training_results = finetuner.Fine_Tune(
        data_splits,
        num_epochs=4,
        batch_size=16,
        learning_rate=2e-5
    )
    
    # Step 3: Evaluate on val set
    print("\n📈 Step 3: Evaluating on val set...")
    test_results = finetuner.evaluate_on_val_set(data_splits)
    
    # Step 4: Test on examples
    print("\n🧪 Step 4: Testing on example sentences...")
    examples = [
        "Revenue surged 45% year-over-year, crushing analyst expectations",
        "Stock plummeted 30% after catastrophic earnings miss",
        "Company announces routine board meeting scheduled for next month",
        "Revenue grew modestly despite challenging market conditions",
        "Beat earnings expectations but lowered full-year guidance",
        "Losses were not as bad as feared by analysts",
        "Q4 EPS of $2.45 beats consensus estimate of $2.20 by 11%",
        "Goldman Sachs upgrades to Buy with $180 price target"
    ]
    
    predictions = finetuner.test_on_examples(examples)
    
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    for pred in predictions:
        print(f"\nText: {pred['text']}")
        print(f"Prediction: {pred['predicted_label'].upper()} (confidence: {pred['confidence']:.2%})")
        print(f"Probabilities:")
        for label, prob in pred['probabilities'].items():
            print(f"  {label:8s}: {prob:.2%}")
    
    print("\n" + "="*70)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"Model saved to: {finetuner.output_dir}")
    print(f"Test F1 Score: {test_results['metrics']['f1_weighted']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
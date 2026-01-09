import json
import logging
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ENTITY DEFINITIONS ====================

# Financial-specific entity types
FINANCIAL_ENTITIES = {
    'ORG': 'Organization/Company',
    'PERSON': 'Person name',
    'MONEY': 'Monetary value',
    'DATE': 'Date/Time period',
    'PERCENT': 'Percentage',
    'PRODUCT': 'Product/Service',
    'LOC': 'Location',
    'TICKER': 'Stock ticker symbol',
    'METRIC': 'Financial metric (EPS, EBITDA, etc.)'
}

# BIO tagging scheme
# B- = Beginning of entity
# I- = Inside entity  
# O = Outside entity (not an entity)


# ==================== DATA PREPARATION ====================

class NERDataConverter:
    """
    Convert various data formats to NER training format
    
    Handles:
    - Raw text with entity annotations
    - CoNLL format
    - JSON format
    - Automatic BIO tag generation
    """
    
    def __init__(self):
        self.entity_types = list(FINANCIAL_ENTITIES.keys())
        
        # Create label list with BIO tags
        self.labels = ['O']  # Outside
        for entity_type in self.entity_types:
            self.labels.append(f'B-{entity_type}')  # Beginning
            self.labels.append(f'I-{entity_type}')  # Inside
        
        # Label to ID mapping
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        logger.info(f"Initialized with {len(self.labels)} labels: {self.labels[:10]}...")
    
    def convert_from_json(self, data: List[Dict]) -> List[Dict]:
        """
        Convert from JSON format to NER training format
        
        Input format:
        {
            "text": "Apple Inc. reported revenue of $90B",
            "entities": [
                {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10},
                {"text": "$90B", "label": "MONEY", "start": 31, "end": 35}
            ]
        }
        
        Output format:
        {
            "tokens": ["Apple", "Inc.", "reported", "revenue", "of", "$", "90B"],
            "ner_tags": ["B-ORG", "I-ORG", "O", "O", "O", "B-MONEY", "I-MONEY"]
        }
        """
        from nltk.tokenize import word_tokenize
        
        converted = []
        skipped = 0
        
        for idx, item in enumerate(data):
            text = item['text']
            entities = item.get('entities', [])
            
            # Tokenize
            try:
                tokens = word_tokenize(text)
            except Exception as e:
                logger.warning(f"Failed to tokenize item {idx}: {e}")
                skipped += 1
                continue
            
            # Initialize all tags as 'O'
            ner_tags = ['O'] * len(tokens)
            
            # Map entities to tokens
            char_to_token = self._create_char_to_token_mapping(text, tokens)
            
            for entity in entities:
                start_char = entity['start']
                end_char = entity['end']
                entity_type = entity['label']
                
                # Validate entity type
                if entity_type not in self.entity_types:
                    logger.warning(f"Unknown entity type '{entity_type}' in item {idx}, skipping")
                    continue
                
                # Find tokens covered by entity
                start_token = char_to_token.get(start_char)
                end_token = char_to_token.get(end_char - 1)
                
                if start_token is not None and end_token is not None:
                    # Assign BIO tags
                    ner_tags[start_token] = f'B-{entity_type}'
                    for i in range(start_token + 1, end_token + 1):
                        if i < len(ner_tags):
                            ner_tags[i] = f'I-{entity_type}'
            
            # Validate all tags are valid
            invalid_tags = [tag for tag in ner_tags if tag not in self.label2id]
            if invalid_tags:
                logger.warning(f"Invalid tags in item {idx}: {set(invalid_tags)}, skipping")
                skipped += 1
                continue
            
            converted.append({
                'tokens': tokens,
                'ner_tags': ner_tags
            })
        
        logger.info(f"Converted {len(converted)} examples, skipped {skipped}")
        return converted
    
    def _create_char_to_token_mapping(self, text: str, tokens: List[str]) -> Dict[int, int]:
        """
        Create mapping from character position to token index
        
        This handles the alignment between character-based entity positions
        and token-based labels.
        """
        mapping = {}
        current_pos = 0
        
        for token_idx, token in enumerate(tokens):
            # Find token in remaining text
            token_pos = text[current_pos:].find(token)
            
            if token_pos != -1:
                start = current_pos + token_pos
                end = start + len(token)
                
                # Map all characters in token to token index
                for char_pos in range(start, end):
                    mapping[char_pos] = token_idx
                
                current_pos = end
        
        return mapping
    
    def create_training_examples(self, 
                                financial_texts: List[str],
                                save_to: Optional[str] = None) -> List[Dict]:
        """
        Create training examples from raw financial texts
        
        This is a TEMPLATE - you need to manually annotate entities!
        
        Args:
            financial_texts: List of financial sentences
            save_to: Optional path to save annotation template
        
        Returns:
            List of examples ready for annotation
        """
        from nltk.tokenize import word_tokenize
        
        examples = []
        
        for text in financial_texts:
            tokens = word_tokenize(text)
            
            # Create template
            example = {
                'text': text,
                'tokens': tokens,
                'entities': [],  # TO BE ANNOTATED
                'annotated': False
            }
            
            examples.append(example)
        
        if save_to:
            with open(save_to, 'w') as f:
                json.dump(examples, f, indent=2)
            logger.info(f"Saved {len(examples)} examples to {save_to}")
            logger.info("Please annotate entities manually!")
        
        return examples


# ==================== NER DATASET ====================

class FinancialNERDataset(Dataset):
    """
    Dataset for Financial NER
    
    Handles:
    - Tokenization with subword alignment
    - Label alignment for subwords
    - Proper padding
    - Label validation
    """
    
    def __init__(self, 
                 examples: List[Dict],
                 tokenizer,
                 label2id: Dict[str, int],
                 max_length: int = 128):
        """
        Args:
            examples: List of {'tokens': [...], 'ner_tags': [...]}
            tokenizer: HuggingFace tokenizer
            label2id: Label to ID mapping
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        
        # Validate examples
        self._validate_examples()
    
    def _validate_examples(self):
        """Validate that all examples have valid labels"""
        num_labels = len(self.label2id)
        valid_examples = []
        
        for idx, example in enumerate(self.examples):
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            
            # Check if all tags are valid
            invalid = False
            for tag in ner_tags:
                if tag not in self.label2id:
                    logger.error(f"Example {idx}: Invalid tag '{tag}'")
                    invalid = True
                    break
            
            if not invalid:
                valid_examples.append(example)
        
        if len(valid_examples) < len(self.examples):
            logger.warning(f"Removed {len(self.examples) - len(valid_examples)} invalid examples")
            self.examples = valid_examples
        
        logger.info(f"Validated {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        
        # Tokenize with subword tokens
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels with subword tokens
        labels = self._align_labels(tokens, ner_tags, tokenized)
        
        # Validate labels are in valid range
        num_labels = len(self.label2id)
        for i, label in enumerate(labels):
            if label != -100 and (label < 0 or label >= num_labels):
                logger.error(f"Invalid label {label} at position {i} (valid range: 0-{num_labels-1})")
                logger.error(f"Tokens: {tokens}")
                logger.error(f"NER tags: {ner_tags}")
                raise ValueError(f"Label {label} out of range [0, {num_labels})")
        
        return {
            'input_ids': tokenized['input_ids'].flatten(),
            'attention_mask': tokenized['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _align_labels(self, tokens, ner_tags, tokenized):
        """
        Align labels with subword tokens
        
        Strategy: 
        - First subword of a word gets the label
        - Other subwords get -100 (ignored in loss)
        """
        word_ids = tokenized.word_ids(batch_index=0)
        labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special token (CLS, SEP, PAD)
                labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word
                if word_idx < len(ner_tags):
                    label = ner_tags[word_idx]
                    label_id = self.label2id.get(label, -100)
                    labels.append(label_id)
                else:
                    logger.warning(f"Word index {word_idx} out of range for ner_tags length {len(ner_tags)}")
                    labels.append(-100)
            else:
                # Subsequent subword - ignore in loss
                labels.append(-100)
            
            previous_word_idx = word_idx
        
        return labels
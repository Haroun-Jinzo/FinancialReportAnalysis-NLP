"""
Named Entity Recognition (NER) Module
Extracts financial entities: companies, people, locations, money, dates, metrics
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import spacy

from models.model_loader import ModelLoader


class FinancialNER:
    """
    Financial Named Entity Recognition
    Combines transformer-based NER with rule-based financial entity extraction
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize NER system
        
        Args:
            use_spacy: Use spaCy for additional entity extraction
        """
        print("Initializing Financial NER...")
        
        # Load transformer model
        self.loader = ModelLoader()
        self.model_data = self.loader.load_model('ner')
        self.pipeline = self.model_data['pipeline']
        
        # Load spaCy for additional features
        self.use_spacy = use_spacy
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                print("✓ spaCy loaded for enhanced NER")
            except:
                print("⚠ spaCy not available, using transformer only")
                self.use_spacy = False
        
        # Financial entity patterns
        self.patterns = {
            'MONEY': r'\$\s?[\d,]+\.?\d*\s?[MBK]?(?:illion)?',
            'PERCENTAGE': r'\d+\.?\d*\s?%',
            'DATE': r'\b(?:Q[1-4]|FY)\s?\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            'METRIC': r'\b(?:EPS|EBITDA|ROE|ROI|P/E|revenue|profit|income|margin|ratio)\b',
            'TICKER': r'\b[A-Z]{1,5}\b(?=\s|$|\))',
        }
        
        # Financial keywords for entity classification
        self.financial_terms = {
            'METRIC': [
                'revenue', 'profit', 'income', 'earnings', 'ebitda', 'eps',
                'margin', 'ratio', 'roe', 'roi', 'cash flow', 'assets',
                'liabilities', 'equity', 'debt', 'shares', 'dividend'
            ],
            'PERIOD': [
                'quarter', 'q1', 'q2', 'q3', 'q4', 'fiscal year', 'fy',
                'annual', 'monthly', 'quarterly', 'ytd', 'year-over-year'
            ]
        }
        
        print("✓ Financial NER initialized")
    
    def extract_entities(self, text: str, 
                        confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Extract all entities from text
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        # 1. Transformer-based NER
        transformer_entities = self._extract_transformer_entities(
            text, 
            confidence_threshold
        )
        entities.extend(transformer_entities)
        
        # 2. spaCy NER (if available)
        if self.use_spacy:
            spacy_entities = self._extract_spacy_entities(text)
            entities.extend(spacy_entities)
        
        # 3. Rule-based financial entities
        financial_entities = self._extract_financial_entities(text)
        entities.extend(financial_entities)
        
        # 4. Deduplicate and merge
        entities = self._deduplicate_entities(entities)
        
        # 5. Add context
        entities = self._add_context(text, entities)
        
        return entities
    
    def _extract_transformer_entities(self, text: str, 
                                     threshold: float) -> List[Dict]:
        """Extract entities using transformer model"""
        results = []
        
        try:
            # Run NER pipeline
            ner_results = self.pipeline(text)
            
            for entity in ner_results:
                if entity['score'] >= threshold:
                    results.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'score': entity['score'],
                        'start': entity['start'],
                        'end': entity['end'],
                        'source': 'transformer'
                    })
        
        except Exception as e:
            print(f"⚠ Transformer NER error: {e}")
        
        return results
    
    def _extract_spacy_entities(self, text: str) -> List[Dict]:
        """Extract entities using spaCy"""
        results = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                results.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'score': 1.0,  # spaCy doesn't provide scores
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'source': 'spacy'
                })
        
        except Exception as e:
            print(f"⚠ spaCy NER error: {e}")
        
        return results
    
    def _extract_financial_entities(self, text: str) -> List[Dict]:
        """Extract financial entities using regex patterns"""
        results = []
        
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                results.append({
                    'text': match.group(0),
                    'label': entity_type,
                    'score': 1.0,
                    'start': match.start(),
                    'end': match.end(),
                    'source': 'pattern'
                })
        
        return results
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Remove duplicate entities
        Prioritize: transformer > spacy > pattern
        """
        if not entities:
            return []
        
        # Sort by start position
        entities = sorted(entities, key=lambda x: x['start'])
        
        # Remove overlaps (keep highest priority)
        priority = {'transformer': 3, 'spacy': 2, 'pattern': 1}
        
        deduplicated = []
        for entity in entities:
            # Check if overlaps with existing entities
            overlaps = False
            for existing in deduplicated:
                if self._entities_overlap(entity, existing):
                    # Keep higher priority entity
                    if priority.get(entity['source'], 0) > priority.get(existing['source'], 0):
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated
    
    def _entities_overlap(self, ent1: Dict, ent2: Dict) -> bool:
        """Check if two entities overlap"""
        return not (ent1['end'] <= ent2['start'] or ent2['end'] <= ent1['start'])
    
    def _add_context(self, text: str, entities: List[Dict], 
                    context_window: int = 50) -> List[Dict]:
        """Add surrounding context to entities"""
        for entity in entities:
            start = max(0, entity['start'] - context_window)
            end = min(len(text), entity['end'] + context_window)
            entity['context'] = text[start:end].strip()
        
        return entities
    
    def extract_by_type(self, text: str, 
                       entity_type: str) -> List[Dict]:
        """
        Extract only specific entity type
        
        Args:
            text: Input text
            entity_type: Entity type (PERSON, ORG, MONEY, etc.)
            
        Returns:
            List of entities of specified type
        """
        all_entities = self.extract_entities(text)
        return [e for e in all_entities if e['label'].upper() == entity_type.upper()]
    
    def get_entity_summary(self, text: str) -> Dict:
        """
        Get summary of all entities in text
        
        Returns:
            Dictionary with entity counts by type
        """
        entities = self.extract_entities(text)
        
        summary = defaultdict(list)
        for entity in entities:
            summary[entity['label']].append(entity['text'])
        
        # Count unique entities
        result = {}
        for label, texts in summary.items():
            unique_texts = list(set(texts))
            result[label] = {
                'count': len(unique_texts),
                'examples': unique_texts[:5]  # Top 5 examples
            }
        
        return result
    
    def extract_financial_metrics(self, text: str) -> Dict:
        """
        Extract financial metrics with their values
        
        Returns:
            Dictionary mapping metrics to values
        """
        metrics = {}
        
        # Extract all entities
        entities = self.extract_entities(text)
        
        # Find metric-value pairs
        for i, entity in enumerate(entities):
            if entity['label'] in ['METRIC', 'ORG']:
                # Look for nearby money/percentage values
                for j in range(max(0, i-2), min(len(entities), i+3)):
                    if entities[j]['label'] in ['MONEY', 'PERCENTAGE']:
                        metric_name = entity['text']
                        value = entities[j]['text']
                        metrics[metric_name] = value
                        break
        
        return metrics
    
    def visualize_entities(self, text: str, max_length: int = 500) -> str:
        """
        Create a simple text visualization of entities
        
        Args:
            text: Input text
            max_length: Maximum text length to display
            
        Returns:
            Formatted string with highlighted entities
        """
        entities = self.extract_entities(text)
        
        # Truncate text if too long
        display_text = text[:max_length] + "..." if len(text) > max_length else text
        
        # Create visualization
        result = ["\n" + "="*60]
        result.append("ENTITY EXTRACTION RESULTS")
        result.append("="*60 + "\n")
        
        # Group by type
        by_type = defaultdict(list)
        for entity in entities:
            if entity['start'] < max_length:  # Only show entities in displayed text
                by_type[entity['label']].append(entity['text'])
        
        # Display by type
        for label, texts in sorted(by_type.items()):
            unique_texts = list(set(texts))
            result.append(f"{label} ({len(unique_texts)}):")
            for text in unique_texts[:10]:  # Max 10 per type
                result.append(f"  • {text}")
            result.append("")
        
        return "\n".join(result)


# Example usage
if __name__ == "__main__":
    # Initialize NER
    ner = FinancialNER()
    
    # Test text
    sample_text = """
    Apple Inc. (AAPL) reported Q3 2024 revenue of $90.1 billion, 
    representing a 15.2% increase year-over-year. Net income was 
    $25.5 billion with EPS of $1.85. CEO Tim Cook stated that 
    iPhone sales grew 12% to $45.8 billion in the quarter.
    """
    
    print("\nTest Text:")
    print(sample_text)
    
    # Extract entities
    entities = ner.extract_entities(sample_text)
    
    print(f"\n✓ Found {len(entities)} entities")
    
    # Show summary
    summary = ner.get_entity_summary(sample_text)
    print("\nEntity Summary:")
    for label, info in summary.items():
        print(f"  {label}: {info['count']} - {info['examples']}")
    
    # Extract financial metrics
    metrics = ner.extract_financial_metrics(sample_text)
    print("\nFinancial Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    print("\n✓ NER Module Ready!")
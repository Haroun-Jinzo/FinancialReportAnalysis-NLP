"""
Advanced Entity Extractor
Combines NER, patterns, and domain knowledge for comprehensive extraction
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ner_model import FinancialNER 


class EntityExtractor:
    """
    Advanced entity extraction system
    Extracts and structures financial entities with metadata
    """
    
    def __init__(self):
        """Initialize entity extractor"""
        print("Initializing Entity Extractor...")
        
        # Load NER model
        self.ner = FinancialNER()
        
        # Load extraction rules
        self.rules = self._load_extraction_rules()
        
        # Entity type mappings
        self.entity_types = {
            'COMPANY': ['ORG', 'ORGANIZATION'],
            'PERSON': ['PER', 'PERSON'],
            'MONEY': ['MONEY', 'MONETARY'],
            'DATE': ['DATE', 'TIME'],
            'LOCATION': ['LOC', 'LOCATION', 'GPE'],
            'PERCENTAGE': ['PERCENT', 'PERCENTAGE'],
            'METRIC': ['METRIC', 'MEASURE']
        }
        
        print("✓ Entity Extractor initialized")
    
    def _load_extraction_rules(self) -> Dict:
        """Load extraction rules from config"""
        try:
            with open('config/extraction_rules.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_rules()
    
    def _default_rules(self) -> Dict:
        """Default extraction rules"""
        return {
            'financial_entities': {
                'MONEY': ['revenue', 'profit', 'loss', 'earnings', 'income', 'sales'],
                'METRIC': ['EBITDA', 'EPS', 'ROE', 'ROI', 'margin', 'ratio', 'P/E'],
                'PERIOD': ['Q1', 'Q2', 'Q3', 'Q4', 'FY', 'fiscal year', 'quarter'],
                'PERCENTAGE': ['growth', 'increase', 'decrease', 'change', 'rate']
            },
            'patterns': {
                'revenue': r'revenue\s*(?:of|was|is|reached|totaled)?\s*\$?[\d,]+\.?\d*\s?[MBK]?(?:illion)?',
                'profit': r'(?:net\s+)?profit\s*(?:of|was|is)?\s*\$?[\d,]+\.?\d*\s?[MBK]?',
                'eps': r'EPS\s*(?:of|was|is)?\s*\$?[\d.]+',
                'growth': r'(?:grew|increased|rose)\s*(?:by\s*)?[\d.]+%',
                'decline': r'(?:declined|decreased|fell)\s*(?:by\s*)?[\d.]+%'
            }
        }
    
    def extract(self, text: str, 
                extract_relations: bool = True) -> Dict:
        """
        Main extraction function
        
        Args:
            text: Input text
            extract_relations: Also extract relationships
            
        Returns:
            Dictionary with all extracted information
        """
        result = {
            'entities': [],
            'metrics': {},
            'relations': [],
            'metadata': {}
        }
        
        # 1. Extract base entities using NER
        entities = self.ner.extract_entities(text)
        
        # 2. Normalize and categorize entities
        normalized_entities = self._normalize_entities(entities)
        result['entities'] = normalized_entities
        
        # 3. Extract financial metrics
        metrics = self._extract_metrics(text, normalized_entities)
        result['metrics'] = metrics
        
        # 4. Extract relations if requested
        if extract_relations:
            relations = self._extract_relations(text, normalized_entities)
            result['relations'] = relations
        
        # 5. Add metadata
        result['metadata'] = {
            'total_entities': len(normalized_entities),
            'entity_types': self._count_by_type(normalized_entities),
            'metrics_found': len(metrics),
            'relations_found': len(result['relations'])
        }
        
        return result
    
    def _normalize_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Normalize and enrich entities
        """
        normalized = []
        
        for entity in entities:
            # Map to standard type
            standard_type = self._map_entity_type(entity['label'])
            
            # Create normalized entity
            norm_entity = {
                'text': entity['text'],
                'type': standard_type,
                'original_label': entity['label'],
                'confidence': entity['score'],
                'start': entity['start'],
                'end': entity['end'],
                'source': entity['source']
            }
            
            # Add type-specific processing
            if standard_type == 'MONEY':
                norm_entity['normalized_value'] = self._normalize_money(entity['text'])
            elif standard_type == 'PERCENTAGE':
                norm_entity['normalized_value'] = self._normalize_percentage(entity['text'])
            elif standard_type == 'DATE':
                norm_entity['normalized_value'] = self._normalize_date(entity['text'])
            
            # Add context
            if 'context' in entity:
                norm_entity['context'] = entity['context']
            
            normalized.append(norm_entity)
        
        return normalized
    
    def _map_entity_type(self, label: str) -> str:
        """Map various entity labels to standard types"""
        label_upper = label.upper()
        
        for standard_type, variations in self.entity_types.items():
            if label_upper in variations:
                return standard_type
        
        return label_upper
    
    def _normalize_money(self, text: str) -> Dict:
        """
        Normalize money values to standard format
        
        Examples:
            "$90.1B" -> {'value': 90.1, 'unit': 'billion', 'currency': 'USD'}
            "5.2M" -> {'value': 5.2, 'unit': 'million', 'currency': 'USD'}
        """
        # Extract numeric value
        value_match = re.search(r'[\d,]+\.?\d*', text)
        if not value_match:
            return {'value': None, 'unit': 'unknown', 'currency': 'USD'}
        
        value = float(value_match.group().replace(',', ''))
        
        # Determine multiplier
        text_upper = text.upper()
        if 'B' in text_upper or 'BILLION' in text_upper:
            unit = 'billion'
            multiplier = 1e9
        elif 'M' in text_upper or 'MILLION' in text_upper:
            unit = 'million'
            multiplier = 1e6
        elif 'K' in text_upper or 'THOUSAND' in text_upper:
            unit = 'thousand'
            multiplier = 1e3
        else:
            unit = 'units'
            multiplier = 1
        
        # Determine currency
        currency = 'USD'  # Default
        if '€' in text or 'EUR' in text_upper:
            currency = 'EUR'
        elif '£' in text or 'GBP' in text_upper:
            currency = 'GBP'
        
        return {
            'value': value,
            'unit': unit,
            'multiplier': multiplier,
            'absolute_value': value * multiplier,
            'currency': currency,
            'formatted': f"{currency} {value}{unit[0].upper()}"
        }
    
    def _normalize_percentage(self, text: str) -> Dict:
        """Normalize percentage values"""
        value_match = re.search(r'[\d.]+', text)
        if not value_match:
            return {'value': None}
        
        value = float(value_match.group())
        
        return {
            'value': value,
            'decimal': value / 100,
            'formatted': f"{value}%"
        }
    
    def _normalize_date(self, text: str) -> Dict:
        """Normalize date/period values"""
        text_upper = text.upper()
        
        # Fiscal quarter pattern
        quarter_match = re.search(r'Q([1-4])\s*(\d{4})', text_upper)
        if quarter_match:
            return {
                'type': 'quarter',
                'quarter': int(quarter_match.group(1)),
                'year': int(quarter_match.group(2)),
                'formatted': f"Q{quarter_match.group(1)} {quarter_match.group(2)}"
            }
        
        # Fiscal year pattern
        fy_match = re.search(r'FY\s*(\d{4})', text_upper)
        if fy_match:
            return {
                'type': 'fiscal_year',
                'year': int(fy_match.group(1)),
                'formatted': f"FY {fy_match.group(1)}"
            }
        
        # Try to parse as standard date
        return {
            'type': 'date',
            'text': text,
            'formatted': text
        }
    
    def _extract_metrics(self, text: str, 
                        entities: List[Dict]) -> Dict:
        """
        Extract financial metrics with their values
        """
        metrics = {}
        
        # Pattern-based extraction
        for metric_name, pattern in self.rules.get('patterns', {}).items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Extract the full match
                full_text = match.group(0)
                
                # Try to find associated value
                value = self._extract_value_from_text(full_text)
                
                if value:
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    
                    metrics[metric_name].append({
                        'text': full_text,
                        'value': value,
                        'position': match.start()
                    })
        
        # Entity-based extraction (find metric-value pairs)
        for i, entity in enumerate(entities):
            if entity['type'] in ['METRIC', 'MONEY', 'PERCENTAGE']:
                # Look for context to determine what this measures
                context = entity.get('context', '')
                metric_type = self._identify_metric_type(context, entity['text'])
                
                if metric_type:
                    if metric_type not in metrics:
                        metrics[metric_type] = []
                    
                    metrics[metric_type].append({
                        'text': entity['text'],
                        'value': entity.get('normalized_value'),
                        'confidence': entity['confidence']
                    })
        
        return metrics
    
    def _extract_value_from_text(self, text: str) -> Optional[Dict]:
        """Extract numeric value from text"""
        # Try money pattern
        money_match = re.search(r'\$?[\d,]+\.?\d*\s?[MBK]?', text)
        if money_match:
            return self._normalize_money(money_match.group())
        
        # Try percentage pattern
        pct_match = re.search(r'[\d.]+%', text)
        if pct_match:
            return self._normalize_percentage(pct_match.group())
        
        return None
    
    def _identify_metric_type(self, context: str, text: str) -> Optional[str]:
        """Identify what type of metric this is based on context"""
        context_lower = context.lower()
        text_lower = text.lower()
        
        metric_keywords = {
            'revenue': ['revenue', 'sales', 'turnover'],
            'profit': ['profit', 'income', 'earnings', 'margin'],
            'eps': ['eps', 'earnings per share'],
            'growth': ['growth', 'increase', 'grew'],
            'dividend': ['dividend', 'payout']
        }
        
        for metric_type, keywords in metric_keywords.items():
            if any(kw in context_lower or kw in text_lower for kw in keywords):
                return metric_type
        
        return None
    
    def _extract_relations(self, text: str, 
                          entities: List[Dict]) -> List[Dict]:
        """
        Extract relationships between entities
        
        Examples:
            - Company reported Revenue
            - CEO stated that...
            - Revenue increased by Percentage
        """
        relations = []
        
        # Common relation patterns
        relation_patterns = [
            # Company - Action - Metric
            {
                'pattern': r'(\w+(?:\s+\w+)?)\s+(reported|announced|posted|generated)\s+([^.]+)',
                'relation': 'REPORTED',
                'subject_type': 'COMPANY',
                'object_type': 'MONEY'
            },
            # Metric - Change - Percentage
            {
                'pattern': r'(\w+)\s+(increased|decreased|grew|fell|rose|dropped)\s+(?:by\s+)?([\d.]+%)',
                'relation': 'CHANGED_BY',
                'subject_type': 'METRIC',
                'object_type': 'PERCENTAGE'
            },
            # Person - Role - Company
            {
                'pattern': r'(?:CEO|CFO|President|Chairman)\s+(\w+(?:\s+\w+)?)\s+(?:of|at)\s+(\w+(?:\s+\w+)?)',
                'relation': 'ROLE_AT',
                'subject_type': 'PERSON',
                'object_type': 'COMPANY'
            }
        ]
        
        for pattern_info in relation_patterns:
            matches = re.finditer(pattern_info['pattern'], text, re.IGNORECASE)
            
            for match in matches:
                relations.append({
                    'subject': match.group(1),
                    'relation': pattern_info['relation'],
                    'object': match.group(3) if match.lastindex >= 3 else match.group(2),
                    'text': match.group(0),
                    'position': match.start()
                })
        
        return relations
    
    def _count_by_type(self, entities: List[Dict]) -> Dict:
        """Count entities by type"""
        counts = defaultdict(int)
        for entity in entities:
            counts[entity['type']] += 1
        return dict(counts)
    
    def extract_by_type(self, text: str, 
                       entity_type: str) -> List[Dict]:
        """Extract only specific entity type"""
        result = self.extract(text, extract_relations=False)
        return [e for e in result['entities'] if e['type'] == entity_type]
    
    def extract_companies(self, text: str) -> List[str]:
        """Extract all company names"""
        companies = self.extract_by_type(text, 'COMPANY')
        return [c['text'] for c in companies]
    
    def extract_money_values(self, text: str) -> List[Dict]:
        """Extract all money values with normalization"""
        money_entities = self.extract_by_type(text, 'MONEY')
        return [
            {
                'text': e['text'],
                'value': e.get('normalized_value'),
                'context': e.get('context', '')
            }
            for e in money_entities
        ]
    
    def get_extraction_summary(self, text: str) -> Dict:
        """Get summary of extraction results"""
        result = self.extract(text)
        
        return {
            'total_entities': result['metadata']['total_entities'],
            'by_type': result['metadata']['entity_types'],
            'metrics': list(result['metrics'].keys()),
            'relations': len(result['relations']),
            'top_entities': [
                {'type': e['type'], 'text': e['text'], 'confidence': e['confidence']}
                for e in sorted(result['entities'], 
                              key=lambda x: x['confidence'], 
                              reverse=True)[:5]
            ]
        }


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = EntityExtractor()
    
    # Sample text
    sample_text = """
    Apple Inc. reported Q3 2024 revenue of $90.1 billion, up 15.2% year-over-year.
    Net income reached $25.5 billion with EPS of $1.85. CEO Tim Cook stated that
    iPhone sales grew 12% to $45.8 billion. The company announced a dividend of
    $0.24 per share.
    """
    
    print("\nExtracting entities...")
    result = extractor.extract(sample_text)
    
    print(f"\n✓ Extraction complete!")
    print(f"  Total entities: {result['metadata']['total_entities']}")
    print(f"  Entity types: {result['metadata']['entity_types']}")
    print(f"  Metrics found: {result['metadata']['metrics_found']}")
    print(f"  Relations found: {result['metadata']['relations_found']}")
    
    # Show sample entities
    print("\nSample entities:")
    for entity in result['entities'][:5]:
        value_str = f" = {entity.get('normalized_value', {}).get('formatted', '')}" if 'normalized_value' in entity else ""
        print(f"  {entity['type']:15s}: {entity['text']}{value_str}")
    
    # Show metrics
    print("\nMetrics:")
    for metric, values in list(result['metrics'].items())[:3]:
        print(f"  {metric}: {values[0]['text'] if values else 'N/A'}")
    
    print("\n✓ Entity Extractor Module Ready!")
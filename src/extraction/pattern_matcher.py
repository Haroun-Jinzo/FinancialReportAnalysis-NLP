"""
Pattern Matcher
Rule-based pattern matching for financial texts
"""

import re
from typing import List, Dict, Pattern, Optional, Callable
from dataclasses import dataclass


@dataclass
class PatternMatch:
    """Data class for pattern matches"""
    pattern_name: str
    matched_text: str
    start: int
    end: int
    groups: Dict[str, str]
    confidence: float
    
    def __str__(self):
        return f"{self.pattern_name}: {self.matched_text}"


class PatternMatcher:
    """
    Flexible pattern matching system for financial documents
    """
    
    def __init__(self):
        """Initialize pattern matcher"""
        print("Initializing Pattern Matcher...")
        
        # Compiled patterns
        self.patterns = self._init_patterns()
        
        # Custom extractors
        self.extractors = self._init_extractors()
        
        print("✓ Pattern Matcher initialized")
    
    def _init_patterns(self) -> Dict[str, Dict]:
        """Initialize pattern library"""
        return {
            # Financial values
            'money_value': {
                'pattern': r'\$\s?([\d,]+\.?\d*)\s?(billion|million|thousand|B|M|K)?',
                'confidence': 0.95,
                'groups': ['value', 'unit']
            },
            'percentage': {
                'pattern': r'([\d.]+)%',
                'confidence': 0.95,
                'groups': ['value']
            },
            
            # Time periods
            'fiscal_quarter': {
                'pattern': r'Q([1-4])\s*(20\d{2})',
                'confidence': 0.9,
                'groups': ['quarter', 'year']
            },
            'fiscal_year': {
                'pattern': r'(?:FY|fiscal year)\s*(20\d{2})',
                'confidence': 0.9,
                'groups': ['year']
            },
            'year_over_year': {
                'pattern': r'year[- ]over[- ]year|YoY|y-o-y',
                'confidence': 0.85,
                'groups': []
            },
            'quarter_over_quarter': {
                'pattern': r'quarter[- ]over[- ]quarter|QoQ|q-o-q',
                'confidence': 0.85,
                'groups': []
            },
            
            # Performance indicators
            'growth_statement': {
                'pattern': r'(revenue|sales|profit|income|earnings)\s+(grew|increased|rose|jumped|surged)\s+(?:by\s+)?([\d.]+)%',
                'confidence': 0.9,
                'groups': ['metric', 'verb', 'amount']
            },
            'decline_statement': {
                'pattern': r'(revenue|sales|profit|income|earnings)\s+(declined|decreased|fell|dropped|slumped)\s+(?:by\s+)?([\d.]+)%',
                'confidence': 0.9,
                'groups': ['metric', 'verb', 'amount']
            },
            'beat_expectations': {
                'pattern': r'(beat|exceeded|surpassed|topped)\s+(analyst\s+)?expectations',
                'confidence': 0.85,
                'groups': ['verb']
            },
            'miss_expectations': {
                'pattern': r'(missed|fell short of|below)\s+(analyst\s+)?expectations',
                'confidence': 0.85,
                'groups': ['verb']
            },
            
            # Financial metrics
            'eps_statement': {
                'pattern': r'(?:diluted\s+)?(?:EPS|earnings per share)\s+(?:of|was|is)\s+\$([\d.]+)',
                'confidence': 0.9,
                'groups': ['value']
            },
            'revenue_statement': {
                'pattern': r'revenue\s+(?:of|was|reached|totaled)\s+\$([\d,.]+)\s?(billion|million|B|M)?',
                'confidence': 0.9,
                'groups': ['value', 'unit']
            },
            'profit_statement': {
                'pattern': r'(?:net\s+)?profit\s+(?:of|was)\s+\$([\d,.]+)\s?(billion|million|B|M)?',
                'confidence': 0.9,
                'groups': ['value', 'unit']
            },
            'margin_statement': {
                'pattern': r'(gross|operating|net|profit)\s+margin\s+(?:of|was|is)\s+([\d.]+)%',
                'confidence': 0.9,
                'groups': ['type', 'value']
            },
            
            # Company info
            'ticker_symbol': {
                'pattern': r'\(([A-Z]{1,5})\)',
                'confidence': 0.85,
                'groups': ['ticker']
            },
            'company_headquarters': {
                'pattern': r'(?:headquartered|based)\s+in\s+([\w\s,]+)',
                'confidence': 0.8,
                'groups': ['location']
            },
            
            # Market cap and valuation
            'market_cap': {
                'pattern': r'market\s+(?:cap|capitalization)\s+of\s+\$([\d,.]+)\s?(billion|million|trillion|B|M|T)?',
                'confidence': 0.85,
                'groups': ['value', 'unit']
            },
            'pe_ratio': {
                'pattern': r'P/E\s+ratio\s+of\s+([\d.]+)',
                'confidence': 0.85,
                'groups': ['value']
            },
            
            # Guidance and forecasts
            'guidance': {
                'pattern': r'(?:guidance|forecast|expects|anticipates)\s+(?:revenue|earnings|sales)\s+(?:of|to be)\s+\$([\d,.]+)\s?(billion|million|B|M)?',
                'confidence': 0.8,
                'groups': ['value', 'unit']
            },
            
            # Risk indicators
            'risk_mention': {
                'pattern': r'\b(risk|uncertainty|challenge|headwind|concern|volatility)\b',
                'confidence': 0.7,
                'groups': ['term']
            },
            'positive_indicator': {
                'pattern': r'\b(opportunity|strength|tailwind|growth|expansion|innovation)\b',
                'confidence': 0.7,
                'groups': ['term']
            }
        }
    
    def _init_extractors(self) -> Dict[str, Callable]:
        """Initialize custom extractors"""
        return {
            'financial_ratios': self._extract_financial_ratios,
            'comparative_statements': self._extract_comparative_statements,
            'forward_looking': self._extract_forward_looking
        }
    
    def match_all(self, text: str, 
                  pattern_names: Optional[List[str]] = None) -> Dict[str, List[PatternMatch]]:
        """
        Match all patterns in text
        
        Args:
            text: Input text
            pattern_names: Optional list of specific patterns to match
            
        Returns:
            Dictionary of matches by pattern name
        """
        results = {}
        
        # Determine which patterns to use
        patterns_to_match = pattern_names if pattern_names else self.patterns.keys()
        
        for pattern_name in patterns_to_match:
            if pattern_name not in self.patterns:
                continue
            
            matches = self.match_pattern(text, pattern_name)
            if matches:
                results[pattern_name] = matches
        
        return results
    
    def match_pattern(self, text: str, pattern_name: str) -> List[PatternMatch]:
        """Match a specific pattern"""
        if pattern_name not in self.patterns:
            return []
        
        pattern_config = self.patterns[pattern_name]
        pattern = pattern_config['pattern']
        
        matches = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Extract groups
            groups = {}
            if 'groups' in pattern_config and pattern_config['groups']:
                for i, group_name in enumerate(pattern_config['groups'], 1):
                    if i <= match.lastindex:
                        groups[group_name] = match.group(i)
            
            pattern_match = PatternMatch(
                pattern_name=pattern_name,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                groups=groups,
                confidence=pattern_config['confidence']
            )
            
            matches.append(pattern_match)
        
        return matches
    
    def _extract_financial_ratios(self, text: str) -> List[Dict]:
        """Extract financial ratios"""
        ratios = []
        
        ratio_patterns = {
            'ROE': r'(?:ROE|return on equity)\s+(?:of|was|is)\s+([\d.]+)%',
            'ROA': r'(?:ROA|return on assets)\s+(?:of|was|is)\s+([\d.]+)%',
            'ROI': r'(?:ROI|return on investment)\s+(?:of|was|is)\s+([\d.]+)%',
            'debt_to_equity': r'debt[- ]to[- ]equity\s+(?:ratio\s+)?(?:of|was|is)\s+([\d.]+)',
            'current_ratio': r'current\s+ratio\s+(?:of|was|is)\s+([\d.]+)',
            'quick_ratio': r'quick\s+ratio\s+(?:of|was|is)\s+([\d.]+)'
        }
        
        for ratio_name, pattern in ratio_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ratios.append({
                    'name': ratio_name,
                    'value': float(match.group(1)),
                    'text': match.group(0),
                    'position': match.start()
                })
        
        return ratios
    
    def _extract_comparative_statements(self, text: str) -> List[Dict]:
        """Extract comparative statements"""
        comparisons = []
        
        comparison_patterns = [
            r'([\w\s]+)\s+compared to\s+([\w\s]+)',
            r'([\w\s]+)\s+versus\s+([\w\s]+)',
            r'([\w\s]+)\s+vs\.?\s+([\w\s]+)',
            r'higher than\s+([\w\s]+)',
            r'lower than\s+([\w\s]+)',
            r'outperformed\s+([\w\s]+)',
            r'underperformed\s+([\w\s]+)'
        ]
        
        for pattern in comparison_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                comparisons.append({
                    'text': match.group(0),
                    'position': match.start(),
                    'type': 'comparison'
                })
        
        return comparisons
    
    def _extract_forward_looking(self, text: str) -> List[Dict]:
        """Extract forward-looking statements"""
        forward = []
        
        forward_indicators = [
            'expect', 'anticipate', 'forecast', 'project', 'estimate',
            'believe', 'plan', 'intend', 'may', 'will', 'should',
            'could', 'outlook', 'guidance', 'target'
        ]
        
        pattern = r'\b(' + '|'.join(forward_indicators) + r')\b[^.]{1,200}'
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            forward.append({
                'text': match.group(0),
                'indicator': match.group(1),
                'position': match.start()
            })
        
        return forward
    
    def extract_custom(self, text: str, extractor_name: str) -> List[Dict]:
        """Use a custom extractor"""
        if extractor_name in self.extractors:
            return self.extractors[extractor_name](text)
        return []
    
    def get_match_statistics(self, text: str) -> Dict:
        """Get statistics on pattern matches"""
        all_matches = self.match_all(text)
        
        stats = {
            'total_matches': sum(len(matches) for matches in all_matches.values()),
            'by_pattern': {
                name: len(matches)
                for name, matches in all_matches.items()
            },
            'avg_confidence': 0
        }
        
        # Calculate average confidence
        all_confidences = [m.confidence for matches in all_matches.values() for m in matches]
        if all_confidences:
            stats['avg_confidence'] = sum(all_confidences) / len(all_confidences)
        
        return stats
    
    def add_custom_pattern(self, name: str, pattern: str, 
                          confidence: float = 0.8, 
                          groups: Optional[List[str]] = None):
        """Add a custom pattern at runtime"""
        self.patterns[name] = {
            'pattern': pattern,
            'confidence': confidence,
            'groups': groups or []
        }


# Example usage
if __name__ == "__main__":
    # Initialize matcher
    matcher = PatternMatcher()
    
    # Sample text
    sample_text = """
    Apple Inc. (AAPL) reported Q3 2024 revenue of $90.1 billion, up 15.2% year-over-year.
    Net profit was $25.5 billion with diluted EPS of $1.85, beating analyst expectations.
    Gross margin improved to 46.3% while operating margin was 30.2%. The company expects
    Q4 revenue to be between $88-92 billion. ROE stands at 25% compared to 22% last year.
    """
    
    print("\nMatching patterns...")
    matches = matcher.match_all(sample_text)
    
    print(f"\n✓ Found {sum(len(m) for m in matches.values())} total matches")
    print(f"  Across {len(matches)} different patterns")
    
    # Show matches by category
    print("\n" + "="*60)
    print("PATTERN MATCHES")
    print("="*60)
    
    for pattern_name, pattern_matches in list(matches.items())[:10]:
        print(f"\n{pattern_name}:")
        for match in pattern_matches[:3]:  # Show first 3 of each
            print(f"  • {match.matched_text}")
            if match.groups:
                for key, value in match.groups.items():
                    print(f"    {key}: {value}")
    
    # Extract custom patterns
    print("\n" + "="*60)
    print("CUSTOM EXTRACTIONS")
    print("="*60)
    
    ratios = matcher.extract_custom(sample_text, 'financial_ratios')
    print(f"\nFinancial Ratios: {len(ratios)}")
    for ratio in ratios:
        print(f"  {ratio['name']}: {ratio['value']}")
    
    forward = matcher.extract_custom(sample_text, 'forward_looking')
    print(f"\nForward-Looking Statements: {len(forward)}")
    for stmt in forward[:3]:
        print(f"  • {stmt['text'][:80]}...")
    
    # Statistics
    stats = matcher.get_match_statistics(sample_text)
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total matches: {stats['total_matches']}")
    print(f"Avg confidence: {stats['avg_confidence']:.2f}")
    
    print("\n✓ Pattern Matcher Module Ready!")
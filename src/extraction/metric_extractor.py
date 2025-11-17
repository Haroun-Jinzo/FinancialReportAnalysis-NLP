"""
Financial Metric Extractor
Specialized extraction and calculation of financial metrics
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FinancialMetric:
    """Data class for financial metrics"""
    name: str
    value: float
    unit: str
    currency: str
    period: Optional[str]
    confidence: float
    raw_text: str
    
    def __str__(self):
        return f"{self.name}: {self.currency} {self.value}{self.unit} ({self.period})"


class MetricExtractor:
    """
    Extract and normalize financial metrics from text
    """
    
    def __init__(self):
        """Initialize metric extractor"""
        print("Initializing Metric Extractor...")
        
        # Define metric patterns with their extraction rules
        self.metric_patterns = {
            'revenue': {
                'keywords': ['revenue', 'sales', 'turnover', 'top line'],
                'pattern': r'(?:revenue|sales|turnover)\s*(?:of|was|is|reached|totaled)?\s*(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(million|billion|M|B|K)?',
                'category': 'income_statement'
            },
            'net_income': {
                'keywords': ['net income', 'net profit', 'net earnings', 'bottom line'],
                'pattern': r'(?:net\s+)?(?:income|profit|earnings)\s*(?:of|was|is)?\s*(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(million|billion|M|B|K)?',
                'category': 'income_statement'
            },
            'gross_profit': {
                'keywords': ['gross profit', 'gross income'],
                'pattern': r'gross\s+(?:profit|income)\s*(?:of|was|is)?\s*(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(million|billion|M|B|K)?',
                'category': 'income_statement'
            },
            'operating_income': {
                'keywords': ['operating income', 'operating profit', 'EBIT'],
                'pattern': r'(?:operating\s+(?:income|profit)|EBIT)\s*(?:of|was|is)?\s*(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(million|billion|M|B|K)?',
                'category': 'income_statement'
            },
            'ebitda': {
                'keywords': ['EBITDA', 'adjusted EBITDA'],
                'pattern': r'EBITDA\s*(?:of|was|is)?\s*(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(million|billion|M|B|K)?',
                'category': 'income_statement'
            },
            'eps': {
                'keywords': ['EPS', 'earnings per share', 'diluted EPS'],
                'pattern': r'(?:diluted\s+)?(?:EPS|earnings\s+per\s+share)\s*(?:of|was|is)?\s*(?:\$|USD)?\s*([\d.]+)',
                'category': 'per_share'
            },
            'dividend': {
                'keywords': ['dividend', 'dividend per share'],
                'pattern': r'dividend\s*(?:per\s+share)?\s*(?:of|was|is)?\s*(?:\$|USD)?\s*([\d.]+)',
                'category': 'per_share'
            },
            'gross_margin': {
                'keywords': ['gross margin', 'gross profit margin'],
                'pattern': r'gross\s+(?:profit\s+)?margin\s*(?:of|was|is)?\s*([\d.]+)%',
                'category': 'margin'
            },
            'operating_margin': {
                'keywords': ['operating margin'],
                'pattern': r'operating\s+margin\s*(?:of|was|is)?\s*([\d.]+)%',
                'category': 'margin'
            },
            'net_margin': {
                'keywords': ['net margin', 'profit margin', 'net profit margin'],
                'pattern': r'(?:net\s+)?(?:profit\s+)?margin\s*(?:of|was|is)?\s*([\d.]+)%',
                'category': 'margin'
            },
            'total_assets': {
                'keywords': ['total assets', 'assets'],
                'pattern': r'total\s+assets\s*(?:of|was|were)?\s*(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(million|billion|M|B|K)?',
                'category': 'balance_sheet'
            },
            'total_liabilities': {
                'keywords': ['total liabilities', 'liabilities'],
                'pattern': r'total\s+liabilities\s*(?:of|was|were)?\s*(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(million|billion|M|B|K)?',
                'category': 'balance_sheet'
            },
            'shareholders_equity': {
                'keywords': ['shareholders equity', 'stockholders equity', 'equity'],
                'pattern': r"(?:shareholders?|stockholders?)\s+equity\s*(?:of|was)?\s*(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(million|billion|M|B|K)?",
                'category': 'balance_sheet'
            },
            'cash': {
                'keywords': ['cash', 'cash and equivalents'],
                'pattern': r'cash\s*(?:and\s+(?:cash\s+)?equivalents)?\s*(?:of|was)?\s*(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(million|billion|M|B|K)?',
                'category': 'balance_sheet'
            },
            'debt': {
                'keywords': ['total debt', 'long-term debt', 'debt'],
                'pattern': r'(?:total\s+|long-term\s+)?debt\s*(?:of|was)?\s*(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(million|billion|M|B|K)?',
                'category': 'balance_sheet'
            }
        }
        
        # Growth and change patterns
        self.change_patterns = {
            'growth': r'(?:grew|increased|rose|jumped|surged)\s*(?:by\s*)?([\d.]+)%',
            'decline': r'(?:declined|decreased|fell|dropped|slumped)\s*(?:by\s*)?([\d.]+)%',
            'yoy': r'([\d.]+)%\s*(?:year-over-year|YoY|y-o-y)',
            'qoq': r'([\d.]+)%\s*(?:quarter-over-quarter|QoQ|q-o-q)'
        }
        
        print("✓ Metric Extractor initialized")
    
    def extract_all_metrics(self, text: str, 
                           period: Optional[str] = None) -> Dict[str, List[FinancialMetric]]:
        """
        Extract all financial metrics from text
        
        Args:
            text: Input text
            period: Optional period identifier (e.g., "Q3 2024")
            
        Returns:
            Dictionary of metric lists by category
        """
        metrics_by_category = {
            'income_statement': [],
            'balance_sheet': [],
            'per_share': [],
            'margin': [],
            'ratios': []
        }
        
        # Extract each metric type
        for metric_name, config in self.metric_patterns.items():
            extracted = self._extract_metric(
                text, 
                metric_name, 
                config, 
                period
            )
            
            if extracted:
                category = config['category']
                metrics_by_category[category].extend(extracted)
        
        return metrics_by_category
    
    def _extract_metric(self, text: str, metric_name: str, 
                       config: Dict, period: Optional[str]) -> List[FinancialMetric]:
        """Extract a specific metric using its configuration"""
        metrics = []
        pattern = config['pattern']
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            try:
                # Extract value
                value_str = match.group(1).replace(',', '')
                value = float(value_str)
                
                # Determine unit
                unit = ''
                multiplier = 1
                
                if match.lastindex >= 2 and match.group(2):
                    unit_str = match.group(2).upper()
                    if 'B' in unit_str or 'BILLION' in unit_str:
                        unit = 'B'
                        multiplier = 1e9
                    elif 'M' in unit_str or 'MILLION' in unit_str:
                        unit = 'M'
                        multiplier = 1e6
                    elif 'K' in unit_str:
                        unit = 'K'
                        multiplier = 1e3
                
                # Create metric object
                metric = FinancialMetric(
                    name=metric_name,
                    value=value * multiplier if multiplier > 1 else value,
                    unit=unit,
                    currency='USD',  # Default
                    period=period or self._extract_period_from_context(text, match.start()),
                    confidence=0.9,  # Pattern-based extraction is high confidence
                    raw_text=match.group(0)
                )
                
                metrics.append(metric)
                
            except (ValueError, IndexError) as e:
                continue
        
        return metrics
    
    def _extract_period_from_context(self, text: str, position: int, 
                                    window: int = 100) -> Optional[str]:
        """Extract time period from surrounding context"""
        # Look at context around the metric
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end]
        
        # Look for quarter patterns
        quarter_match = re.search(r'Q([1-4])\s*(\d{4})', context)
        if quarter_match:
            return f"Q{quarter_match.group(1)} {quarter_match.group(2)}"
        
        # Look for fiscal year
        fy_match = re.search(r'(?:FY|fiscal year)\s*(\d{4})', context, re.IGNORECASE)
        if fy_match:
            return f"FY {fy_match.group(1)}"
        
        # Look for year
        year_match = re.search(r'\b(20\d{2})\b', context)
        if year_match:
            return year_match.group(1)
        
        return None
    
    def extract_growth_metrics(self, text: str) -> List[Dict]:
        """
        Extract growth and change metrics
        
        Returns:
            List of growth/change dictionaries
        """
        changes = []
        
        for change_type, pattern in self.change_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                value = float(match.group(1))
                
                # Determine if positive or negative
                direction = 'positive' if change_type == 'growth' else 'negative'
                if change_type in ['yoy', 'qoq']:
                    # Need to check context for direction
                    context = text[max(0, match.start()-50):match.start()]
                    if any(word in context.lower() for word in ['decline', 'decrease', 'fell', 'drop']):
                        direction = 'negative'
                    else:
                        direction = 'positive'
                
                changes.append({
                    'type': change_type,
                    'value': value,
                    'direction': direction,
                    'text': match.group(0),
                    'position': match.start()
                })
        
        return changes
    
    def calculate_ratios(self, metrics: Dict[str, List[FinancialMetric]]) -> Dict[str, float]:
        """
        Calculate financial ratios from extracted metrics
        
        Args:
            metrics: Dictionary of extracted metrics by category
            
        Returns:
            Dictionary of calculated ratios
        """
        ratios = {}
        
        # Flatten metrics for easier access
        all_metrics = {}
        for category_metrics in metrics.values():
            for metric in category_metrics:
                all_metrics[metric.name] = metric.value
        
        # Calculate common ratios
        
        # Return on Equity (ROE) = Net Income / Shareholders Equity
        if 'net_income' in all_metrics and 'shareholders_equity' in all_metrics:
            ratios['roe'] = (all_metrics['net_income'] / all_metrics['shareholders_equity']) * 100
        
        # Debt to Equity = Total Debt / Shareholders Equity
        if 'debt' in all_metrics and 'shareholders_equity' in all_metrics:
            ratios['debt_to_equity'] = all_metrics['debt'] / all_metrics['shareholders_equity']
        
        # Current Ratio = Current Assets / Current Liabilities (if available)
        # Asset Turnover = Revenue / Total Assets
        if 'revenue' in all_metrics and 'total_assets' in all_metrics:
            ratios['asset_turnover'] = all_metrics['revenue'] / all_metrics['total_assets']
        
        # Return on Assets (ROA) = Net Income / Total Assets
        if 'net_income' in all_metrics and 'total_assets' in all_metrics:
            ratios['roa'] = (all_metrics['net_income'] / all_metrics['total_assets']) * 100
        
        return ratios
    
    def extract_metric_by_name(self, text: str, 
                              metric_name: str) -> Optional[FinancialMetric]:
        """Extract a specific metric by name"""
        if metric_name not in self.metric_patterns:
            return None
        
        config = self.metric_patterns[metric_name]
        extracted = self._extract_metric(text, metric_name, config, None)
        
        return extracted[0] if extracted else None
    
    def get_metric_summary(self, text: str) -> Dict:
        """Get summary of all extractable metrics"""
        all_metrics = self.extract_all_metrics(text)
        
        summary = {
            'total_metrics': sum(len(metrics) for metrics in all_metrics.values()),
            'by_category': {
                category: len(metrics) 
                for category, metrics in all_metrics.items() 
                if metrics
            },
            'metrics_found': []
        }
        
        # Add details of found metrics
        for category, metrics in all_metrics.items():
            for metric in metrics:
                summary['metrics_found'].append({
                    'name': metric.name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'category': category
                })
        
        return summary
    
    def format_metric(self, metric: FinancialMetric, 
                     include_period: bool = True) -> str:
        """Format metric for display"""
        formatted = f"{metric.name.replace('_', ' ').title()}: "
        
        # Format value based on unit
        if metric.unit == 'B':
            formatted += f"${metric.value/1e9:.2f}B"
        elif metric.unit == 'M':
            formatted += f"${metric.value/1e6:.2f}M"
        elif metric.unit == 'K':
            formatted += f"${metric.value/1e3:.2f}K"
        else:
            formatted += f"${metric.value:.2f}"
        
        if include_period and metric.period:
            formatted += f" ({metric.period})"
        
        return formatted


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = MetricExtractor()
    
    # Sample text
    sample_text = """
    Apple Inc. reported strong Q3 2024 results. Revenue reached $90.1 billion,
    up 15.2% year-over-year. Net income was $25.5 billion with EPS of $1.85.
    Gross margin improved to 46.3% while operating margin was 30.2%.
    The company ended the quarter with total assets of $350 billion and
    shareholders equity of $65 billion.
    """
    
    print("\nExtracting metrics...")
    all_metrics = extractor.extract_all_metrics(sample_text, period="Q3 2024")
    
    print("\n" + "="*60)
    print("EXTRACTED METRICS")
    print("="*60)
    
    for category, metrics in all_metrics.items():
        if metrics:
            print(f"\n{category.replace('_', ' ').title()}:")
            for metric in metrics:
                print(f"  {extractor.format_metric(metric)}")
    
    # Extract growth metrics
    print("\n" + "="*60)
    print("GROWTH METRICS")
    print("="*60)
    
    changes = extractor.extract_growth_metrics(sample_text)
    for change in changes:
        direction_symbol = "↑" if change['direction'] == 'positive' else "↓"
        print(f"  {direction_symbol} {change['type'].upper()}: {change['value']}%")
    
    # Calculate ratios
    print("\n" + "="*60)
    print("CALCULATED RATIOS")
    print("="*60)
    
    ratios = extractor.calculate_ratios(all_metrics)
    for ratio_name, value in ratios.items():
        print(f"  {ratio_name.replace('_', ' ').upper()}: {value:.2f}")
    
    print("\n✓ Metric Extractor Module Ready!")
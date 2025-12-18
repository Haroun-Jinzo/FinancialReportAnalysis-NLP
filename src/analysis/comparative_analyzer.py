"""
Comparative Analyzer
Compare financial documents and metrics across companies/periods
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction.metric_extractor import MetricExtractor
from models.sentiment_model import FinancialSentiment


@dataclass
class Comparison:
    """Comparison result"""
    entity1: str
    entity2: str
    metric: str
    value1: float
    value2: float
    difference: float
    pct_difference: float
    winner: str
    
    def __str__(self):
        return f"{self.entity1} vs {self.entity2} ({self.metric}): {self.pct_difference:+.1f}%"


class ComparativeAnalyzer:
    """
    Compare financial documents and extract insights
    """
    
    def __init__(self):
        """Initialize comparative analyzer"""
        print("Initializing Comparative Analyzer...")
        self.metric_extractor = MetricExtractor()
        self.sentiment_analyzer = FinancialSentiment()
        print("✓ Comparative Analyzer initialized")
    
    def compare_documents(self, doc1: Dict, doc2: Dict) -> Dict:
        """
        Compare two financial documents
        
        Args:
            doc1: {'name': str, 'text': str, 'period': str}
            doc2: {'name': str, 'text': str, 'period': str}
            
        Returns:
            Comprehensive comparison
        """
        result = {
            'entities': (doc1['name'], doc2['name']),
            'periods': (doc1.get('period'), doc2.get('period')),
            'metrics': {},
            'sentiment': {},
            'summary': {}
        }
        
        # Extract metrics from both
        metrics1 = self.metric_extractor.extract_all_metrics(doc1['text'], doc1.get('period'))
        metrics2 = self.metric_extractor.extract_all_metrics(doc2['text'], doc2.get('period'))
        
        # Compare each metric
        comparisons = self._compare_metrics(doc1['name'], metrics1, doc2['name'], metrics2)
        result['metrics'] = comparisons
        
        # Compare sentiment
        sentiment1 = self.sentiment_analyzer.analyze(doc1['text'])
        sentiment2 = self.sentiment_analyzer.analyze(doc2['text'])
        
        result['sentiment'] = {
            'entity1': sentiment1['label'],
            'entity2': sentiment2['label'],
            'comparison': self._compare_sentiment(sentiment1, sentiment2)
        }
        
        # Generate summary
        result['summary'] = self._generate_comparison_summary(result)
        
        return result
    
    def _compare_metrics(self, name1: str, metrics1: Dict,
                        name2: str, metrics2: Dict) -> List[Comparison]:
        """Compare metrics from two documents"""
        comparisons = []
        
        # Flatten metrics
        flat1 = self._flatten_metrics(metrics1)
        flat2 = self._flatten_metrics(metrics2)
        
        # Find common metrics
        common_metrics = set(flat1.keys()) & set(flat2.keys())
        
        for metric_name in common_metrics:
            value1 = flat1[metric_name]
            value2 = flat2[metric_name]
            
            diff = value2 - value1
            pct_diff = (diff / value1 * 100) if value1 != 0 else 0
            
            winner = name2 if value2 > value1 else name1
            
            comparisons.append(Comparison(
                entity1=name1,
                entity2=name2,
                metric=metric_name,
                value1=value1,
                value2=value2,
                difference=diff,
                pct_difference=pct_diff,
                winner=winner
            ))
        
        return comparisons
    
    def _flatten_metrics(self, metrics: Dict) -> Dict[str, float]:
        """Flatten nested metrics dictionary"""
        flat = {}
        
        for category, metric_list in metrics.items():
            if not metric_list:
                continue
            
            for metric in metric_list:
                key = metric.name
                flat[key] = metric.value
        
        return flat
    
    def _compare_sentiment(self, sent1: Dict, sent2: Dict) -> str:
        """Compare sentiment between documents"""
        label1 = sent1['label']
        label2 = sent2['label']
        
        if label1 == label2:
            return f"Both {label1.lower()}"
        elif label1 == 'POSITIVE' and label2 == 'NEGATIVE':
            return "Entity 1 more positive"
        elif label1 == 'NEGATIVE' and label2 == 'POSITIVE':
            return "Entity 2 more positive"
        else:
            return f"Entity 1 {label1.lower()}, Entity 2 {label2.lower()}"
    
    def _generate_comparison_summary(self, comparison: Dict) -> Dict:
        """Generate summary of comparison"""
        metrics = comparison['metrics']
        
        if not metrics:
            return {'message': 'No comparable metrics found'}
        
        # Count wins
        entity1, entity2 = comparison['entities']
        wins1 = sum(1 for m in metrics if m.winner == entity1)
        wins2 = sum(1 for m in metrics if m.winner == entity2)
        
        # Average difference
        avg_diff = np.mean([abs(m.pct_difference) for m in metrics])
        
        # Determine overall winner
        if wins1 > wins2:
            overall_winner = entity1
        elif wins2 > wins1:
            overall_winner = entity2
        else:
            overall_winner = "Tied"
        
        return {
            'total_comparisons': len(metrics),
            'entity1_wins': wins1,
            'entity2_wins': wins2,
            'overall_winner': overall_winner,
            'avg_difference': avg_diff,
            'sentiment_comparison': comparison['sentiment']['comparison']
        }
    
    def compare_multiple(self, documents: List[Dict],
                        metric_name: str) -> Dict:
        """
        Compare a specific metric across multiple documents
        
        Args:
            documents: List of documents
            metric_name: Metric to compare
            
        Returns:
            Ranking and analysis
        """
        results = []
        
        for doc in documents:
            metrics = self.metric_extractor.extract_all_metrics(doc['text'], doc.get('period'))
            flat_metrics = self._flatten_metrics(metrics)
            
            if metric_name in flat_metrics:
                results.append({
                    'entity': doc['name'],
                    'period': doc.get('period'),
                    'value': flat_metrics[metric_name]
                })
        
        # Sort by value
        results.sort(key=lambda x: x['value'], reverse=True)
        
        # Add rankings
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        # Calculate statistics
        values = [r['value'] for r in results]
        
        return {
            'metric': metric_name,
            'rankings': results,
            'statistics': {
                'highest': max(values) if values else 0,
                'lowest': min(values) if values else 0,
                'average': np.mean(values) if values else 0,
                'median': np.median(values) if values else 0,
                'std_dev': np.std(values) if values else 0
            }
        }
    
    def peer_comparison(self, target_doc: Dict,
                       peer_docs: List[Dict]) -> Dict:
        """
        Compare target company against peers
        
        Args:
            target_doc: Target company document
            peer_docs: List of peer company documents
            
        Returns:
            Peer comparison analysis
        """
        target_name = target_doc['name']
        
        # Combine all documents for comparison
        all_docs = [target_doc] + peer_docs
        
        # Extract common metrics
        all_metrics = {}
        for doc in all_docs:
            metrics = self.metric_extractor.extract_all_metrics(doc['text'], doc.get('period'))
            all_metrics[doc['name']] = self._flatten_metrics(metrics)
        
        # Find common metrics across all documents
        common_metrics = set(all_metrics[target_name].keys())
        for metrics in all_metrics.values():
            common_metrics &= set(metrics.keys())
        
        # Compare on each metric
        comparisons = {}
        for metric in common_metrics:
            values = {name: metrics[metric] for name, metrics in all_metrics.items()}
            
            target_value = values[target_name]
            peer_values = [v for name, v in values.items() if name != target_name]
            
            peer_avg = np.mean(peer_values) if peer_values else 0
            
            comparisons[metric] = {
                'target_value': target_value,
                'peer_average': peer_avg,
                'vs_peers': ((target_value - peer_avg) / peer_avg * 100) if peer_avg != 0 else 0,
                'rank': sorted(values.values(), reverse=True).index(target_value) + 1,
                'percentile': (1 - (sorted(values.values(), reverse=True).index(target_value) / len(values))) * 100
            }
        
        return {
            'target': target_name,
            'peer_count': len(peer_docs),
            'metrics_compared': len(comparisons),
            'comparisons': comparisons,
            'overall_performance': self._calculate_overall_performance(comparisons)
        }
    
    def _calculate_overall_performance(self, comparisons: Dict) -> str:
        """Calculate overall performance vs peers"""
        if not comparisons:
            return "INSUFFICIENT_DATA"
        
        # Count how many metrics are above peer average
        above_peer = sum(1 for c in comparisons.values() if c['vs_peers'] > 0)
        total = len(comparisons)
        
        pct_above = above_peer / total
        
        if pct_above >= 0.7:
            return "OUTPERFORMING"
        elif pct_above >= 0.5:
            return "AT_PAR"
        elif pct_above >= 0.3:
            return "BELOW_PAR"
        else:
            return "UNDERPERFORMING"
    
    def generate_comparison_report(self, comparison: Dict) -> str:
        """Generate human-readable comparison report"""
        report = []
        
        report.append(f"\n{'='*60}")
        report.append(f"COMPARATIVE ANALYSIS")
        report.append(f"{'='*60}")
        
        entity1, entity2 = comparison['entities']
        report.append(f"\nComparing: {entity1} vs {entity2}")
        
        if comparison['periods'][0] and comparison['periods'][1]:
            report.append(f"Periods: {comparison['periods'][0]} vs {comparison['periods'][1]}")
        
        # Summary
        summary = comparison['summary']
        report.append(f"\nOverall Winner: {summary['overall_winner']}")
        report.append(f"Comparisons: {summary['total_comparisons']}")
        report.append(f"  {entity1}: {summary['entity1_wins']} wins")
        report.append(f"  {entity2}: {summary['entity2_wins']} wins")
        report.append(f"Average Difference: {summary['avg_difference']:.1f}%")
        
        # Metrics
        report.append(f"\nMetric Comparisons:")
        for comp in comparison['metrics'][:10]:  # Show top 10
            symbol = ">" if comp.value1 > comp.value2 else "<"
            report.append(f"  {comp.metric}:")
            report.append(f"    {entity1}: {comp.value1:.2f}")
            report.append(f"    {entity2}: {comp.value2:.2f}")
            report.append(f"    Difference: {comp.pct_difference:+.1f}%")
        
        # Sentiment
        report.append(f"\nSentiment:")
        report.append(f"  {entity1}: {comparison['sentiment']['entity1']}")
        report.append(f"  {entity2}: {comparison['sentiment']['entity2']}")
        report.append(f"  Analysis: {comparison['sentiment']['comparison']}")
        
        report.append(f"\n{'='*60}")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    analyzer = ComparativeAnalyzer()
    
    # Sample documents
    doc1 = {
        'name': 'Apple',
        'period': 'Q3 2024',
        'text': '''
        Apple reported revenue of $90.1 billion with net income of $25.5 billion.
        Gross margin was 46.3% and operating margin reached 30.2%.
        The strong performance exceeded analyst expectations.
        '''
    }
    
    doc2 = {
        'name': 'Microsoft',
        'period': 'Q3 2024',
        'text': '''
        Microsoft generated revenue of $62.0 billion with net income of $22.3 billion.
        Gross margin stood at 69.5% and operating margin was 42.8%.
        Cloud revenue grew significantly year-over-year.
        '''
    }
    
    print("\nComparing documents...")
    comparison = analyzer.compare_documents(doc1, doc2)
    
    # Generate report
    report = analyzer.generate_comparison_report(comparison)
    print(report)
    
    print("\n✓ Comparative Analyzer Module Ready!")
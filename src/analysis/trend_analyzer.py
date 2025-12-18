"""
Trend Analyzer
Analyzes financial trends over time
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction.metric_extractor import MetricExtractor, FinancialMetric


@dataclass
class TrendPoint:
    """Data point in a trend"""
    period: str
    value: float
    metric: str
    change_pct: Optional[float] = None
    
    def __str__(self):
        change_str = f" ({self.change_pct:+.1f}%)" if self.change_pct else ""
        return f"{self.period}: {self.value}{change_str}"


@dataclass
class TrendAnalysis:
    """Complete trend analysis"""
    metric: str
    direction: str  # INCREASING, DECREASING, STABLE, VOLATILE
    strength: float  # 0-1
    data_points: List[TrendPoint]
    average: float
    volatility: float
    forecast: Optional[float] = None


class TrendAnalyzer:
    """
    Analyze financial trends across multiple periods
    """
    
    def __init__(self):
        """Initialize trend analyzer"""
        print("Initializing Trend Analyzer...")
        self.metric_extractor = MetricExtractor()
        print("✓ Trend Analyzer initialized")
    
    def analyze_trend(self, documents: List[Dict[str, str]], 
                     metric_name: str) -> TrendAnalysis:
        """
        Analyze trend for a specific metric across documents
        
        Args:
            documents: List of dicts with 'period' and 'text'
            metric_name: Name of metric to analyze (e.g., 'revenue')
            
        Returns:
            TrendAnalysis object
        """
        # Extract metric from each document
        data_points = []
        
        for doc in documents:
            period = doc['period']
            text = doc['text']
            
            # Extract metric
            metric = self.metric_extractor.extract_metric_by_name(text, metric_name)
            
            if metric:
                data_points.append(TrendPoint(
                    period=period,
                    value=metric.value,
                    metric=metric_name
                ))
        
        if not data_points:
            return self._empty_analysis(metric_name)
        
        # Calculate changes
        for i in range(1, len(data_points)):
            prev_value = data_points[i-1].value
            curr_value = data_points[i].value
            change_pct = ((curr_value - prev_value) / prev_value) * 100
            data_points[i].change_pct = change_pct
        
        # Analyze trend
        values = [p.value for p in data_points]
        changes = [p.change_pct for p in data_points if p.change_pct is not None]
        
        # Determine direction
        if changes:
            avg_change = np.mean(changes)
            if avg_change > 2:
                direction = "INCREASING"
            elif avg_change < -2:
                direction = "DECREASING"
            elif np.std(changes) > 5:
                direction = "VOLATILE"
            else:
                direction = "STABLE"
        else:
            direction = "INSUFFICIENT_DATA"
        
        # Calculate strength (consistency of direction)
        if changes:
            positive_changes = sum(1 for c in changes if c > 0)
            strength = abs((positive_changes / len(changes)) - 0.5) * 2
        else:
            strength = 0.0
        
        # Calculate statistics
        average = np.mean(values)
        volatility = np.std(values) / average if average > 0 else 0
        
        # Simple forecast (linear projection)
        forecast = None
        if len(values) >= 2:
            # Simple linear regression
            x = np.arange(len(values))
            y = np.array(values)
            slope = np.polyfit(x, y, 1)[0]
            forecast = values[-1] + slope
        
        return TrendAnalysis(
            metric=metric_name,
            direction=direction,
            strength=strength,
            data_points=data_points,
            average=average,
            volatility=volatility,
            forecast=forecast
        )
    
    def _empty_analysis(self, metric_name: str) -> TrendAnalysis:
        """Return empty analysis when no data found"""
        return TrendAnalysis(
            metric=metric_name,
            direction="NO_DATA",
            strength=0.0,
            data_points=[],
            average=0.0,
            volatility=0.0
        )
    
    def analyze_multiple_metrics(self, documents: List[Dict[str, str]],
                                 metrics: List[str]) -> Dict[str, TrendAnalysis]:
        """
        Analyze trends for multiple metrics
        
        Args:
            documents: List of documents with periods
            metrics: List of metric names to analyze
            
        Returns:
            Dictionary mapping metric names to TrendAnalysis
        """
        results = {}
        
        for metric in metrics:
            analysis = self.analyze_trend(documents, metric)
            results[metric] = analysis
        
        return results
    
    def detect_anomalies(self, trend: TrendAnalysis,
                        threshold: float = 2.0) -> List[Dict]:
        """
        Detect anomalies in trend data
        
        Args:
            trend: TrendAnalysis object
            threshold: Z-score threshold for anomaly
            
        Returns:
            List of anomaly dictionaries
        """
        if len(trend.data_points) < 3:
            return []
        
        values = [p.value for p in trend.data_points]
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        anomalies = []
        for point in trend.data_points:
            z_score = abs((point.value - mean) / std)
            
            if z_score > threshold:
                anomalies.append({
                    'period': point.period,
                    'value': point.value,
                    'z_score': z_score,
                    'type': 'HIGH' if point.value > mean else 'LOW',
                    'severity': 'HIGH' if z_score > 3 else 'MEDIUM'
                })
        
        return anomalies
    
    def calculate_growth_rate(self, trend: TrendAnalysis) -> Dict:
        """
        Calculate various growth rates
        
        Returns:
            Dictionary with growth metrics
        """
        if len(trend.data_points) < 2:
            return {'error': 'Insufficient data'}
        
        values = [p.value for p in trend.data_points]
        
        # Period-over-period growth
        pop_growth = []
        for i in range(1, len(values)):
            growth = ((values[i] - values[i-1]) / values[i-1]) * 100
            pop_growth.append(growth)
        
        # Compound Annual Growth Rate (CAGR)
        n = len(values) - 1
        cagr = ((values[-1] / values[0]) ** (1/n) - 1) * 100 if n > 0 else 0
        
        return {
            'average_growth': np.mean(pop_growth) if pop_growth else 0,
            'cagr': cagr,
            'min_growth': min(pop_growth) if pop_growth else 0,
            'max_growth': max(pop_growth) if pop_growth else 0,
            'growth_volatility': np.std(pop_growth) if len(pop_growth) > 1 else 0
        }
    
    def identify_turning_points(self, trend: TrendAnalysis) -> List[Dict]:
        """
        Identify trend reversals and turning points
        
        Returns:
            List of turning points
        """
        if len(trend.data_points) < 3:
            return []
        
        turning_points = []
        changes = [p.change_pct for p in trend.data_points if p.change_pct is not None]
        
        for i in range(1, len(changes)):
            prev_change = changes[i-1]
            curr_change = changes[i]
            
            # Detect sign change (reversal)
            if prev_change * curr_change < 0:
                turning_points.append({
                    'period': trend.data_points[i+1].period,
                    'type': 'REVERSAL',
                    'from_direction': 'UP' if prev_change > 0 else 'DOWN',
                    'to_direction': 'DOWN' if prev_change > 0 else 'UP',
                    'magnitude': abs(curr_change - prev_change)
                })
            
            # Detect acceleration
            elif abs(curr_change) > abs(prev_change) * 1.5:
                turning_points.append({
                    'period': trend.data_points[i+1].period,
                    'type': 'ACCELERATION',
                    'direction': 'UP' if curr_change > 0 else 'DOWN',
                    'magnitude': abs(curr_change - prev_change)
                })
        
        return turning_points
    
    def compare_to_benchmark(self, trend: TrendAnalysis,
                           benchmark_values: List[float]) -> Dict:
        """
        Compare trend to benchmark
        
        Args:
            trend: TrendAnalysis to compare
            benchmark_values: List of benchmark values for same periods
            
        Returns:
            Comparison metrics
        """
        if len(trend.data_points) != len(benchmark_values):
            return {'error': 'Mismatched data lengths'}
        
        actual_values = [p.value for p in trend.data_points]
        
        # Calculate differences
        differences = [a - b for a, b in zip(actual_values, benchmark_values)]
        pct_differences = [(a - b) / b * 100 for a, b in zip(actual_values, benchmark_values) if b != 0]
        
        # Performance metrics
        outperformance_periods = sum(1 for d in differences if d > 0)
        
        return {
            'avg_difference': np.mean(differences),
            'avg_pct_difference': np.mean(pct_differences) if pct_differences else 0,
            'outperformance_rate': outperformance_periods / len(differences),
            'max_outperformance': max(pct_differences) if pct_differences else 0,
            'max_underperformance': min(pct_differences) if pct_differences else 0,
            'periods_above_benchmark': outperformance_periods,
            'periods_below_benchmark': len(differences) - outperformance_periods
        }
    
    def generate_trend_report(self, trend: TrendAnalysis) -> str:
        """Generate human-readable trend report"""
        report = []
        
        report.append(f"\n{'='*60}")
        report.append(f"TREND ANALYSIS: {trend.metric.upper()}")
        report.append(f"{'='*60}")
        
        # Direction and strength
        report.append(f"\nDirection: {trend.direction}")
        report.append(f"Strength: {trend.strength:.1%}")
        report.append(f"Average Value: {trend.average:.2f}")
        report.append(f"Volatility: {trend.volatility:.1%}")
        
        if trend.forecast:
            report.append(f"Next Period Forecast: {trend.forecast:.2f}")
        
        # Data points
        report.append(f"\nHistorical Data:")
        for point in trend.data_points:
            report.append(f"  {point}")
        
        # Growth rates
        if len(trend.data_points) >= 2:
            growth = self.calculate_growth_rate(trend)
            report.append(f"\nGrowth Metrics:")
            report.append(f"  Average Growth: {growth['average_growth']:+.1f}%")
            report.append(f"  CAGR: {growth['cagr']:+.1f}%")
            report.append(f"  Range: {growth['min_growth']:+.1f}% to {growth['max_growth']:+.1f}%")
        
        # Anomalies
        anomalies = self.detect_anomalies(trend)
        if anomalies:
            report.append(f"\nAnomalies Detected: {len(anomalies)}")
            for anomaly in anomalies:
                report.append(f"  {anomaly['period']}: {anomaly['type']} ({anomaly['severity']})")
        
        # Turning points
        turning_points = self.identify_turning_points(trend)
        if turning_points:
            report.append(f"\nTurning Points: {len(turning_points)}")
            for tp in turning_points[:3]:  # Show first 3
                report.append(f"  {tp['period']}: {tp['type']}")
        
        report.append(f"\n{'='*60}")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    analyzer = TrendAnalyzer()
    
    # Sample documents (simulating quarterly reports)
    documents = [
        {
            'period': 'Q1 2024',
            'text': 'Revenue reached $85.3 billion with net income of $23.2 billion.'
        },
        {
            'period': 'Q2 2024',
            'text': 'Revenue was $87.5 billion with net income of $24.1 billion.'
        },
        {
            'period': 'Q3 2024',
            'text': 'Revenue totaled $90.1 billion with net income of $25.5 billion.'
        },
        {
            'period': 'Q4 2024',
            'text': 'Revenue of $92.8 billion with net income of $26.8 billion.'
        }
    ]
    
    print("\nAnalyzing revenue trend...")
    trend = analyzer.analyze_trend(documents, 'revenue')
    
    # Generate report
    report = analyzer.generate_trend_report(trend)
    print(report)
    
    # Growth rates
    growth = analyzer.calculate_growth_rate(trend)
    print(f"\nCAGR: {growth['cagr']:.2f}%")
    
    print("\n✓ Trend Analyzer Module Ready!")
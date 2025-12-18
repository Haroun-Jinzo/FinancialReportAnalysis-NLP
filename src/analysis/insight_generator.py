"""
Insight Generator
Generates actionable insights from financial analysis
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.trend_analyzer import TrendAnalyzer, TrendAnalysis
from analysis.risk_analyzer import RiskAnalyzer
from extraction.metric_extractor import MetricExtractor
from models.sentiment_model import FinancialSentiment


@dataclass
class Insight:
    """Actionable insight"""
    type: str  # OPPORTUNITY, WARNING, TREND, ANOMALY
    category: str
    title: str
    description: str
    impact: str  # HIGH, MEDIUM, LOW
    confidence: float
    action_items: List[str]
    supporting_data: Dict
    
    def __str__(self):
        return f"[{self.type}] {self.title} (Impact: {self.impact})"


class InsightGenerator:
    """
    Generate actionable insights from financial data
    """
    
    def __init__(self):
        """Initialize insight generator"""
        print("Initializing Insight Generator...")
        
        self.trend_analyzer = TrendAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.metric_extractor = MetricExtractor()
        self.sentiment_analyzer = FinancialSentiment()
        
        print("✓ Insight Generator initialized")
    
    def generate_insights(self, documents: List[Dict],
                         analysis_results: Optional[Dict] = None) -> List[Insight]:
        """
        Generate insights from documents and analysis
        
        Args:
            documents: List of financial documents
            analysis_results: Pre-computed analysis results (optional)
            
        Returns:
            List of Insight objects
        """
        insights = []
        
        # If no analysis provided, compute it
        if not analysis_results:
            analysis_results = self._analyze_documents(documents)
        
        # Generate different types of insights
        insights.extend(self._generate_trend_insights(analysis_results.get('trends', {})))
        insights.extend(self._generate_risk_insights(analysis_results.get('risks', [])))
        insights.extend(self._generate_performance_insights(analysis_results.get('metrics', {})))
        insights.extend(self._generate_sentiment_insights(analysis_results.get('sentiment', {})))
        insights.extend(self._generate_anomaly_insights(analysis_results.get('anomalies', [])))
        
        # Sort by impact and confidence
        impact_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        insights.sort(key=lambda i: (impact_order.get(i.impact, 3), -i.confidence))
        
        return insights
    
    def _analyze_documents(self, documents: List[Dict]) -> Dict:
        """Perform comprehensive analysis on documents"""
        results = {
            'trends': {},
            'risks': [],
            'metrics': {},
            'sentiment': {},
            'anomalies': []
        }
        
        # Analyze trends for key metrics
        if len(documents) >= 2:
            for metric in ['revenue', 'net_income', 'eps']:
                trend = self.trend_analyzer.analyze_trend(documents, metric)
                if trend.data_points:
                    results['trends'][metric] = trend
                    
                    # Check for anomalies
                    anomalies = self.trend_analyzer.detect_anomalies(trend)
                    results['anomalies'].extend(anomalies)
        
        # Analyze risks from latest document
        if documents:
            latest_text = documents[-1]['text']
            risks = self.risk_analyzer.analyze_risks(latest_text)
            results['risks'] = risks
            
            # Sentiment
            sentiment = self.sentiment_analyzer.analyze(latest_text)
            results['sentiment'] = sentiment
        
        return results
    
    def _generate_trend_insights(self, trends: Dict[str, TrendAnalysis]) -> List[Insight]:
        """Generate insights from trend analysis"""
        insights = []
        
        for metric, trend in trends.items():
            if not trend.data_points or len(trend.data_points) < 2:
                continue
            
            # Strong growth insight
            if trend.direction == "INCREASING" and trend.strength > 0.7:
                growth = self.trend_analyzer.calculate_growth_rate(trend)
                
                insights.append(Insight(
                    type="OPPORTUNITY",
                    category="Growth",
                    title=f"Strong {metric.replace('_', ' ').title()} Growth",
                    description=f"{metric.replace('_', ' ').title()} showing consistent growth of {growth['average_growth']:.1f}% with {trend.forecast:.2f} forecast.",
                    impact="HIGH",
                    confidence=trend.strength,
                    action_items=[
                        "Continue current growth strategies",
                        "Consider increasing investment",
                        "Explore scaling opportunities"
                    ],
                    supporting_data={
                        'trend_direction': trend.direction,
                        'growth_rate': growth['average_growth'],
                        'forecast': trend.forecast
                    }
                ))
            
            # Decline warning
            elif trend.direction == "DECREASING" and trend.strength > 0.6:
                growth = self.trend_analyzer.calculate_growth_rate(trend)
                
                insights.append(Insight(
                    type="WARNING",
                    category="Decline",
                    title=f"Declining {metric.replace('_', ' ').title()}",
                    description=f"{metric.replace('_', ' ').title()} declining at {growth['average_growth']:.1f}% average rate. Immediate action required.",
                    impact="HIGH",
                    confidence=trend.strength,
                    action_items=[
                        "Investigate root causes",
                        "Implement corrective measures",
                        "Review operational efficiency"
                    ],
                    supporting_data={
                        'trend_direction': trend.direction,
                        'decline_rate': growth['average_growth'],
                        'forecast': trend.forecast
                    }
                ))
            
            # Volatility warning
            elif trend.direction == "VOLATILE" and trend.volatility > 0.15:
                insights.append(Insight(
                    type="WARNING",
                    category="Volatility",
                    title=f"High {metric.replace('_', ' ').title()} Volatility",
                    description=f"{metric.replace('_', ' ').title()} showing high volatility ({trend.volatility:.1%}). Consider stabilization measures.",
                    impact="MEDIUM",
                    confidence=0.8,
                    action_items=[
                        "Analyze sources of volatility",
                        "Implement smoothing strategies",
                        "Diversify revenue streams"
                    ],
                    supporting_data={
                        'volatility': trend.volatility,
                        'average': trend.average
                    }
                ))
        
        return insights
    
    def _generate_risk_insights(self, risks: List) -> List[Insight]:
        """Generate insights from risk analysis"""
        insights = []
        
        # High severity risks
        high_risks = [r for r in risks if r.severity in ['HIGH', 'CRITICAL']]
        
        if high_risks:
            for risk in high_risks[:3]:  # Top 3
                insights.append(Insight(
                    type="WARNING",
                    category=risk.category.replace('_', ' ').title(),
                    title=f"{risk.category.replace('_', ' ').title()} Risk Identified",
                    description=risk.description,
                    impact=risk.severity,
                    confidence=risk.confidence,
                    action_items=[
                        "Assess risk mitigation options",
                        "Develop contingency plans",
                        "Monitor risk indicators"
                    ],
                    supporting_data={
                        'indicators': risk.indicators,
                        'context': risk.context
                    }
                ))
        
        return insights
    
    def _generate_performance_insights(self, metrics: Dict) -> List[Insight]:
        """Generate insights from performance metrics"""
        insights = []
        
        # This would compare metrics against benchmarks/targets
        # Simplified for now
        
        return insights
    
    def _generate_sentiment_insights(self, sentiment: Dict) -> List[Insight]:
        """Generate insights from sentiment analysis"""
        insights = []
        
        if not sentiment:
            return insights
        
        # Negative sentiment warning
        if sentiment.get('label') == 'NEGATIVE' and sentiment.get('score', 0) > 0.7:
            insights.append(Insight(
                type="WARNING",
                category="Sentiment",
                title="Negative Sentiment Detected",
                description="Document sentiment is predominantly negative, indicating potential challenges or concerns.",
                impact="MEDIUM",
                confidence=sentiment['score'],
                action_items=[
                    "Review underlying issues",
                    "Prepare communication strategy",
                    "Monitor stakeholder reactions"
                ],
                supporting_data={
                    'sentiment': sentiment['label'],
                    'score': sentiment['score']
                }
            ))
        
        # Very positive sentiment
        elif sentiment.get('label') == 'POSITIVE' and sentiment.get('score', 0) > 0.8:
            insights.append(Insight(
                type="OPPORTUNITY",
                category="Sentiment",
                title="Strong Positive Sentiment",
                description="Document reflects strong positive sentiment, indicating confidence and good performance.",
                impact="MEDIUM",
                confidence=sentiment['score'],
                action_items=[
                    "Leverage positive momentum",
                    "Communicate success stories",
                    "Reinforce successful strategies"
                ],
                supporting_data={
                    'sentiment': sentiment['label'],
                    'score': sentiment['score']
                }
            ))
        
        return insights
    
    def _generate_anomaly_insights(self, anomalies: List) -> List[Insight]:
        """Generate insights from detected anomalies"""
        insights = []
        
        for anomaly in anomalies:
            if anomaly.get('severity') == 'HIGH':
                insights.append(Insight(
                    type="ANOMALY",
                    category="Data Anomaly",
                    title=f"Unusual Value Detected in {anomaly.get('period', 'Period')}",
                    description=f"{anomaly.get('type', 'Anomaly')} value detected with z-score of {anomaly.get('z_score', 0):.2f}.",
                    impact="MEDIUM",
                    confidence=0.85,
                    action_items=[
                        "Verify data accuracy",
                        "Investigate underlying causes",
                        "Determine if one-time or recurring"
                    ],
                    supporting_data=anomaly
                ))
        
        return insights
    
    def generate_executive_summary(self, insights: List[Insight]) -> str:
        """Generate executive summary from insights"""
        if not insights:
            return "No significant insights identified."
        
        summary = []
        
        # Key highlights
        opportunities = [i for i in insights if i.type == "OPPORTUNITY"]
        warnings = [i for i in insights if i.type == "WARNING"]
        
        summary.append("EXECUTIVE SUMMARY")
        summary.append("="*60)
        
        if opportunities:
            summary.append(f"\n✓ Opportunities ({len(opportunities)}):")
            for opp in opportunities[:3]:
                summary.append(f"  • {opp.title}")
        
        if warnings:
            summary.append(f"\n⚠ Warnings ({len(warnings)}):")
            for warn in warnings[:3]:
                summary.append(f"  • {warn.title}")
        
        return "\n".join(summary)
    
    def generate_insight_report(self, insights: List[Insight]) -> str:
        """Generate detailed insight report"""
        report = []
        
        report.append(f"\n{'='*60}")
        report.append(f"INSIGHT REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"{'='*60}")
        
        report.append(f"\nTotal Insights: {len(insights)}")
        
        # Group by type
        by_type = {}
        for insight in insights:
            if insight.type not in by_type:
                by_type[insight.type] = []
            by_type[insight.type].append(insight)
        
        for insight_type, items in by_type.items():
            report.append(f"\n{insight_type} ({len(items)}):")
            report.append("-" * 60)
            
            for insight in items:
                report.append(f"\n• {insight.title}")
                report.append(f"  Impact: {insight.impact} | Confidence: {insight.confidence:.0%}")
                report.append(f"  {insight.description}")
                
                if insight.action_items:
                    report.append(f"  Action Items:")
                    for action in insight.action_items:
                        report.append(f"    - {action}")
        
        report.append(f"\n{'='*60}")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    generator = InsightGenerator()
    
    # Sample documents
    documents = [
        {
            'period': 'Q1 2024',
            'text': 'Revenue was $85B with positive growth trends.'
        },
        {
            'period': 'Q2 2024',
            'text': 'Revenue reached $88B, continuing upward trajectory.'
        },
        {
            'period': 'Q3 2024',
            'text': 'Revenue of $90B. Strong performance across all segments.'
        }
    ]
    
    print("\nGenerating insights...")
    insights = generator.generate_insights(documents)
    
    # Generate report
    report = generator.generate_insight_report(insights)
    print(report)
    
    print("\n✓ Insight Generator Module Ready!")
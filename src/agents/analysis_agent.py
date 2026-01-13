import sys
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import analysis modules
from analysis.trend_analyzer import TrendAnalyzer
from analysis.comparative_analyzer import ComparativeAnalyzer
from analysis.risk_analyzer import RiskAnalyzer
from analysis.insight_generator import InsightGenerator
from extraction.metric_extractor import MetricExtractor
from models.sentiment_model import FinancialSentiment

logger = logging.getLogger(__name__)


class AnalysisAgent:
    
    def __init__(self):
        logger.info("📊 Starting Analysis Agent...")
        
        # Initialize all analyzers
        self.trend_analyzer = TrendAnalyzer()
        self.comparative_analyzer = ComparativeAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.insight_generator = InsightGenerator()
        self.metric_extractor = MetricExtractor()
        self.sentiment_analyzer = FinancialSentiment()
        
        logger.info("✓ Analysis Agent ready")
    
    def analyze_trends(self, documents: list, metrics: list = None) -> dict:
        logger.info(f"📈 Analyzing trends ({len(documents)} documents)")
        
        # Use default metrics if none specified
        if not metrics:
            metrics = ['revenue', 'net_income', 'eps']
        
        results = {}
        
        # Analyze each metric
        for metric in metrics:
            logger.info(f"  Analyzing {metric}...")
            
            # Get trend data from analyzer
            trend = self.trend_analyzer.analyze_trend(documents, metric)
            
            # Only include if we found data
            if trend.data_points:
                results[metric] = {
                    'direction': trend.direction,     # INCREASING, DECREASING, STABLE, VOLATILE
                    'strength': trend.strength,       # 0-1, higher = more consistent
                    'average': trend.average,
                    'volatility': trend.volatility,
                    'forecast': trend.forecast,       # Prediction for next period
                    'data': [
                        {
                            'period': point.period,
                            'value': point.value,
                            'change_pct': point.change_pct  # % change from previous
                        }
                        for point in trend.data_points
                    ]
                }
                
                logger.info(f"    {metric}: {trend.direction} (strength: {trend.strength:.2f})")
        
        # Create summary
        summary = self._summarize_trends(results)
        
        logger.info(f"✓ Analyzed {len(results)} metrics")
        logger.info(f"  Improving: {len(summary['improving'])}")
        logger.info(f"  Declining: {len(summary['declining'])}")
        
        return {
            'trends': results,
            'summary': summary
        }
    
    def analyze_risks(self, text: str, detailed: bool = True) -> dict:
        logger.info("⚠️  Analyzing risks")
        
        # Find all risks in the text
        risks = self.risk_analyzer.analyze_risks(text)
        
        # Calculate overall risk score
        risk_score = self.risk_analyzer.calculate_risk_score(risks)
        
        # Build results
        results = {
            'risk_level': risk_score['risk_level'],      # LOW, MEDIUM, HIGH, CRITICAL
            'risk_score': risk_score['total_score'],     # 0-100 (higher = more risky)
            'total_risks': len(risks),
            'by_severity': risk_score.get('by_severity', {}),    # Count by severity
            'by_category': risk_score.get('by_category', {})    # Count by category
        }
        
        # Add detailed risk information if requested
        if detailed:
            results['risks'] = [
                {
                    'category': risk.category,
                    'description': risk.description[:100] + '...' if len(risk.description) > 100 else risk.description,
                    'severity': risk.severity,
                    'confidence': risk.confidence
                }
                for risk in risks[:10]  # Top 10 most important
            ]
        
        logger.info(f"✓ Risk Level: {results['risk_level']} (Score: {results['risk_score']:.1f})")
        logger.info(f"  Total risks: {results['total_risks']}")
        
        return results
    
    def analyze_sentiment(self, text: str) -> dict:
        logger.info("😊 Analyzing sentiment")
        
        # Get overall sentiment from analyzer
        sentiment = self.sentiment_analyzer.get_overall_sentiment(text)
        
        result = {
            'sentiment': sentiment['overall_sentiment'],  # POSITIVE, NEGATIVE, NEUTRAL
            'confidence': sentiment['confidence'],
            'breakdown': sentiment['sentence_breakdown']
        }
        
        logger.info(f"✓ Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
        
        return result
    
    def compare_documents(self, doc1: dict, doc2: dict, analysis_type: str = "comprehensive") -> dict:
        
        logger.info(f"⚖️  Comparing: {doc1['name']} vs {doc2['name']}")
        
        results = {}
        
        # Compare metrics (revenue, profit, etc)
        if analysis_type in ['comprehensive', 'metrics']:
            logger.info("  Comparing metrics...")
            comparison = self.comparative_analyzer.compare_documents(doc1, doc2)
            results['metrics'] = comparison
        
        # Compare sentiment (positive/negative)
        if analysis_type in ['comprehensive', 'sentiment']:
            logger.info("  Comparing sentiment...")
            sent1 = self.sentiment_analyzer.get_overall_sentiment(doc1['text'])
            sent2 = self.sentiment_analyzer.get_overall_sentiment(doc2['text'])
            
            results['sentiment'] = {
                'entity1': sent1['overall_sentiment'],
                'entity2': sent2['overall_sentiment']
            }
        
        logger.info("✓ Comparison complete")
        return results
    
    def comprehensive_analysis(self, documents: list, include_risks: bool = True, include_insights: bool = True) -> dict:
        logger.info(f"🎯 Comprehensive analysis ({len(documents)} documents)")
        
        results = {}
        
        # 1. Trend Analysis (if we have multiple documents)
        if len(documents) >= 2:
            logger.info("  → Step 1: Analyzing trends")
            results['trends'] = self.analyze_trends(documents)
        else:
            logger.info("  → Step 1: Skipped (need 2+ documents for trends)")
        
        # 2. Sentiment Analysis (for each document)
        logger.info("  → Step 2: Analyzing sentiment")
        results['sentiment'] = {
            'by_period': []
        }
        
        for doc in documents:
            sent = self.sentiment_analyzer.get_overall_sentiment(doc['text'])
            results['sentiment']['by_period'].append({
                'period': doc.get('period', 'Unknown'),
                'sentiment': sent['overall_sentiment'],
                'confidence': sent['confidence']
            })
        
        # 3. Risk Analysis (on the latest document)
        if include_risks and documents:
            logger.info("  → Step 3: Analyzing risks")
            latest_doc = documents[-1]
            results['risks'] = self.analyze_risks(latest_doc['text'])
        else:
            logger.info("  → Step 3: Skipped (risks not requested)")
        
        # 4. Generate Insights (recommendations)
        if include_insights:
            logger.info("  → Step 4: Generating insights")
            insights = self.insight_generator.generate_insights(documents)
            
            results['insights'] = {
                'total': len(insights),
                'by_type': {},
                'top_5': []
            }
            
            # Organize insights by type
            for insight in insights:
                if insight.type not in results['insights']['by_type']:
                    results['insights']['by_type'][insight.type] = 0
                results['insights']['by_type'][insight.type] += 1
            
            # Get top 5 insights
            results['insights']['top_5'] = [
                {
                    'type': i.type,
                    'title': i.title,
                    'description': i.description,
                    'impact': i.impact,
                    'action_items': i.action_items
                }
                for i in insights[:5]
            ]
        else:
            logger.info("  → Step 4: Skipped (insights not requested)")
        
        logger.info("✓ Comprehensive analysis complete")
        
        return results
    
    def generate_insights(self, documents: list) -> dict:
        logger.info("💡 Generating insights")
        
        # Generate insights using the insight generator
        insights = self.insight_generator.generate_insights(documents)
        
        result = {
            'total_insights': len(insights),
            'by_type': {},
            'top_insights': []
        }
        
        # Count by type
        for insight in insights:
            if insight.type not in result['by_type']:
                result['by_type'][insight.type] = 0
            result['by_type'][insight.type] += 1
        
        # Get top 5 insights
        result['top_insights'] = [
            {
                'type': i.type,
                'title': i.title,
                'description': i.description,
                'impact': i.impact,
                'action_items': i.action_items
            }
            for i in insights[:5]
        ]
        
        logger.info(f"✓ Generated {len(insights)} insights")
        
        return result
    
    def _summarize_trends(self, trends: dict) -> dict:
        summary = {
            'improving': [],
            'declining': [],
            'stable': []
        }
        
        for metric, data in trends.items():
            direction = data['direction']
            
            if direction == 'INCREASING':
                summary['improving'].append(metric)
            elif direction == 'DECREASING':
                summary['declining'].append(metric)
            elif direction == 'STABLE':
                summary['stable'].append(metric)
        
        return summary


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Setup logging to see what's happening
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    print("\n" + "="*70)
    print("ANALYSIS AGENT - DEMO")
    print("="*70)
    
    # Create the agent
    agent = AnalysisAgent()
    
    # Sample documents (simulating quarterly reports)
    sample_docs = [
        {
            'period': 'Q1 2024',
            'text': 'Revenue reached $85 billion with strong growth. Net income $23 billion. Markets are stable.'
        },
        {
            'period': 'Q2 2024',
            'text': 'Revenue $88 billion. Net income $24 billion. Continued positive performance despite challenges.'
        },
        {
            'period': 'Q3 2024',
            'text': 'Revenue $90 billion. Net income $25.5 billion. Excellent results. Some supply chain concerns noted.'
        }
    ]
    
    # Example 1: Comprehensive Analysis
    print("\n📊 Running comprehensive analysis...")
    results = agent.comprehensive_analysis(sample_docs)
    
    print("\n✓ Analysis Complete!\n")
    
    # Show trends
    if 'trends' in results:
        print("📈 TRENDS:")
        summary = results['trends']['summary']
        if summary['improving']:
            print(f"  Improving: {', '.join(summary['improving'])}")
        if summary['declining']:
            print(f"  Declining: {', '.join(summary['declining'])}")
    
    # Show sentiment
    if 'sentiment' in results:
        print("\n😊 SENTIMENT:")
        for period_data in results['sentiment']['by_period']:
            print(f"  {period_data['period']}: {period_data['sentiment']}")
    
    # Show risks
    if 'risks' in results:
        print("\n⚠️  RISKS:")
        print(f"  Level: {results['risks']['risk_level']}")
        print(f"  Total: {results['risks']['total_risks']}")
    
    # Show insights
    if 'insights' in results:
        print("\n💡 INSIGHTS:")
        print(f"  Total: {results['insights']['total']}")
        if results['insights']['top_5']:
            print("\n  Top Insight:")
            top = results['insights']['top_5'][0]
            print(f"    {top['title']} ({top['impact']} impact)")
    
    print("\n" + "="*70)
    print("✓ Demo complete! The agent is ready to use.")
    print("="*70 + "\n")
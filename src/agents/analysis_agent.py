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
        """
        Analyze how metrics change over time
        
        Use this when you have multiple documents from different periods
        and want to see if things are getting better or worse.
        
        Args:
            documents: List of documents, each with 'period' and 'text'
                      Example: [
                          {'period': 'Q1 2024', 'text': 'Revenue $85B...'},
                          {'period': 'Q2 2024', 'text': 'Revenue $88B...'},
                          {'period': 'Q3 2024', 'text': 'Revenue $90B...'}
                      ]
            metrics: Which metrics to analyze (optional)
                    Example: ['revenue', 'net_income', 'eps']
                    Default: ['revenue', 'net_income', 'eps']
        
        Returns:
            Dictionary with trend information:
            {
                'trends': {
                    'revenue': {
                        'direction': 'INCREASING',  # INCREASING, DECREASING, STABLE
                        'strength': 0.95,           # 0-1 (how consistent)
                        'average': 87666666666.67,
                        'forecast': 92000000000,    # Next period prediction
                        'data': [...]               # Historical data points
                    }
                },
                'summary': {
                    'improving': ['revenue'],
                    'declining': [],
                    'stable': []
                }
            }
        
        Example:
            docs = [
                {'period': 'Q1', 'text': 'Revenue $85B'},
                {'period': 'Q2', 'text': 'Revenue $88B'},
                {'period': 'Q3', 'text': 'Revenue $90B'}
            ]
            
            result = agent.analyze_trends(docs, ['revenue'])
            
            if result['trends']['revenue']['direction'] == 'INCREASING':
                print("📈 Revenue is going up!")
        """
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
        """
        Identify financial risks in the document
        
        Use this to find out what could go wrong based on what's written
        in the document. It looks for risk keywords and patterns.
        
        Args:
            text: The document text to analyze
            detailed: If True, includes full risk descriptions
                     If False, just shows counts and levels
        
        Returns:
            Dictionary with risk information:
            {
                'risk_level': 'HIGH',           # LOW, MEDIUM, HIGH, CRITICAL
                'risk_score': 67.5,             # 0-100
                'total_risks': 5,
                'by_severity': {
                    'HIGH': 2,
                    'MEDIUM': 3,
                    'LOW': 0
                },
                'by_category': {
                    'MARKET_RISK': 2,
                    'OPERATIONAL_RISK': 1,
                    'FINANCIAL_RISK': 2
                },
                'risks': [...]  # Detailed list (if detailed=True)
            }
        
        Example:
            text = "The company faces significant market volatility..."
            
            risks = agent.analyze_risks(text)
            
            if risks['risk_level'] == 'HIGH':
                print("⚠️  High risk detected!")
                print(f"Total risks: {risks['total_risks']}")
        """
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
        """
        Analyze the sentiment (tone) of the document
        
        Determines if the document is positive (good news), negative (bad news),
        or neutral. Useful for understanding the overall tone.
        
        Args:
            text: Document text to analyze
        
        Returns:
            Dictionary with sentiment information:
            {
                'sentiment': 'POSITIVE',        # POSITIVE, NEGATIVE, NEUTRAL
                'confidence': 0.85,             # 0-1 (how confident)
                'breakdown': {
                    'POSITIVE': 10,  # Number of positive sentences
                    'NEGATIVE': 2,   # Number of negative sentences
                    'NEUTRAL': 5     # Number of neutral sentences
                }
            }
        
        Example:
            text = "Revenue exceeded expectations. Strong growth continues."
            
            sentiment = agent.analyze_sentiment(text)
            
            print(f"Sentiment: {sentiment['sentiment']}")
            # Output: Sentiment: POSITIVE
        """
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
        """
        Compare two documents side-by-side
        
        Use this to see which document/company is performing better.
        
        Args:
            doc1: First document with 'name', 'text', and 'period'
                 Example: {'name': 'Apple', 'text': '...', 'period': 'Q3 2024'}
            doc2: Second document (same format)
            analysis_type: What to compare:
                          - 'comprehensive': Everything
                          - 'metrics': Just numbers
                          - 'sentiment': Just tone
        
        Returns:
            Dictionary with comparison:
            {
                'metrics': {
                    'summary': {
                        'overall_winner': 'Apple',
                        'total_comparisons': 5
                    }
                },
                'sentiment': {
                    'entity1': 'POSITIVE',
                    'entity2': 'NEUTRAL'
                }
            }
        
        Example:
            apple = {'name': 'Apple', 'text': 'Revenue $90B...', 'period': 'Q3'}
            msft = {'name': 'Microsoft', 'text': 'Revenue $62B...', 'period': 'Q3'}
            
            comparison = agent.compare_documents(apple, msft)
            
            winner = comparison['metrics']['summary']['overall_winner']
            print(f"Winner: {winner}")
        """
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
        """
        Run all analyses on the documents
        
        This is the "do everything" method. It runs:
        - Trend analysis (if 2+ documents)
        - Sentiment analysis for each document
        - Risk analysis (on latest document)
        - Insight generation (recommendations)
        
        Args:
            documents: List of documents with 'period' and 'text'
            include_risks: Include risk analysis (default: True)
            include_insights: Generate insights (default: True)
        
        Returns:
            Dictionary with all analysis results:
            {
                'trends': {...},          # If 2+ documents
                'sentiment': {...},       # For all documents
                'risks': {...},          # If include_risks=True
                'insights': {...}        # If include_insights=True
            }
        
        Example:
            docs = [
                {'period': 'Q1', 'text': '...'},
                {'period': 'Q2', 'text': '...'}
            ]
            
            results = agent.comprehensive_analysis(docs)
            
            print(f"Trends: {results['trends']['summary']}")
            print(f"Risk Level: {results['risks']['risk_level']}")
            print(f"Insights: {results['insights']['total']}")
        """
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
        """
        Generate actionable insights and recommendations
        
        Looks at all the analysis and creates recommendations like:
        - "Revenue is declining - investigate root causes"
        - "High risk detected - implement mitigation plan"
        
        Args:
            documents: List of documents to analyze
        
        Returns:
            Dictionary with insights:
            {
                'total_insights': 5,
                'by_type': {
                    'WARNING': 2,
                    'OPPORTUNITY': 3
                },
                'top_insights': [
                    {
                        'type': 'WARNING',
                        'title': 'Declining Revenue',
                        'description': 'Revenue down 5%',
                        'impact': 'HIGH',
                        'action_items': ['Investigate causes', ...]
                    }
                ]
            }
        
        Example:
            insights = agent.generate_insights(documents)
            
            for insight in insights['top_insights']:
                print(f"{insight['type']}: {insight['title']}")
                print(f"Actions: {insight['action_items']}")
        """
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
    
    # ==================== HELPER METHODS ====================
    
    def _summarize_trends(self, trends: dict) -> dict:
        """
        Create a simple summary of trends
        
        Takes the detailed trend data and creates lists of:
        - improving: Metrics going up
        - declining: Metrics going down
        - stable: Metrics staying the same
        """
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
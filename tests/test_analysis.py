"""
Week 4 Testing: Trend Analysis & Insights
Tests analysis and insight generation modules
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Sample documents for testing
SAMPLE_DOCS = [
    {
        'period': 'Q1 2024',
        'name': 'Company A',
        'text': '''
        Company reported Q1 2024 results. Revenue was $85.3 billion with 
        net income of $23.2 billion. Strong performance across all segments.
        '''
    },
    {
        'period': 'Q2 2024',
        'name': 'Company A',
        'text': '''
        Q2 2024 showed continued growth. Revenue reached $87.5 billion with 
        net income of $24.1 billion. Market conditions remain favorable.
        '''
    },
    {
        'period': 'Q3 2024',
        'name': 'Company A',
        'text': '''
        Q3 2024 results exceeded expectations. Revenue totaled $90.1 billion
        with net income of $25.5 billion. Innovation driving growth.
        '''
    }
]

RISK_TEXT = """
The company faces significant market volatility and competitive pressure.
Supply chain disruptions continue to pose operational challenges. We are
monitoring potential regulatory changes. Cybersecurity threats remain a
concern. However, we maintain adequate liquidity.
"""


def test_trend_analyzer():
    """Test Trend Analyzer"""
    print("\n" + "="*60)
    print("TEST 1: Trend Analyzer")
    print("="*60)
    
    try:
        from src.analysis.trend_analyzer import TrendAnalyzer
        
        print("Initializing Trend Analyzer...")
        analyzer = TrendAnalyzer()
        
        print("\nAnalyzing revenue trend...")
        trend = analyzer.analyze_trend(SAMPLE_DOCS, 'revenue')
        
        print(f"✓ Trend Analysis Complete")
        print(f"  Direction: {trend.direction}")
        print(f"  Strength: {trend.strength:.2f}")
        print(f"  Data Points: {len(trend.data_points)}")
        print(f"  Average: {trend.average:.2f}")
        
        # Test growth calculation
        if len(trend.data_points) >= 2:
            growth = analyzer.calculate_growth_rate(trend)
            print(f"\nGrowth Metrics:")
            print(f"  Average Growth: {growth['average_growth']:.1f}%")
            print(f"  CAGR: {growth['cagr']:.1f}%")
        
        # Test anomaly detection
        anomalies = analyzer.detect_anomalies(trend)
        print(f"\nAnomalies: {len(anomalies)}")
        
        success = (
            len(trend.data_points) > 0 and
            trend.direction in ['INCREASING', 'DECREASING', 'STABLE', 'VOLATILE', 'NO_DATA']
        )
        
        return success
        
    except Exception as e:
        print(f"✗ Trend Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparative_analyzer():
    """Test Comparative Analyzer"""
    print("\n" + "="*60)
    print("TEST 2: Comparative Analyzer")
    print("="*60)
    
    try:
        from src.analysis.comparative_analyzer import ComparativeAnalyzer
        
        print("Initializing Comparative Analyzer...")
        analyzer = ComparativeAnalyzer()
        
        doc1 = {
            'name': 'Apple',
            'period': 'Q3 2024',
            'text': '''
            Apple reported revenue of $90.1 billion with net income of $25.5 billion.
            Gross margin was 46.3% and strong growth across all segments.
            '''
        }
        
        doc2 = {
            'name': 'Microsoft',
            'period': 'Q3 2024',
            'text': '''
            Microsoft generated revenue of $62.0 billion with net income of $22.3 billion.
            Gross margin stood at 69.5% and cloud revenue grew significantly.
            '''
        }
        
        print("\nComparing documents...")
        comparison = analyzer.compare_documents(doc1, doc2)
        
        print(f"✓ Comparison Complete")
        print(f"  Entities: {comparison['entities']}")
        print(f"  Metrics Compared: {len(comparison['metrics'])}")
        print(f"  Winner: {comparison['summary']['overall_winner']}")
        
        success = (
            len(comparison['metrics']) > 0 and
            'summary' in comparison
        )
        
        return success
        
    except Exception as e:
        print(f"✗ Comparative Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_analyzer():
    """Test Risk Analyzer"""
    print("\n" + "="*60)
    print("TEST 3: Risk Analyzer")
    print("="*60)
    
    try:
        from src.analysis.risk_analyzer import RiskAnalyzer
        
        print("Initializing Risk Analyzer...")
        analyzer = RiskAnalyzer()
        
        print("\nAnalyzing risks...")
        risks = analyzer.analyze_risks(RISK_TEXT)
        
        print(f"✓ Risk Analysis Complete")
        print(f"  Total Risks: {len(risks)}")
        
        if risks:
            print(f"\nSample risks:")
            for risk in risks[:3]:
                print(f"  • {risk.category}: {risk.severity}")
        
        # Calculate risk score
        score = analyzer.calculate_risk_score(risks)
        print(f"\nRisk Score:")
        print(f"  Level: {score['risk_level']}")
        print(f"  Score: {score['total_score']:.1f}/100")
        
        success = len(risks) > 0
        
        return success
        
    except Exception as e:
        print(f"✗ Risk Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_insight_generator():
    """Test Insight Generator"""
    print("\n" + "="*60)
    print("TEST 4: Insight Generator")
    print("="*60)
    
    try:
        from src.analysis.insight_generator import InsightGenerator
        
        print("Initializing Insight Generator...")
        generator = InsightGenerator()
        
        print("\nGenerating insights...")
        insights = generator.generate_insights(SAMPLE_DOCS)
        
        print(f"✓ Insight Generation Complete")
        print(f"  Total Insights: {len(insights)}")
        
        if insights:
            # Group by type
            by_type = {}
            for insight in insights:
                if insight.type not in by_type:
                    by_type[insight.type] = 0
                by_type[insight.type] += 1
            
            print(f"\nInsights by type:")
            for itype, count in by_type.items():
                print(f"  {itype}: {count}")
            
            print(f"\nTop insights:")
            for insight in insights[:3]:
                print(f"  • {insight.title} ({insight.impact})")
        
        success = len(insights) > 0
        
        return success
        
    except Exception as e:
        print(f"✗ Insight Generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test complete analysis pipeline"""
    print("\n" + "="*60)
    print("TEST 5: Integration Test")
    print("="*60)
    
    try:
        from src.analysis.trend_analyzer import TrendAnalyzer
        from src.analysis.risk_analyzer import RiskAnalyzer
        from src.analysis.insight_generator import InsightGenerator
        
        print("Loading all analysis modules...")
        trend = TrendAnalyzer()
        risk = RiskAnalyzer()
        insights = InsightGenerator()
        
        print("✓ All modules loaded")
        
        print("\nRunning complete analysis...")
        
        # Analyze trends
        revenue_trend = trend.analyze_trend(SAMPLE_DOCS, 'revenue')
        
        # Analyze risks
        risks = risk.analyze_risks(RISK_TEXT)
        risk_score = risk.calculate_risk_score(risks)
        
        # Generate insights
        all_insights = insights.generate_insights(SAMPLE_DOCS)
        
        print("\nAnalysis Results:")
        print(f"  Trend Direction: {revenue_trend.direction}")
        print(f"  Risks Identified: {len(risks)}")
        print(f"  Risk Level: {risk_score['risk_level']}")
        print(f"  Insights Generated: {len(all_insights)}")
        
        success = (
            len(revenue_trend.data_points) > 0 and
            len(risks) > 0 and
            len(all_insights) > 0
        )
        
        if success:
            print("\n✓ Integration test passed!")
        else:
            print("\n⚠ Integration test incomplete")
        
        return success
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Week 4 tests"""
    print("\n" + "="*60)
    print("WEEK 4 - TREND ANALYSIS & INSIGHTS TESTING")
    print("="*60)
    
    results = {}
    
    # Run tests
    results['Trend Analyzer'] = test_trend_analyzer()
    results['Comparative Analyzer'] = test_comparative_analyzer()
    results['Risk Analyzer'] = test_risk_analyzer()
    results['Insight Generator'] = test_insight_generator()
    results['Integration'] = test_integration()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:25s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Week 4 complete!")
        print("\nYou now have:")
        print("  ✓ Trend Analysis")
        print("  ✓ Comparative Analysis")
        print("  ✓ Risk Assessment")
        print("  ✓ Insight Generation")
        print("  ✓ Complete Analysis Pipeline")
        print("\nReady for Week 5-7: Dashboard & Deployment!")
    else:
        print("\n⚠ Some tests failed. Check errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
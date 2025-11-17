"""
Week 3 Testing: Entity Extraction & Analysis
Tests extraction modules and analysis pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Sample financial text for testing
SAMPLE_TEXT = """
Apple Inc. (AAPL) reported exceptional Q3 2024 financial results on July 30, 2024.
The company generated revenue of $90.1 billion, representing a 15.2% increase 
year-over-year. Net income reached $25.5 billion with diluted EPS of $1.85,
significantly beating analyst expectations of $1.52.

CEO Tim Cook stated that "iPhone sales grew 12% to $45.8 billion, driven by
strong demand for our latest models." The Services segment also performed well,
reaching $24.2 billion in revenue.

Gross margin improved to 46.3% compared to 44.5% in the prior quarter, while
operating margin expanded to 30.2%. The company ended the quarter with total
assets of $350 billion and shareholders equity of $65 billion.

Apple announced a quarterly dividend of $0.24 per share and expects Q4 2024
revenue to be between $88-92 billion. ROE stands at 39%, demonstrating strong
capital efficiency.
"""


def test_entity_extractor():
    """Test Entity Extractor"""
    print("\n" + "="*60)
    print("TEST 1: Entity Extractor")
    print("="*60)
    
    try:
        from src.extraction.entity_extractor import EntityExtractor
        
        print("Initializing Entity Extractor...")
        extractor = EntityExtractor()
        
        print("\nExtracting entities...")
        result = extractor.extract(SAMPLE_TEXT)
        
        print(f"✓ Extraction complete!")
        print(f"  Total entities: {result['metadata']['total_entities']}")
        print(f"  Metrics found: {result['metadata']['metrics_found']}")
        print(f"  Relations found: {result['metadata']['relations_found']}")
        
        # Show sample entities
        print("\nSample entities:")
        for entity in result['entities'][:5]:
            print(f"  {entity['type']:12s}: {entity['text']}")
        
        # Show metrics
        if result['metrics']:
            print("\nExtracted metrics:")
            for metric_name, values in list(result['metrics'].items())[:3]:
                print(f"  {metric_name}: {len(values)} occurrences")
        
        # Check if key entities were found
        entity_types = result['metadata']['entity_types']
        has_companies = entity_types.get('COMPANY', 0) > 0
        has_money = entity_types.get('MONEY', 0) > 0
        has_dates = entity_types.get('DATE', 0) > 0
        
        success = has_companies and has_money and result['metadata']['total_entities'] > 10
        
        if success:
            print("\n✓ Entity extraction working correctly")
        else:
            print("\n⚠ Some entities may not be extracted correctly")
        
        return success
        
    except Exception as e:
        print(f"✗ Entity extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metric_extractor():
    """Test Metric Extractor"""
    print("\n" + "="*60)
    print("TEST 2: Metric Extractor")
    print("="*60)
    
    try:
        from src.extraction.metric_extractor import MetricExtractor
        
        print("Initializing Metric Extractor...")
        extractor = MetricExtractor()
        
        print("\nExtracting metrics...")
        all_metrics = extractor.extract_all_metrics(SAMPLE_TEXT, period="Q3 2024")
        
        # Count total metrics
        total_metrics = sum(len(metrics) for metrics in all_metrics.values())
        print(f"✓ Extracted {total_metrics} metrics")
        
        # Show by category
        print("\nMetrics by category:")
        for category, metrics in all_metrics.items():
            if metrics:
                print(f"  {category}: {len(metrics)}")
                for metric in metrics[:2]:
                    formatted = extractor.format_metric(metric)
                    print(f"    • {formatted}")
        
        # Test growth metrics
        changes = extractor.extract_growth_metrics(SAMPLE_TEXT)
        print(f"\n✓ Found {len(changes)} growth/change metrics")
        
        # Calculate ratios
        ratios = extractor.calculate_ratios(all_metrics)
        if ratios:
            print(f"\n✓ Calculated {len(ratios)} financial ratios")
            for ratio_name, value in list(ratios.items())[:3]:
                print(f"  {ratio_name}: {value:.2f}")
        
        # Success criteria
        success = (
            total_metrics >= 5 and
            len(all_metrics.get('income_statement', [])) > 0
        )
        
        return success
        
    except Exception as e:
        print(f"✗ Metric extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_relation_extractor():
    """Test Relation Extractor"""
    print("\n" + "="*60)
    print("TEST 3: Relation Extractor")
    print("="*60)
    
    try:
        from src.extraction.relation_extractor import RelationExtractor
        
        print("Initializing Relation Extractor...")
        extractor = RelationExtractor()
        
        print("\nExtracting relations...")
        relations = extractor.extract_relations(SAMPLE_TEXT)
        
        print(f"✓ Found {len(relations)} relations")
        
        # Show sample relations
        print("\nSample relations:")
        for rel in relations[:5]:
            print(f"  {rel.subject} --[{rel.relation}]--> {rel.object}")
        
        # Get statistics
        stats = extractor.get_relation_statistics(SAMPLE_TEXT)
        print(f"\nRelation statistics:")
        print(f"  Unique subjects: {stats['unique_subjects']}")
        print(f"  Unique objects: {stats['unique_objects']}")
        print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
        
        # Build knowledge graph
        graph = extractor.build_knowledge_graph(SAMPLE_TEXT)
        print(f"\nKnowledge graph:")
        print(f"  Nodes: {len(graph['nodes'])}")
        print(f"  Edges: {len(graph['edges'])}")
        
        success = len(relations) >= 3 and stats['unique_subjects'] > 0
        
        return success
        
    except Exception as e:
        print(f"✗ Relation extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pattern_matcher():
    """Test Pattern Matcher"""
    print("\n" + "="*60)
    print("TEST 4: Pattern Matcher")
    print("="*60)
    
    try:
        from src.extraction.pattern_matcher import PatternMatcher
        
        print("Initializing Pattern Matcher...")
        matcher = PatternMatcher()
        
        print("\nMatching patterns...")
        matches = matcher.match_all(SAMPLE_TEXT)
        
        total_matches = sum(len(m) for m in matches.values())
        print(f"✓ Found {total_matches} pattern matches")
        print(f"  Across {len(matches)} different patterns")
        
        # Show sample matches
        print("\nSample pattern matches:")
        for pattern_name, pattern_matches in list(matches.items())[:5]:
            if pattern_matches:
                print(f"  {pattern_name}: {pattern_matches[0].matched_text}")
        
        # Test custom extractors
        ratios = matcher.extract_custom(SAMPLE_TEXT, 'financial_ratios')
        print(f"\n✓ Custom extractor found {len(ratios)} financial ratios")
        
        forward = matcher.extract_custom(SAMPLE_TEXT, 'forward_looking')
        print(f"✓ Found {len(forward)} forward-looking statements")
        
        # Get statistics
        stats = matcher.get_match_statistics(SAMPLE_TEXT)
        print(f"\nPattern statistics:")
        print(f"  Total matches: {stats['total_matches']}")
        print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
        
        success = total_matches >= 10 and len(matches) >= 5
        
        return success
        
    except Exception as e:
        print(f"✗ Pattern matcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test all extraction modules together"""
    print("\n" + "="*60)
    print("TEST 5: Integration Test")
    print("="*60)
    
    try:
        from src.extraction.entity_extractor import EntityExtractor
        from src.extraction.metric_extractor import MetricExtractor
        from src.extraction.relation_extractor import RelationExtractor
        from src.extraction.pattern_matcher import PatternMatcher
        
        print("Loading all extraction modules...")
        entity_ext = EntityExtractor()
        metric_ext = MetricExtractor()
        relation_ext = RelationExtractor()
        pattern_match = PatternMatcher()
        
        print("✓ All modules loaded")
        
        print("\nRunning complete extraction pipeline...")
        
        # Extract everything
        entities = entity_ext.extract(SAMPLE_TEXT)
        metrics = metric_ext.extract_all_metrics(SAMPLE_TEXT)
        relations = relation_ext.extract_relations(SAMPLE_TEXT)
        patterns = pattern_match.match_all(SAMPLE_TEXT)
        
        # Create comprehensive analysis
        analysis = {
            'entities': {
                'total': entities['metadata']['total_entities'],
                'by_type': entities['metadata']['entity_types'],
                'metrics': len(entities['metrics'])
            },
            'financial_metrics': {
                'total': sum(len(m) for m in metrics.values()),
                'categories': list(k for k, v in metrics.items() if v)
            },
            'relations': {
                'total': len(relations),
                'unique_entities': len(set(r.subject for r in relations) | set(r.object for r in relations))
            },
            'patterns': {
                'total': sum(len(m) for m in patterns.values()),
                'types': len(patterns)
            }
        }
        
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nEntities:")
        print(f"  Total: {analysis['entities']['total']}")
        print(f"  Types: {', '.join(analysis['entities']['by_type'].keys())}")
        
        print(f"\nFinancial Metrics:")
        print(f"  Total: {analysis['financial_metrics']['total']}")
        print(f"  Categories: {', '.join(analysis['financial_metrics']['categories'])}")
        
        print(f"\nRelations:")
        print(f"  Total: {analysis['relations']['total']}")
        print(f"  Unique entities: {analysis['relations']['unique_entities']}")
        
        print(f"\nPattern Matches:")
        print(f"  Total: {analysis['patterns']['total']}")
        print(f"  Pattern types: {analysis['patterns']['types']}")
        
        # Success criteria
        success = (
            analysis['entities']['total'] > 5 and
            analysis['financial_metrics']['total'] > 3 and
            analysis['relations']['total'] > 2 and
            analysis['patterns']['total'] > 5
        )
        
        if success:
            print("\n✓ Integration test passed!")
            print("  All extraction modules working together correctly")
        else:
            print("\n⚠ Integration test needs attention")
        
        return success
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Week 3 tests"""
    print("\n" + "="*60)
    print("WEEK 3 - EXTRACTION & ANALYSIS TESTING SUITE")
    print("="*60)
    print("\nSample Text Preview:")
    print(SAMPLE_TEXT[:200] + "...")
    
    results = {}
    
    # Run tests
    print("\n" + "="*60)
    print("Running Tests...")
    print("="*60)
    
    results['Entity Extractor'] = test_entity_extractor()
    results['Metric Extractor'] = test_metric_extractor()
    results['Relation Extractor'] = test_relation_extractor()
    results['Pattern Matcher'] = test_pattern_matcher()
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
        print("\n🎉 All tests passed! Week 3 complete!")
        print("\nYou now have:")
        print("  ✓ Advanced Entity Extraction")
        print("  ✓ Financial Metric Extraction")
        print("  ✓ Relationship Extraction")
        print("  ✓ Pattern Matching System")
        print("  ✓ Complete Extraction Pipeline")
        print("\nReady for Week 4: Trend Analysis & Insights")
    else:
        print("\n⚠ Some tests failed. Check errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
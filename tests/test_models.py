"""
Week 2 Testing: NLP Models
Tests NER, Sentiment, QA, and Summarization modules
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Sample financial text for testing
SAMPLE_TEXT = """
Apple Inc. (AAPL) reported outstanding financial results for Q3 2024. 
The company generated revenue of $90.1 billion, representing a 15.2% 
increase year-over-year. Net income reached $25.5 billion, with earnings 
per share (EPS) of $1.85, significantly beating analyst expectations of 
$1.52. CEO Tim Cook stated that the strong performance was driven by 
robust iPhone sales, which grew 12% to $45.8 billion. The Services 
segment also performed exceptionally well, reaching $24.2 billion. 
The company announced a quarterly dividend of $0.24 per share and 
expressed confidence in future growth prospects.
"""


def test_ner():
    """Test Named Entity Recognition"""
    print("\n" + "="*60)
    print("TEST 1: Named Entity Recognition (NER)")
    print("="*60)
    
    try:
        from src.models.ner_model import FinancialNER
        
        print("Initializing NER model...")
        ner = FinancialNER()
        
        print("\nExtracting entities...")
        entities = ner.extract_entities(SAMPLE_TEXT)
        
        print(f"✓ Found {len(entities)} entities")
        
        # Show sample entities
        print("\nSample entities:")
        for entity in entities[:5]:
            print(f"  {entity['label']:15s}: {entity['text']} (score: {entity['score']:.2f})")
        
        # Get summary
        summary = ner.get_entity_summary(SAMPLE_TEXT)
        print(f"\nEntity types found: {len(summary)}")
        for label, info in list(summary.items())[:3]:
            print(f"  {label}: {info['count']} unique")
        
        # Extract financial metrics
        metrics = ner.extract_financial_metrics(SAMPLE_TEXT)
        print(f"\nFinancial metrics: {len(metrics)}")
        for metric, value in list(metrics.items())[:3]:
            print(f"  {metric}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ NER test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment():
    """Test Sentiment Analysis"""
    print("\n" + "="*60)
    print("TEST 2: Sentiment Analysis")
    print("="*60)
    
    try:
        from src.models.sentiment_model import FinancialSentiment
        
        print("Initializing sentiment analyzer...")
        sentiment = FinancialSentiment()
        
        print("\nAnalyzing sentiment...")
        result = sentiment.analyze(SAMPLE_TEXT)
        
        print(f"✓ Sentiment: {result['label']}")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Confidence: {result['confidence']}")
        
        # Test different text types
        test_texts = {
            'Positive': "Revenue exceeded expectations with strong growth.",
            'Negative': "The company reported declining sales and losses.",
            'Neutral': "The quarterly report was released on schedule."
        }
        
        print("\nTesting different sentiments:")
        for expected, text in test_texts.items():
            result = sentiment.analyze(text)
            match = "✓" if expected.upper() in result['label'] else "✗"
            print(f"  {match} Expected {expected}, got {result['label']} (score: {result['score']:.2f})")
        
        # Overall sentiment
        overall = sentiment.get_overall_sentiment(SAMPLE_TEXT)
        print(f"\nOverall sentiment: {overall['overall_sentiment']}")
        print(f"  Confidence: {overall['confidence']:.2f}")
        print(f"  Sentences analyzed: {overall['total_sentences']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sentiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qa():
    """Test Question Answering"""
    print("\n" + "="*60)
    print("TEST 3: Question Answering")
    print("="*60)
    
    try:
        from src.models.qa_model import FinancialQA
        
        print("Initializing QA system...")
        qa = FinancialQA()
        
        # Test questions
        test_questions = [
            ("What was the revenue?", "$90.1 billion"),
            ("What was the EPS?", "$1.85"),
            ("Who is the CEO?", "Tim Cook"),
            ("What was the dividend?", "$0.24")
        ]
        
        print("\nTesting questions:")
        correct = 0
        
        for question, expected_keyword in test_questions:
            result = qa.answer(question, SAMPLE_TEXT)
            
            # Check if expected keyword in answer
            if result['answer'] and expected_keyword.lower() in result['answer'].lower():
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"  {status} Q: {question}")
            print(f"     A: {result['answer']} (score: {result['score']:.2f})")
        
        accuracy = (correct / len(test_questions)) * 100
        print(f"\nQA Accuracy: {accuracy:.0f}% ({correct}/{len(test_questions)})")
        
        # Extract key facts
        print("\nExtracting key facts...")
        facts = qa.extract_key_facts(SAMPLE_TEXT)
        print(f"✓ Extracted {len(facts)} facts:")
        for fact_type, data in facts.items():
            print(f"  {fact_type}: {data['value']} (conf: {data['confidence']:.2f})")
        
        return accuracy >= 50  # At least 50% correct
        
    except Exception as e:
        print(f"✗ QA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_summarization():
    """Test Summarization"""
    print("\n" + "="*60)
    print("TEST 4: Summarization")
    print("="*60)
    
    try:
        from src.models.summarizer_model import FinancialSummarizer
        
        print("Initializing summarizer...")
        summarizer = FinancialSummarizer()
        
        # Generate summaries
        print("\nGenerating summaries...")
        
        short = summarizer.summarize(SAMPLE_TEXT, summary_type='short')
        balanced = summarizer.summarize(SAMPLE_TEXT, summary_type='balanced')
        
        print(f"✓ Short summary ({short['summary_length']} words):")
        print(f"  {short['summary'][:100]}...")
        
        print(f"\n✓ Balanced summary ({balanced['summary_length']} words):")
        print(f"  {balanced['summary'][:150]}...")
        
        # Verify summaries are shorter than original
        original_words = len(SAMPLE_TEXT.split())
        short_ok = short['summary_length'] < original_words
        balanced_ok = balanced['summary_length'] < original_words
        
        print(f"\nCompression:")
        print(f"  Original: {original_words} words")
        print(f"  Short: {short['summary_length']} words ({short['compression_ratio']:.1%})")
        print(f"  Balanced: {balanced['summary_length']} words ({balanced['compression_ratio']:.1%})")
        
        # Extract key sentences
        key_sentences = summarizer.extract_key_sentences(SAMPLE_TEXT, 3)
        print(f"\n✓ Extracted {len(key_sentences)} key sentences")
        
        return short_ok and balanced_ok
        
    except Exception as e:
        print(f"✗ Summarization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test all models together"""
    print("\n" + "="*60)
    print("TEST 5: Integration Test")
    print("="*60)
    
    try:
        from src.models.ner_model import FinancialNER
        from src.models.sentiment_model import FinancialSentiment
        from src.models.qa_model import FinancialQA
        from src.models.summarizer_model import FinancialSummarizer
        
        print("Loading all models...")
        ner = FinancialNER()
        sentiment = FinancialSentiment()
        qa = FinancialQA()
        summarizer = FinancialSummarizer()
        
        print("✓ All models loaded")
        
        print("\nProcessing sample document...")
        
        # NER
        entities = ner.extract_entities(SAMPLE_TEXT)
        
        # Sentiment
        sentiment_result = sentiment.get_overall_sentiment(SAMPLE_TEXT)
        
        # QA
        key_facts = qa.extract_key_facts(SAMPLE_TEXT)
        
        # Summary
        summary = summarizer.summarize(SAMPLE_TEXT, summary_type='balanced')
        
        # Create analysis report
        report = {
            'entities_found': len(entities),
            'sentiment': sentiment_result['overall_sentiment'],
            'key_facts_extracted': len(key_facts),
            'summary_generated': bool(summary.get('summary'))
        }
        
        print("\nAnalysis Report:")
        for key, value in report.items():
            print(f"  {key}: {value}")
        
        # All should have produced results
        success = all([
            report['entities_found'] > 0,
            report['sentiment'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
            report['key_facts_extracted'] > 0,
            report['summary_generated']
        ])
        
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
    """Run all Week 2 tests"""
    print("\n" + "="*60)
    print("WEEK 2 - NLP MODELS TESTING SUITE")
    print("="*60)
    print("\nSample Text:")
    print(SAMPLE_TEXT[:200] + "...")
    
    results = {}
    
    # Run tests
    print("\n" + "="*60)
    print("Running Tests...")
    print("="*60)
    
    results['NER'] = test_ner()
    results['Sentiment'] = test_sentiment()
    results['QA'] = test_qa()
    results['Summarization'] = test_summarization()
    results['Integration'] = test_integration()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Week 2 complete!")
        print("\nYou now have:")
        print("  ✓ Named Entity Recognition")
        print("  ✓ Sentiment Analysis")
        print("  ✓ Question Answering")
        print("  ✓ Text Summarization")
        print("\nReady for Week 3: Entity Extraction & Analysis")
    else:
        print("\n⚠ Some tests failed. Check errors above.")
        print("Note: First run may be slower due to model downloads.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
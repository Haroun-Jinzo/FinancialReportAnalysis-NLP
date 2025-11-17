"""
Week 1 Testing: Document Parsing and Text Cleaning
Run this to verify your setup is working correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.document_parser import DocumentParser
from src.preprocessing.text_cleaner import TextCleaner


def test_document_parser():
    """Test document parser with sample text"""
    print("\n" + "="*60)
    print("TEST 1: Document Parser")
    print("="*60)
    
    parser = DocumentParser(enable_ocr=False)
    
    # Create a sample text file for testing
    test_file = Path("data/raw/test_financial_report.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    sample_content = """
    FINANCIAL REPORT - Q3 2024
    
    Executive Summary
    The company reported strong financial results for Q3 2024.
    Revenue reached $125 million, up 18% year-over-year.
    Net income was $32.5 million with EPS of $2.15.
    
    Key Metrics:
    - Revenue: $125M
    - Net Income: $32.5M  
    - Operating Margin: 26%
    - EPS: $2.15
    
    The strong performance was driven by increased demand and 
    operational efficiency improvements.
    """
    
    with open(test_file, 'w') as f:
        f.write(sample_content)
    
    print(f"✓ Created test file: {test_file}")
    
    # Parse the document
    try:
        result = parser.parse(str(test_file))
        print(f"✓ Successfully parsed document")
        print(f"  - Format: {result['format']}")
        print(f"  - Text length: {len(result['text'])} chars")
        print(f"  - Lines: {len(result['lines'])}")
        
        # Get summary
        summary = parser.get_document_summary(result)
        print(f"  - Word count: {summary['word_count']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Parser test failed: {e}")
        return False


def test_text_cleaner():
    """Test text cleaner"""
    print("\n" + "="*60)
    print("TEST 2: Text Cleaner")
    print("="*60)
    
    cleaner = TextCleaner()
    
    # Sample financial text
    sample = """
    Apple Inc. (AAPL) reported Q3 2024 revenue of $90.1 billion, 
    a 15.2% increase Y-o-Y. Net income: $25.5B. EPS was $1.85 
    (vs $1.52 expected). iPhone revenue grew 12% to $45.8B.
    """
    
    print("Original text:")
    print(sample)
    print()
    
    # Test cleaning
    try:
        cleaned = cleaner.clean(
            sample,
            preserve_financial=True,
            lowercase=False
        )
        print("✓ Text cleaned successfully")
        print("Cleaned text:")
        print(cleaned)
        print()
        
        # Test number extraction
        numbers = cleaner.extract_financial_numbers(sample)
        print(f"✓ Extracted {len(numbers)} financial numbers:")
        for num in numbers[:3]:  # Show first 3
            print(f"  - {num['value']} {num['unit']}")
        print()
        
        # Test keyword extraction
        keywords = cleaner.extract_keywords(sample, top_n=5)
        print(f"✓ Extracted keywords:")
        for word, count in keywords:
            print(f"  - {word}: {count}")
        print()
        
        # Test sentence extraction
        sentences = cleaner.extract_sentences(sample)
        print(f"✓ Extracted {len(sentences)} sentences")
        
        # Test financial detection
        is_financial = cleaner.is_financial_text(sample)
        print(f"✓ Is financial text: {is_financial}")
        
        return True
        
    except Exception as e:
        print(f"✗ Cleaner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test parser and cleaner together"""
    print("\n" + "="*60)
    print("TEST 3: Integration Test")
    print("="*60)
    
    try:
        # Parse document
        parser = DocumentParser()
        test_file = "data/raw/test_financial_report.txt"
        
        parsed = parser.parse(test_file)
        print(f"✓ Parsed document: {parsed['filename']}")
        
        # Clean the text
        cleaner = TextCleaner()
        cleaned_text = cleaner.clean(parsed['text'], preserve_financial=True)
        print(f"✓ Cleaned text ({len(cleaned_text)} chars)")
        
        # Extract information
        numbers = cleaner.extract_financial_numbers(parsed['text'])
        keywords = cleaner.extract_keywords(cleaned_text, top_n=5)
        
        print(f"✓ Extracted {len(numbers)} numbers, {len(keywords)} keywords")
        
        # Create simple pipeline result
        pipeline_result = {
            'document': parsed['filename'],
            'raw_text_length': len(parsed['text']),
            'cleaned_text_length': len(cleaned_text),
            'financial_numbers_count': len(numbers),
            'top_keywords': [k[0] for k in keywords],
            'is_financial': cleaner.is_financial_text(parsed['text'])
        }
        
        print("\nPipeline Result:")
        for key, value in pipeline_result.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Week 1 tests"""
    print("\n" + "="*60)
    print("WEEK 1 - TESTING SUITE")
    print("Document Parsing & Text Cleaning")
    print("="*60)
    
    results = {}
    
    # Run tests
    results['Parser'] = test_document_parser()
    results['Cleaner'] = test_text_cleaner()
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
        print("\n🎉 All tests passed! Week 1 complete!")
        print("\nYou can now:")
        print("  1. Process real financial documents")
        print("  2. Move to Week 2: NLP Models")
    else:
        print("\n⚠ Some tests failed. Check errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
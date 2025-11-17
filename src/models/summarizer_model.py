"""
Text Summarization Module
Generate summaries of financial documents using BART
"""

from typing import List, Dict, Optional
import re

from model_loader import ModelLoader


class FinancialSummarizer:
    """
    Abstractive summarization for financial documents
    Uses BART (Bidirectional and Auto-Regressive Transformers)
    """
    
    def __init__(self):
        """Initialize summarizer"""
        print("Initializing Financial Summarizer...")
        
        # Load BART model
        self.loader = ModelLoader()
        self.model_data = self.loader.load_model('summarization')
        self.pipeline = self.model_data['pipeline']
        
        # Summary configuration
        self.default_config = {
            'min_length': 50,
            'max_length': 150,
            'do_sample': False,
            'early_stopping': True
        }
        
        print("✓ Financial Summarizer initialized")
    
    def summarize(self, text: str,
                 summary_type: str = 'balanced',
                 max_length: Optional[int] = None,
                 min_length: Optional[int] = None) -> Dict:
        """
        Generate summary of text
        
        Args:
            text: Input text to summarize
            summary_type: 'short', 'balanced', or 'detailed'
            max_length: Maximum summary length (overrides type)
            min_length: Minimum summary length (overrides type)
            
        Returns:
            Dictionary with summary and metadata
        """
        if not text or len(text.strip()) < 100:
            return {
                'summary': text,
                'original_length': len(text),
                'summary_length': len(text),
                'compression_ratio': 1.0,
                'warning': 'Text too short to summarize'
            }
        
        # Set length based on summary type
        if summary_type == 'short':
            min_len = min_length or 30
            max_len = max_length or 80
        elif summary_type == 'detailed':
            min_len = min_length or 100
            max_len = max_length or 250
        else:  # balanced
            min_len = min_length or 50
            max_len = max_length or 150
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Generate summary
            summary_result = self.pipeline(
                processed_text,
                min_length=min_len,
                max_length=max_len,
                do_sample=False,
                early_stopping=True
            )[0]
            
            summary = summary_result['summary_text']
            
            # Post-process summary
            summary = self._postprocess_summary(summary)
            
            # Calculate metrics
            original_length = len(text.split())
            summary_length = len(summary.split())
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            
            return {
                'summary': summary,
                'original_length': original_length,
                'summary_length': summary_length,
                'compression_ratio': compression_ratio,
                'summary_type': summary_type
            }
            
        except Exception as e:
            return {
                'summary': None,
                'error': str(e)
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before summarization"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # BART has a max token limit of ~1024
        # Truncate if too long (roughly 800 words = 1000 tokens)
        words = text.split()
        if len(words) > 800:
            text = ' '.join(words[:800])
        
        return text
    
    def _postprocess_summary(self, summary: str) -> str:
        """Post-process generated summary"""
        # Remove extra whitespace
        summary = ' '.join(summary.split())
        
        # Ensure ends with period
        if summary and not summary[-1] in '.!?':
            summary += '.'
        
        # Capitalize first letter
        if summary:
            summary = summary[0].upper() + summary[1:]
        
        return summary
    
    def summarize_long_document(self, text: str,
                                chunk_size: int = 800,
                                final_summary_length: int = 200) -> Dict:
        """
        Summarize long documents using chunking strategy
        
        Args:
            text: Long input text
            chunk_size: Size of each chunk (in words)
            final_summary_length: Length of final summary
            
        Returns:
            Dictionary with hierarchical summary
        """
        words = text.split()
        
        # If short enough, summarize directly
        if len(words) <= chunk_size:
            return self.summarize(text, max_length=final_summary_length)
        
        # Split into chunks
        chunks = []
        for i in range(0, len(words), chunk_size - 100):  # 100 word overlap
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            result = self.summarize(chunk, summary_type='short')
            if result.get('summary'):
                chunk_summaries.append(result['summary'])
        
        # Combine chunk summaries
        combined = ' '.join(chunk_summaries)
        
        # Final summarization
        final_result = self.summarize(
            combined,
            max_length=final_summary_length
        )
        
        return {
            'summary': final_result.get('summary'),
            'original_length': len(words),
            'num_chunks': len(chunks),
            'chunk_summaries': chunk_summaries,
            'method': 'hierarchical'
        }
    
    def extract_key_sentences(self, text: str, num_sentences: int = 3) -> List[str]:
        """
        Extract key sentences (extractive summarization)
        Complement to abstractive summarization
        
        Args:
            text: Input text
            num_sentences: Number of sentences to extract
            
        Returns:
            List of key sentences
        """
        from nltk.tokenize import sent_tokenize
        from collections import Counter
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return sentences
        
        # Score sentences by keyword frequency
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        for word in stopwords:
            word_freq.pop(word, None)
        
        # Score each sentence
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            words_in_sentence = re.findall(r'\b\w+\b', sentence.lower())
            
            for word in words_in_sentence:
                score += word_freq.get(word, 0)
            
            # Boost if contains financial keywords
            financial_keywords = ['revenue', 'profit', 'earnings', 'growth', 'increase']
            if any(kw in sentence.lower() for kw in financial_keywords):
                score *= 1.5
            
            sentence_scores.append((score, i, sentence))
        
        # Sort by score and take top N
        sentence_scores.sort(reverse=True)
        top_sentences = sentence_scores[:num_sentences]
        
        # Sort back by original order
        top_sentences.sort(key=lambda x: x[1])
        
        return [sent for _, _, sent in top_sentences]
    
    def generate_executive_summary(self, text: str) -> Dict:
        """
        Generate executive summary with key highlights
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary with executive summary components
        """
        # Generate abstractive summary
        main_summary = self.summarize(text, summary_type='balanced')
        
        # Extract key sentences
        key_points = self.extract_key_sentences(text, num_sentences=5)
        
        # Generate short overview
        overview = self.summarize(text, summary_type='short')
        
        return {
            'overview': overview.get('summary'),
            'main_summary': main_summary.get('summary'),
            'key_points': key_points,
            'word_count': main_summary.get('summary_length', 0)
        }
    
    def summarize_by_section(self, text: str,
                            section_headers: Optional[List[str]] = None) -> Dict:
        """
        Summarize document by sections
        
        Args:
            text: Full document text
            section_headers: List of section headers to look for
            
        Returns:
            Dictionary of section summaries
        """
        if section_headers is None:
            # Common financial report sections
            section_headers = [
                'Executive Summary',
                'Financial Highlights',
                'Revenue',
                'Operations',
                'Outlook'
            ]
        
        sections = self._split_by_sections(text, section_headers)
        
        summaries = {}
        for section_name, section_text in sections.items():
            if len(section_text.split()) > 50:  # Only summarize substantial sections
                result = self.summarize(section_text, summary_type='short')
                summaries[section_name] = result.get('summary')
        
        return summaries
    
    def _split_by_sections(self, text: str, 
                          headers: List[str]) -> Dict[str, str]:
        """Split text by section headers"""
        sections = {}
        text_lower = text.lower()
        
        for header in headers:
            header_lower = header.lower()
            
            # Find header position
            pos = text_lower.find(header_lower)
            if pos != -1:
                # Find next header or end of text
                next_pos = len(text)
                for other_header in headers:
                    if other_header != header:
                        other_pos = text_lower.find(other_header.lower(), pos + len(header))
                        if other_pos != -1 and other_pos < next_pos:
                            next_pos = other_pos
                
                # Extract section text
                section_text = text[pos:next_pos].strip()
                sections[header] = section_text
        
        return sections
    
    def compare_summaries(self, text: str) -> Dict:
        """
        Generate multiple summary types for comparison
        
        Returns:
            Dictionary with different summary types
        """
        return {
            'short': self.summarize(text, summary_type='short')['summary'],
            'balanced': self.summarize(text, summary_type='balanced')['summary'],
            'detailed': self.summarize(text, summary_type='detailed')['summary'],
            'extractive': ' '.join(self.extract_key_sentences(text, 3))
        }


# Example usage
if __name__ == "__main__":
    # Initialize summarizer
    summarizer = FinancialSummarizer()
    
    # Sample text
    sample_text = """
    Apple Inc. reported its financial results for the third fiscal quarter
    of 2024, ended June 29, 2024. The Company posted quarterly revenue of 
    $90.1 billion, up 15 percent year over year, and quarterly earnings per 
    diluted share of $1.85. The results exceeded analyst expectations across
    all major categories. iPhone revenue grew 12 percent to $45.8 billion,
    driven by strong demand for the iPhone 15 Pro models. Services revenue
    reached an all-time high of $24.2 billion, up 14 percent year over year.
    Mac revenue was $8.2 billion, up 2 percent, while iPad revenue increased
    to $7.4 billion. The company's gross margin improved to 46.3 percent.
    CEO Tim Cook stated that the strong performance was driven by the strength
    of the iPhone lineup and the growing services business. The company
    also announced a quarterly dividend of $0.24 per share.
    """
    
    print("\nOriginal Text:")
    print(sample_text)
    print(f"\nOriginal length: {len(sample_text.split())} words")
    
    print("\n" + "="*60)
    print("SUMMARIZATION EXAMPLES")
    print("="*60)
    
    # Short summary
    short = summarizer.summarize(sample_text, summary_type='short')
    print(f"\nShort Summary ({short['summary_length']} words):")
    print(short['summary'])
    
    # Balanced summary
    balanced = summarizer.summarize(sample_text, summary_type='balanced')
    print(f"\nBalanced Summary ({balanced['summary_length']} words):")
    print(balanced['summary'])
    
    # Key sentences
    key_sentences = summarizer.extract_key_sentences(sample_text, 3)
    print(f"\nKey Sentences (extractive):")
    for i, sent in enumerate(key_sentences, 1):
        print(f"{i}. {sent}")
    
    print("\n✓ Summarization Module Ready!")
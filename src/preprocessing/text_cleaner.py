import re
import string
from typing import List, Dict, Optional
import unicodedata

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)


class TextCleaner:
    """
    Clean and preprocess text for NLP analysis
    Preserves financial information while removing noise
    """

    def __init__(self, language: str = 'english'):
        """
        Initialize text cleaner
        
        Args:
            language: Language for stopwords and processing
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

        # load SpaCy models (small for speed)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Warning: Spacy model not loaded. Some features disabled.")
            self.nlp = None

        # Financial terms to preserve (don't remove as stopwords)
        self.financial_terms = {
            'revenue', 'profit', 'loss', 'income', 'earnings', 'ebitda',
            'assets', 'liabilities', 'equity', 'debt', 'cash', 'flow',
            'margin', 'quarter', 'fiscal', 'year', 'growth', 'decline',
            'eps', 'roe', 'roi', 'ratio', 'million', 'billion', 'increase',
            'decrease', 'market', 'share', 'stock', 'dividend'
        }

        # Patterns for financial entities (preserve these)
        self.financial_patterns = {
            'money': r'\$[\d,]+\.?\d*[MBK]?',
            'percentage': r'\d+\.?\d*%',
            'date': r'\b(?:Q[1-4]|FY)\s*\d{4}\b',
            'number': r'\b\d+\.?\d*[MBK]?\b'
        }
    
    def clean(self, text:str, remove_stopwords: bool = False,
              lemmatize: bool = True,
              lowercase: bool = True,
              remove_numbers: bool = False,
              preserve_financial: bool = True) -> str:
        """
        Main cleaning function
        
        Args:
            text: Input text
            remove_stopwords: Remove common words
            lemmatize: Convert words to base form
            lowercase: Convert to lowercase
            remove_numbers: Remove numeric values
            preserve_financial: Keep financial terms and numbers
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return""
        
        #Step 1: Normalize unicode
        text = self._normalize_unicode(text)

        #Step 2: preserve financial entities (replace with placeholders)
        protected_entities = {}
        if preserve_financial:
            text, protected_entities = self._protect_financial_entities(text)

        #Step 3: Remove Special Characters but keep instance structure
        text = self._clean_special_chars(text)

        #Step 4: Normalize whitespace
        text = self._normalize_whitespace(text)

        #Step 5: lowercase
        if lowercase:
            text = text.lower()
        
        #Step 6: Tokenization
        tokens = word_tokenize(text)

        #Step 7: remove stopwords (careful with financial context)
        if remove_stopwords:
            tokens = [t for t in tokens
                      if t.lower() not in self.stop_words
                      or t.lower() in self.financial_terms]
            
        #Step 8: remove numbers usually keep for financial docs
        if remove_numbers and not preserve_financial:
            tokens = [t for t in tokens if not t.isdigit()]

        #Step 9: lemmatization
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        #Step 10: Reconstruct text
        text = ''.join(tokens)

        #Step 11: Restore Protected entities
        if preserve_financial:
            text = self._restore_financial_entities(text, protected_entities)

        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Convert to NFKD form and encode/decode to remove accents
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('utf-8')
        return text
    
    def _protect_financial_entities(self, text: str) -> tuple:
        """
        Replace financial entities with placeholders
        Returns: (modified_text, mapping_dict)
        """
        protected = {}
        counter = 0
        
        # Protect each pattern type
        for pattern_name, pattern in self.financial_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                placeholder = f"__{pattern_name.upper()}_{counter}__"
                protected[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
                counter += 1
        
        return text, protected
    
    def _restore_financial_entities(self, text: str, protected: Dict) -> str:
        """Restore protected financial entities"""
        for placeholder, original in protected.items():
            text = text.replace(placeholder, original)
        return text
    
    def _clean_special_chars(self, text: str) -> str:
        """Remove or replace special characters"""
        # Keep periods, commas, and important punctuation
        # Remove other special chars
        text = re.sub(r'[^\w\s\.\,\$\%\-\(\)]', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = text.replace('©', '')
        text = text.replace('®', '')
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text
        
        Returns:
            List of sentences
        """
        sentences = sent_tokenize(text)
        # Clean each sentence
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[tuple]:
        """
        Extract important keywords using TF-IDF-like approach
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of (word, score) tuples
        """
        from collections import Counter
        
        # Clean and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords
        tokens = [t for t in tokens 
                 if t.isalnum() 
                 and t not in self.stop_words
                 and len(t) > 2]
        
        # Count frequencies
        freq = Counter(tokens)
        
        # Boost financial terms
        for term in self.financial_terms:
            if term in freq:
                freq[term] *= 2
        
        # Return top N
        return freq.most_common(top_n)
    
    def chunk_text(self, text: str, chunk_size: int = 500, 
                   overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for processing
        Useful for long documents with token limits
        
        Args:
            text: Input text
            chunk_size: Size of each chunk in words
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += (chunk_size - overlap)
        
        return chunks
    
    def remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate text from financial documents
        """
        boilerplate_patterns = [
            r'This document contains forward-looking statements.*?(?=\n\n|\Z)',
            r'Page \d+ of \d+',
            r'Copyright © \d{4}',
            r'All rights reserved',
            r'Confidential and proprietary',
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text
    
    def extract_financial_numbers(self, text: str) -> List[Dict]:
        """
        Extract all financial numbers with context
        
        Returns:
            List of dicts with number, unit, and context
        """
        numbers = []
        
        # Pattern for money values
        money_pattern = r'(\$?[\d,]+\.?\d*)\s*(million|billion|M|B|K)?'
        matches = re.finditer(money_pattern, text, re.IGNORECASE)
        
        for match in matches:
            value = match.group(1).replace(',', '').replace('$', '')
            unit = match.group(2) if match.group(2) else 'units'
            
            # Get surrounding context (20 chars before and after)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            
            try:
                numeric_value = float(value)
                numbers.append({
                    'value': numeric_value,
                    'unit': unit.upper() if unit else 'UNITS',
                    'context': context.strip(),
                    'original': match.group(0)
                })
            except:
                continue
        
        return numbers
    
    def is_financial_text(self, text: str) -> bool:
        """
        Check if text is financial-related
        
        Returns:
            True if text contains financial keywords
        """
        text_lower = text.lower()
        
        # Count financial keywords
        keyword_count = sum(1 for term in self.financial_terms 
                           if term in text_lower)
        
        # Check for financial patterns
        has_money = bool(re.search(self.financial_patterns['money'], text))
        has_percentage = bool(re.search(self.financial_patterns['percentage'], text))
        
        # Consider it financial if it has keywords or patterns
        return keyword_count >= 3 or has_money or has_percentage


# Example usage
if __name__ == "__main__":
    cleaner = TextCleaner()
    
    # Test text
    sample_text = """
    Apple Inc. reported revenue of $90.1 billion in Q3 2024, representing 
    a 15% increase from the previous quarter. Net income was $25.5B, with 
    EPS of $1.85. The company's strong performance was driven by iPhone sales.
    """
    
    print("Original:")
    print(sample_text)
    print("\n" + "="*60 + "\n")
    
    print("Cleaned (preserve financial):")
    cleaned = cleaner.clean(sample_text, preserve_financial=True)
    print(cleaned)
    print("\n" + "="*60 + "\n")
    
    print("Extracted numbers:")
    numbers = cleaner.extract_financial_numbers(sample_text)
    for num in numbers:
        print(f"  {num['value']} {num['unit']}: {num['context']}")
    print("\n" + "="*60 + "\n")
    
    print("Keywords:")
    keywords = cleaner.extract_keywords(sample_text, top_n=10)
    for word, count in keywords:
        print(f"  {word}: {count}")
    
    print("\n✓ Text Cleaner Module Ready!")
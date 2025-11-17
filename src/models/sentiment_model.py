"""
Financial Sentiment Analysis Module
Analyzes sentiment in financial texts using FinBERT
"""

from typing import List, Dict, Optional, Tuple
from collections import Counter
import numpy as np

from model_loader import ModelLoader


class FinancialSentiment:
    """
    Financial sentiment analyzer using FinBERT
    Trained specifically on financial texts
    """
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        print("Initializing Financial Sentiment Analyzer...")
        
        # Load FinBERT model
        self.loader = ModelLoader()
        self.model_data = self.loader.load_model('sentiment')
        self.pipeline = self.model_data['pipeline']
        
        # Sentiment labels mapping
        self.label_mapping = {
            'positive': 'POSITIVE',
            'negative': 'NEGATIVE',
            'neutral': 'NEUTRAL'
        }
        
        # Keywords for sentiment boosting
        self.positive_keywords = [
            'profit', 'growth', 'increase', 'strong', 'exceed', 'outperform',
            'success', 'gain', 'rise', 'improve', 'innovation', 'expansion',
            'beat', 'surge', 'soar', 'record', 'milestone'
        ]
        
        self.negative_keywords = [
            'loss', 'decline', 'decrease', 'weak', 'miss', 'underperform',
            'fail', 'drop', 'fall', 'worsen', 'challenge', 'risk',
            'concern', 'plunge', 'slump', 'crisis', 'warning'
        ]
        
        print("✓ Financial Sentiment Analyzer initialized")
    
    def analyze(self, text: str, return_all_scores: bool = False) -> Dict:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            return_all_scores: Return scores for all labels
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not isinstance(text, str):
            return {
                'label': 'NEUTRAL',
                'score': 0.0,
                'error': 'Invalid input text'
            }
        
        try:
            # Run sentiment analysis
            result = self.pipeline(text[:512])[0]  # FinBERT max length is 512
            
            # Map label
            label = result['label'].upper()
            if label not in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                label = self.label_mapping.get(result['label'].lower(), 'NEUTRAL')
            
            output = {
                'label': label,
                'score': float(result['score']),
                'confidence': 'HIGH' if result['score'] > 0.8 else 'MEDIUM' if result['score'] > 0.6 else 'LOW'
            }
            
            # Add keyword analysis
            keyword_sentiment = self._analyze_keywords(text)
            output['keyword_analysis'] = keyword_sentiment
            
            # Return all scores if requested
            if return_all_scores:
                output['all_scores'] = self._get_all_scores(text)
            
            return output
            
        except Exception as e:
            return {
                'label': 'NEUTRAL',
                'score': 0.0,
                'error': str(e)
            }
    
    def _get_all_scores(self, text: str) -> Dict:
        """Get probability scores for all sentiment labels"""
        try:
            import torch
            from torch.nn.functional import softmax
            
            # Tokenize
            tokenizer = self.model_data['tokenizer']
            model = self.model_data['model']
            
            inputs = tokenizer(text[:512], return_tensors="pt", 
                             truncation=True, padding=True)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                probs = softmax(outputs.logits, dim=-1)[0]
            
            # Map to labels (FinBERT uses: 0=positive, 1=negative, 2=neutral)
            return {
                'POSITIVE': float(probs[0]),
                'NEGATIVE': float(probs[1]),
                'NEUTRAL': float(probs[2])
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_keywords(self, text: str) -> Dict:
        """Analyze sentiment based on financial keywords"""
        text_lower = text.lower()
        
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return {
                'positive_keywords': 0,
                'negative_keywords': 0,
                'keyword_sentiment': 'NEUTRAL'
            }
        
        positive_ratio = positive_count / total
        
        if positive_ratio > 0.6:
            keyword_sentiment = 'POSITIVE'
        elif positive_ratio < 0.4:
            keyword_sentiment = 'NEGATIVE'
        else:
            keyword_sentiment = 'NEUTRAL'
        
        return {
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'keyword_sentiment': keyword_sentiment
        }
    
    def analyze_sentences(self, text: str) -> List[Dict]:
        """
        Analyze sentiment of each sentence
        
        Args:
            text: Input text
            
        Returns:
            List of sentence sentiment results
        """
        from nltk.tokenize import sent_tokenize
        
        sentences = sent_tokenize(text)
        results = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 10:  # Skip very short sentences
                sentiment = self.analyze(sentence)
                results.append({
                    'sentence_id': i,
                    'sentence': sentence,
                    'sentiment': sentiment['label'],
                    'score': sentiment['score']
                })
        
        return results
    
    def get_overall_sentiment(self, text: str, 
                             method: str = 'weighted') -> Dict:
        """
        Get overall sentiment of document
        
        Args:
            text: Input text
            method: 'weighted' or 'majority'
            
        Returns:
            Overall sentiment analysis
        """
        sentence_sentiments = self.analyze_sentences(text)
        
        if not sentence_sentiments:
            return {
                'overall_sentiment': 'NEUTRAL',
                'confidence': 0.0
            }
        
        if method == 'weighted':
            # Weighted average by confidence scores
            sentiment_scores = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
            
            for sent in sentence_sentiments:
                label = sent['sentiment']
                score = sent['score']
                sentiment_scores[label] += score
            
            # Normalize
            total = sum(sentiment_scores.values())
            if total > 0:
                sentiment_scores = {k: v/total for k, v in sentiment_scores.items()}
            
            overall = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[overall]
            
        else:  # majority vote
            sentiments = [s['sentiment'] for s in sentence_sentiments]
            sentiment_counts = Counter(sentiments)
            overall = sentiment_counts.most_common(1)[0][0]
            confidence = sentiment_counts[overall] / len(sentiments)
        
        return {
            'overall_sentiment': overall,
            'confidence': confidence,
            'sentence_breakdown': {
                'POSITIVE': sum(1 for s in sentence_sentiments if s['sentiment'] == 'POSITIVE'),
                'NEGATIVE': sum(1 for s in sentence_sentiments if s['sentiment'] == 'NEGATIVE'),
                'NEUTRAL': sum(1 for s in sentence_sentiments if s['sentiment'] == 'NEUTRAL')
            },
            'total_sentences': len(sentence_sentiments)
        }
    
    def compare_sentiments(self, texts: List[str]) -> Dict:
        """
        Compare sentiment across multiple texts
        
        Args:
            texts: List of texts to compare
            
        Returns:
            Comparison results
        """
        results = []
        
        for i, text in enumerate(texts):
            sentiment = self.get_overall_sentiment(text)
            results.append({
                'text_id': i,
                'sentiment': sentiment['overall_sentiment'],
                'confidence': sentiment['confidence'],
                'preview': text[:100] + "..." if len(text) > 100 else text
            })
        
        # Calculate aggregate statistics
        sentiments = [r['sentiment'] for r in results]
        sentiment_dist = Counter(sentiments)
        
        return {
            'individual_results': results,
            'aggregate': {
                'most_common': sentiment_dist.most_common(1)[0][0],
                'distribution': dict(sentiment_dist),
                'avg_confidence': np.mean([r['confidence'] for r in results])
            }
        }
    
    def extract_sentiment_phrases(self, text: str, 
                                  sentiment_type: str = 'positive') -> List[Dict]:
        """
        Extract phrases with specific sentiment
        
        Args:
            text: Input text
            sentiment_type: 'positive', 'negative', or 'neutral'
            
        Returns:
            List of phrases with that sentiment
        """
        sentence_sentiments = self.analyze_sentences(text)
        
        matching_phrases = []
        for sent_data in sentence_sentiments:
            if sent_data['sentiment'].lower() == sentiment_type.lower():
                matching_phrases.append({
                    'text': sent_data['sentence'],
                    'score': sent_data['score']
                })
        
        # Sort by score
        matching_phrases.sort(key=lambda x: x['score'], reverse=True)
        
        return matching_phrases
    
    def sentiment_timeline(self, texts: List[str], 
                          labels: Optional[List[str]] = None) -> Dict:
        """
        Analyze sentiment over time (for sequential documents)
        
        Args:
            texts: List of texts in chronological order
            labels: Optional labels for each text (e.g., dates)
            
        Returns:
            Timeline data
        """
        if labels is None:
            labels = [f"Doc {i+1}" for i in range(len(texts))]
        
        timeline = []
        for i, text in enumerate(texts):
            sentiment = self.get_overall_sentiment(text)
            timeline.append({
                'label': labels[i],
                'sentiment': sentiment['overall_sentiment'],
                'confidence': sentiment['confidence'],
                'positive_ratio': sentiment['sentence_breakdown']['POSITIVE'] / sentiment['total_sentences']
            })
        
        return {
            'timeline': timeline,
            'trend': self._calculate_trend(timeline)
        }
    
    def _calculate_trend(self, timeline: List[Dict]) -> str:
        """Calculate overall sentiment trend"""
        if len(timeline) < 2:
            return "INSUFFICIENT_DATA"
        
        # Map sentiment to numeric values
        sentiment_values = {
            'POSITIVE': 1,
            'NEUTRAL': 0,
            'NEGATIVE': -1
        }
        
        values = [sentiment_values[t['sentiment']] for t in timeline]
        
        # Simple trend calculation
        if values[-1] > values[0]:
            return "IMPROVING"
        elif values[-1] < values[0]:
            return "DECLINING"
        else:
            return "STABLE"


# Example usage
if __name__ == "__main__":
    # Initialize sentiment analyzer
    sentiment = FinancialSentiment()
    
    # Test texts
    positive_text = "The company reported record profits with revenue exceeding expectations by 20%."
    negative_text = "Sales declined sharply due to increased competition and rising costs."
    neutral_text = "The company maintained operations during the quarter."
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS EXAMPLES")
    print("="*60)
    
    for label, text in [("Positive", positive_text), 
                        ("Negative", negative_text), 
                        ("Neutral", neutral_text)]:
        result = sentiment.analyze(text)
        print(f"\n{label} Example:")
        print(f"  Text: {text}")
        print(f"  Sentiment: {result['label']} (score: {result['score']:.3f})")
        print(f"  Confidence: {result['confidence']}")
    
    print("\n✓ Sentiment Analysis Module Ready!")
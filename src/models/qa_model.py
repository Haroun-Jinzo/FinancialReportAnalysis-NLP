"""
Question Answering Module
Answer questions about financial documents using RoBERTa
"""

from typing import List, Dict, Optional, Tuple
import re

from model_loader import ModelLoader


class FinancialQA:
    """
    Question Answering system for financial documents
    Uses RoBERTa trained on SQuAD 2.0
    """
    
    def __init__(self):
        """Initialize QA system"""
        print("Initializing Financial Question Answering System...")
        
        # Load QA model
        self.loader = ModelLoader()
        self.model_data = self.loader.load_model('qa')
        self.pipeline = self.model_data['pipeline']
        
        # Common financial question templates
        self.question_templates = {
            'revenue': [
                "What was the revenue?",
                "What was the total revenue?",
                "How much revenue did the company generate?"
            ],
            'profit': [
                "What was the profit?",
                "What was the net income?",
                "How much profit did the company make?"
            ],
            'eps': [
                "What was the EPS?",
                "What was the earnings per share?",
                "What is the EPS?"
            ],
            'growth': [
                "What was the growth rate?",
                "How much did revenue grow?",
                "What is the year-over-year growth?"
            ],
            'ceo': [
                "Who is the CEO?",
                "Who is the chief executive officer?",
                "Who leads the company?"
            ]
        }
        
        print("✓ Financial QA System initialized")
    
    def answer(self, question: str, context: str,
              confidence_threshold: float = 0.5) -> Dict:
        """
        Answer a question based on context
        
        Args:
            question: Question to answer
            context: Context text containing the answer
            confidence_threshold: Minimum confidence for valid answer
            
        Returns:
            Dictionary with answer and metadata
        """
        if not question or not context:
            return {
                'question': question,
                'answer': None,
                'score': 0.0,
                'error': 'Question and context are required'
            }
        
        try:
            # Truncate context if too long (RoBERTa limit)
            max_context_length = 3000
            if len(context) > max_context_length:
                context = self._smart_truncate(context, question, max_context_length)
            
            # Get answer from model
            result = self.pipeline(
                question=question,
                context=context
            )
            
            answer = result['answer']
            score = result['score']
            
            # Post-process answer
            answer = self._clean_answer(answer)
            
            # Check if answer is valid
            if score < confidence_threshold:
                return {
                    'question': question,
                    'answer': answer,
                    'score': score,
                    'confidence': 'LOW',
                    'warning': 'Low confidence answer'
                }
            
            # Get extended context around answer
            extended_answer = self._get_extended_context(
                context, 
                answer,
                result['start'],
                result['end']
            )
            
            return {
                'question': question,
                'answer': answer,
                'score': score,
                'confidence': 'HIGH' if score > 0.8 else 'MEDIUM',
                'start': result['start'],
                'end': result['end'],
                'context': extended_answer
            }
            
        except Exception as e:
            return {
                'question': question,
                'answer': None,
                'score': 0.0,
                'error': str(e)
            }
    
    def _smart_truncate(self, context: str, question: str, 
                       max_length: int) -> str:
        """
        Intelligently truncate context to keep relevant parts
        """
        # Extract keywords from question
        keywords = self._extract_keywords(question)
        
        # Split context into sentences
        sentences = context.split('.')
        
        # Score sentences by keyword relevance
        scored_sentences = []
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            score = sum(1 for kw in keywords if kw in sent_lower)
            scored_sentences.append((score, i, sent))
        
        # Sort by relevance
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        # Take top relevant sentences up to max_length
        selected = []
        current_length = 0
        
        for score, idx, sent in scored_sentences:
            if current_length + len(sent) <= max_length:
                selected.append((idx, sent))
                current_length += len(sent)
            else:
                break
        
        # Sort back to original order
        selected.sort(key=lambda x: x[0])
        
        return '. '.join([sent for _, sent in selected])
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Remove common question words
        stopwords = {'what', 'when', 'where', 'who', 'how', 'why', 'is', 'was', 
                    'the', 'a', 'an', 'did', 'do', 'does'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and format answer"""
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        # Capitalize first letter if it's a proper answer
        if answer and len(answer) > 1:
            if not answer[0].isupper() and not answer[0].isdigit():
                answer = answer[0].upper() + answer[1:]
        
        return answer.strip()
    
    def _get_extended_context(self, full_context: str, answer: str,
                             start: int, end: int, window: int = 100) -> str:
        """Get extended context around answer"""
        context_start = max(0, start - window)
        context_end = min(len(full_context), end + window)
        
        extended = full_context[context_start:context_end]
        
        # Add ellipsis if truncated
        if context_start > 0:
            extended = "..." + extended
        if context_end < len(full_context):
            extended = extended + "..."
        
        return extended.strip()
    
    def answer_multiple(self, questions: List[str], 
                       context: str) -> List[Dict]:
        """
        Answer multiple questions about the same context
        
        Args:
            questions: List of questions
            context: Context text
            
        Returns:
            List of answer dictionaries
        """
        results = []
        
        for question in questions:
            answer = self.answer(question, context)
            results.append(answer)
        
        return results
    
    def extract_key_facts(self, context: str, 
                         fact_types: Optional[List[str]] = None) -> Dict:
        """
        Extract key facts using predefined question templates
        
        Args:
            context: Context text
            fact_types: Types of facts to extract (revenue, profit, etc.)
            
        Returns:
            Dictionary of extracted facts
        """
        if fact_types is None:
            fact_types = ['revenue', 'profit', 'eps', 'growth']
        
        facts = {}
        
        for fact_type in fact_types:
            if fact_type in self.question_templates:
                # Try multiple question phrasings
                best_answer = None
                best_score = 0
                
                for question in self.question_templates[fact_type]:
                    result = self.answer(question, context)
                    
                    if result['score'] > best_score:
                        best_score = result['score']
                        best_answer = result['answer']
                
                if best_answer and best_score > 0.3:
                    facts[fact_type] = {
                        'value': best_answer,
                        'confidence': best_score
                    }
        
        return facts
    
    def find_answer_span(self, context: str, answer: str) -> List[Tuple[int, int]]:
        """
        Find all occurrences of answer in context
        
        Returns:
            List of (start, end) positions
        """
        spans = []
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        start = 0
        while True:
            pos = context_lower.find(answer_lower, start)
            if pos == -1:
                break
            spans.append((pos, pos + len(answer)))
            start = pos + 1
        
        return spans
    
    def interactive_qa(self, context: str, max_questions: int = 10):
        """
        Interactive QA session (for testing)
        
        Args:
            context: Context text
            max_questions: Maximum number of questions
        """
        print("\n" + "="*60)
        print("INTERACTIVE QA SESSION")
        print("="*60)
        print("\nContext preview:")
        print(context[:300] + "..." if len(context) > 300 else context)
        print("\n" + "="*60)
        print("Ask questions about the text (type 'quit' to exit)")
        print("="*60 + "\n")
        
        question_count = 0
        
        while question_count < max_questions:
            question = input(f"\nQuestion {question_count + 1}: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            result = self.answer(question, context)
            
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['confidence']} (score: {result['score']:.3f})")
            
            if 'context' in result:
                print(f"Context: {result['context']}")
            
            question_count += 1
        
        print(f"\n✓ QA session ended ({question_count} questions answered)")
    
    def generate_faq(self, context: str, 
                    num_questions: int = 5) -> List[Dict]:
        """
        Generate FAQ from document using common questions
        
        Args:
            context: Context text
            num_questions: Number of FAQ items to generate
            
        Returns:
            List of Q&A pairs
        """
        faq = []
        
        # Collect all template questions
        all_questions = []
        for questions in self.question_templates.values():
            all_questions.extend(questions)
        
        # Answer each question and keep high-confidence ones
        for question in all_questions:
            if len(faq) >= num_questions:
                break
            
            result = self.answer(question, context)
            
            if result['score'] > 0.6 and result['answer']:
                faq.append({
                    'question': question,
                    'answer': result['answer'],
                    'confidence': result['score']
                })
        
        # Sort by confidence
        faq.sort(key=lambda x: x['confidence'], reverse=True)
        
        return faq[:num_questions]


# Example usage
if __name__ == "__main__":
    # Initialize QA system
    qa = FinancialQA()
    
    # Sample context
    sample_context = """
    Apple Inc. reported strong financial results for Q3 2024. 
    The company generated revenue of $90.1 billion, representing 
    a 15.2% increase year-over-year. Net income reached $25.5 billion,
    with earnings per share (EPS) of $1.85, beating analyst expectations
    of $1.52. CEO Tim Cook attributed the strong performance to robust
    iPhone sales, which grew 12% to $45.8 billion. The company also
    announced a quarterly dividend of $0.24 per share.
    """
    
    print("\nSample Context:")
    print(sample_context)
    print("\n" + "="*60)
    
    # Test questions
    questions = [
        "What was the revenue?",
        "Who is the CEO?",
        "What was the EPS?",
        "How much did iPhone sales grow?"
    ]
    
    print("\nTesting Questions:")
    for q in questions:
        result = qa.answer(q, sample_context)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']} (confidence: {result['score']:.3f})")
    
    # Extract key facts
    print("\n" + "="*60)
    print("Key Facts Extraction:")
    facts = qa.extract_key_facts(sample_context)
    for fact_type, data in facts.items():
        print(f"  {fact_type}: {data['value']} (confidence: {data['confidence']:.3f})")
    
    print("\n✓ Question Answering Module Ready!")
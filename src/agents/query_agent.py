import sys
from pathlib import Path
from datetime import datetime
import re
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import required modules
from models.qa_model import FinancialQA
from extraction.metric_extractor import MetricExtractor

logger = logging.getLogger(__name__)


class QueryAgent:
    """   
    examples of question to ask:
    - Simple questions: "What was the revenue?"
    - Comparisons: "Compare Apple and Microsoft"
    - Trends: "Show me the revenue trend"
    - Searches: "Find documents about earnings"
    """
    
    def __init__(self, document_agent=None, analysis_agent=None):
        logger.info("💬 Starting Query Agent...")
        
        # Initialize Q&A and metric extraction
        self.qa = FinancialQA()
        self.metric_extractor = MetricExtractor()
        
        # Store references to other agents
        self.document_agent = document_agent
        self.analysis_agent = analysis_agent
        
        # Keep track of past queries
        self.history = []
        
        logger.info("✓ Query Agent ready")
    
    # ==================== MAIN QUERY METHOD ====================
    
    def query(self, query_text: str, context_doc_ids: list = None) -> dict:
        """
        Main method - answers ANY question about your documents
        
        This is the only method you need! Just ask your question in plain English
        and the agent figures out how to answer it.
        
        Args:
            query_text: Your question in plain English
                       Examples:
                       - "What was the revenue?"
                       - "Compare Apple and Microsoft"
                       - "Show me the profit trend"
                       - "Is the company doing well?"
            
            context_doc_ids: (Optional) Limit search to specific documents
                            If not provided, searches all documents
        
        Returns:
            Dictionary with the answer:
            {
                'answer': 'Revenue was $90 billion',
                'confidence': 0.95,      # 0-1 (how confident)
                'type': 'metric',        # Type of query
                'query_id': 'query_123'  # For tracking
            }
        
        Examples:
            # Simple question
            result = agent.query("What was the revenue?")
            print(result['answer'])
            # → "Revenue: Q3 2024: $90B"
            
            # Comparison
            result = agent.query("Compare Apple and Microsoft")
            print(result['answer'])
            # → "Comparison: Apple vs Microsoft..."
            
            # Trend
            result = agent.query("Show profit trend")
            print(result['answer'])
            # → "Profit Trend: Direction: INCREASING..."
        """
        logger.info(f"❓ Query: {query_text}")
        
        # Step 1: Figure out what type of question this is
        query_type = self._classify_query(query_text)
        logger.info(f"  Detected type: {query_type}")
        
        # Step 2: Route to the right handler based on type
        if query_type == "metric":
            # Questions about specific numbers
            result = self._handle_metric_query(query_text, context_doc_ids)
            
        elif query_type == "comparison":
            # Questions comparing two things
            result = self._handle_comparison(query_text, context_doc_ids)
            
        elif query_type == "trend":
            # Questions about changes over time
            result = self._handle_trend_query(query_text, context_doc_ids)
            
        else:  # "question"
            # General questions
            result = self._handle_question(query_text, context_doc_ids)
        
        # Step 3: Add metadata
        result['query_type'] = query_type
        result['query_id'] = f"query_{len(self.history) + 1}"
        
        # Step 4: Save to history
        self.history.append({
            'query_id': result['query_id'],
            'query': query_text,
            'type': query_type,
            'answer': result['answer'],
            'timestamp': datetime.now()
        })
        
        logger.info(f"✓ Answered (confidence: {result.get('confidence', 0):.2f})")
        
        return result
    
    def _handle_question(self, question: str, doc_ids: list) -> dict:
        """
        Handle general questions like:
        - "Who is the CEO?"
        - "What products were mentioned?"
        - "When was the report published?"
        
        How it works:
        1. Get the documents
        2. Search each document for the answer
        3. Return the best answer found
        """
        # Check if we have document agent
        if not self.document_agent:
            return {
                'answer': 'Cannot answer: Document agent not available',
                'confidence': 0.0
            }
        
        # Get documents to search
        if doc_ids:
            # Use specific documents
            docs = [self.document_agent.get_document(id) for id in doc_ids]
            docs = [d for d in docs if d]
        else:
            # Search all completed documents
            docs = self.document_agent.get_all_documents(status="completed")
        
        if not docs:
            return {
                'answer': 'No documents available to search',
                'confidence': 0.0
            }
        
        logger.info(f"  Searching {len(docs)} documents...")
        
        # Try to find answer in each document
        best_answer = None
        best_score = 0
        source_doc = None
        
        for doc in docs:
            # Use Q&A model to find answer
            result = self.qa.answer(question, doc.text)
            
            # Keep track of best answer
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = result['answer']
                source_doc = doc.id
        
        # Return answer if confident enough
        if best_answer and best_score > 0.3:
            return {
                'answer': best_answer,
                'confidence': best_score,
                'source': source_doc
            }
        else:
            return {
                'answer': "I couldn't find a confident answer to this question in the documents.",
                'confidence': 0.0
            }
    
    def _handle_metric_query(self, query: str, doc_ids: list) -> dict:
        """
        Handle metric queries like:
        - "What was the revenue?"
        - "Show me the EPS"
        - "What's the profit margin?"
        
        How it works:
        1. Figure out which metric they want (revenue, profit, etc)
        2. Extract that metric from documents
        3. Format and return the results
        """
        # Check if we have document agent
        if not self.document_agent:
            return {
                'answer': 'Cannot answer: Document agent not available',
                'confidence': 0.0
            }
        
        # Figure out which metric they're asking about
        metric_name = self._extract_metric_name(query)
        
        if not metric_name:
            return {
                'answer': 'Could not identify which metric you are asking about',
                'confidence': 0.0
            }
        
        logger.info(f"  Extracting metric: {metric_name}")
        
        # Get documents
        if doc_ids:
            docs = [self.document_agent.get_document(id) for id in doc_ids]
            docs = [d for d in docs if d]
        else:
            docs = self.document_agent.get_all_documents(status="completed")
        
        if not docs:
            return {
                'answer': 'No documents available',
                'confidence': 0.0
            }
        
        # Extract metric from each document
        results = []
        for doc in docs[:5]:  # Limit to first 5 documents
            metric = self.metric_extractor.extract_metric_by_name(doc.text, metric_name)
            
            if metric:
                period = doc.metadata.get('period', 'Unknown')
                results.append(f"{period}: {metric.value} {metric.unit}")
        
        # Format answer
        if results:
            metric_title = metric_name.replace('_', ' ').title()
            answer = f"{metric_title}:\n" + "\n".join(results)
            
            return {
                'answer': answer,
                'confidence': 0.9
            }
        else:
            return {
                'answer': f"Could not find {metric_name.replace('_', ' ')} in the documents",
                'confidence': 0.0
            }
    
    def _handle_comparison(self, query: str, doc_ids: list) -> dict:
        """
        Handle comparison queries like:
        - "Compare Apple and Microsoft"
        - "Which company did better?"
        - "What's the difference between Q1 and Q2?"
        
        How it works:
        1. Get two documents to compare
        2. Run comparison analysis
        3. Format the results
        """
        # Check if we have both agents
        if not self.analysis_agent or not self.document_agent:
            return {
                'answer': 'Cannot compare: Required agents not available',
                'confidence': 0.0
            }
        
        # Get documents to compare
        if doc_ids and len(doc_ids) >= 2:
            # Use provided documents
            doc1 = self.document_agent.get_document(doc_ids[0])
            doc2 = self.document_agent.get_document(doc_ids[1])
        else:
            # Try to find documents automatically
            docs = self.document_agent.get_all_documents(status="completed")
            
            if len(docs) >= 2:
                doc1, doc2 = docs[0], docs[1]
            else:
                return {
                    'answer': 'Need at least 2 documents to compare',
                    'confidence': 0.0
                }
        
        if not doc1 or not doc2:
            return {
                'answer': 'Could not find documents to compare',
                'confidence': 0.0
            }
        
        logger.info(f"  Comparing documents: {doc1.id} vs {doc2.id}")
        
        # Prepare documents for comparison
        doc1_data = {
            'name': doc1.metadata.get('company', doc1.filename),
            'text': doc1.text,
            'period': doc1.metadata.get('period')
        }
        
        doc2_data = {
            'name': doc2.metadata.get('company', doc2.filename),
            'text': doc2.text,
            'period': doc2.metadata.get('period')
        }
        
        # Run comparison
        comparison = self.analysis_agent.compare_documents(doc1_data, doc2_data)
        
        # Format answer
        answer = f"Comparison: {doc1_data['name']} vs {doc2_data['name']}\n"
        
        if 'metrics' in comparison and 'summary' in comparison['metrics']:
            summary = comparison['metrics']['summary']
            answer += f"Winner: {summary.get('overall_winner', 'N/A')}\n"
            answer += f"Metrics compared: {summary.get('total_comparisons', 0)}"
        
        return {
            'answer': answer,
            'confidence': 0.8,
            'details': comparison
        }
    
    def _handle_trend_query(self, query: str, doc_ids: list) -> dict:
        """
        Handle trend queries like:
        - "Show me the revenue trend"
        - "How is profit changing?"
        - "What's the trend in earnings?"
        
        How it works:
        1. Get multiple documents (need 2+ for trends)
        2. Extract the metric from each
        3. Run trend analysis
        4. Format the results
        """
        # Check if we have both agents
        if not self.analysis_agent or not self.document_agent:
            return {
                'answer': 'Cannot analyze trends: Required agents not available',
                'confidence': 0.0
            }
        
        # Get documents
        if doc_ids:
            docs = [self.document_agent.get_document(id) for id in doc_ids]
            docs = [d for d in docs if d]
        else:
            docs = self.document_agent.get_all_documents(status="completed")
        
        if len(docs) < 2:
            return {
                'answer': 'Need at least 2 documents for trend analysis',
                'confidence': 0.0
            }
        
        logger.info(f"  Analyzing trend across {len(docs)} documents")
        
        # Prepare documents
        trend_docs = [
            {
                'period': doc.metadata.get('period', f'Period {i+1}'),
                'text': doc.text
            }
            for i, doc in enumerate(docs)
        ]
        
        # Figure out which metric they want
        metric_name = self._extract_metric_name(query) or 'revenue'
        
        logger.info(f"  Metric: {metric_name}")
        
        # Run trend analysis
        trend_result = self.analysis_agent.analyze_trends(trend_docs, [metric_name])
        
        # Format answer
        if trend_result['trends']:
            trend_data = trend_result['trends'][metric_name]
            
            metric_title = metric_name.replace('_', ' ').title()
            answer = f"{metric_title} Trend:\n"
            answer += f"Direction: {trend_data['direction']}\n"
            answer += f"Average: {trend_data['average']:.2f}\n"
            
            if trend_data.get('forecast'):
                answer += f"Forecast: {trend_data['forecast']:.2f}"
        else:
            answer = f"Could not analyze {metric_name} trend"
        
        return {
            'answer': answer,
            'confidence': 0.8
        }
    
    def _classify_query(self, query: str) -> str:
        """
        Figure out what type of question this is
        
        Types:
        - question: General question (Who, What, When, Where, Why, How)
        - metric: Asking for a specific number or metric
        - comparison: Comparing two things
        - trend: Asking about changes over time
        
        How it works:
        1. Look for keyword patterns
        2. Check for specific words
        3. Return the most likely type
        """
        query_lower = query.lower()
        
        # Check for comparison keywords
        comparison_words = ['compare', 'versus', 'vs', 'vs.', 'difference', 'better', 'worse']
        if any(word in query_lower for word in comparison_words):
            return 'comparison'
        
        # Check for trend keywords
        trend_words = ['trend', 'over time', 'change', 'growth', 'historical', 'evolution']
        if any(word in query_lower for word in trend_words):
            return 'trend'
        
        # Check for metric keywords
        metric_words = ['revenue', 'profit', 'income', 'eps', 'margin', 'ebitda', 'sales']
        if any(word in query_lower for word in metric_words):
            return 'metric'
        
        # Default to general question
        return 'question'
    
    def _extract_metric_name(self, query: str) -> str:
        """
        Extract the metric name from a query
        
        Examples:
        - "What was the revenue?" → "revenue"
        - "Show me the EPS" → "eps"
        - "What's the profit margin?" → "gross_margin"
        """
        query_lower = query.lower()
        
        # Map common terms to metric names
        metric_map = {
            'revenue': 'revenue',
            'sales': 'revenue',
            'profit': 'net_income',
            'income': 'net_income',
            'earnings': 'net_income',
            'eps': 'eps',
            'earnings per share': 'eps',
            'margin': 'gross_margin',
            'gross margin': 'gross_margin',
            'operating margin': 'operating_margin',
            'net margin': 'net_margin',
            'ebitda': 'ebitda'
        }
        
        # Check each keyword
        for keyword, metric in metric_map.items():
            if keyword in query_lower:
                return metric
        
        return None
    
    def get_history(self, limit: int = 10) -> list:
        """
        Get recent query history
        
        Args:
            limit: Number of recent queries to return
        
        Returns:
            List of recent queries
        
        Example:
            history = agent.get_history(5)
            for query in history:
                print(f"{query['query']} → {query['answer']}")
        """
        return self.history[-limit:]
    
    def clear_history(self):
        """
        Clear the query history
        
        Example:
            agent.clear_history()
        """
        self.history.clear()
        logger.info("Query history cleared")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    print("\n" + "="*70)
    print("QUERY AGENT - DEMO")
    print("="*70)
    
    # Create agent (without other agents for demo)
    agent = QueryAgent()
    
    # Example queries to test classification
    test_queries = [
        "What was the revenue?",
        "Compare Apple and Microsoft",
        "Show me the profit trend",
        "Who is the CEO?",
        "What's the EPS?",
        "How did earnings change over time?"
    ]
    
    print("\n🔍 Testing Query Classification:\n")
    
    for query in test_queries:
        query_type = agent._classify_query(query)
        metric = agent._extract_metric_name(query)
        
        print(f"Query: {query}")
        print(f"  Type: {query_type}")
        if metric:
            print(f"  Metric: {metric}")
        print()
    
    print("="*70)
    print("\n✓ Demo complete!")
    print("\nTo use with real documents, initialize with agents:")
    print("  agent = QueryAgent(document_agent, analysis_agent)")
    print("  result = agent.query('What was the revenue?')")
    print("\n" + "="*70 + "\n")
import sys
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent))

from agents.document_agent import DocumentAgent
from agents.analysis_agent import AnalysisAgent
from agents.query_agent import QueryAgent

logger = logging.getLogger(__name__)


class FinancialAnalysisOrchestrator:
    """
    Master coordinator - runs the complete analysis pipeline
    
    Pipeline Steps:
    1. Parse document (PDF/Word/Excel)
    2. Extract text
    3. Process and clean text
    4. Extract entities and keywords
    5. Analyze (trends, risks, sentiment)
    6. Generate insights
    
    Simple Usage:
        orchestrator = FinancialAnalysisOrchestrator()
        results = orchestrator.process_document("report.pdf")
    """
    
    def __init__(self):
        """Initialize orchestrator and all agents"""
        logger.info("="*70)
        logger.info("🚀 INITIALIZING FINANCIAL ANALYSIS ORCHESTRATOR")
        logger.info("="*70)
        
        # Initialize all agents
        logger.info("\n[1/3] Starting Document Agent...")
        self.document_agent = DocumentAgent(storage_path="data/documents")
        
        logger.info("\n[2/3] Starting Analysis Agent...")
        self.analysis_agent = AnalysisAgent()
        
        logger.info("\n[3/3] Starting Query Agent...")
        self.query_agent = QueryAgent(
            document_agent=self.document_agent,
            analysis_agent=self.analysis_agent
        )
        
        logger.info("\n" + "="*70)
        logger.info("✓ ORCHESTRATOR READY")
        logger.info("="*70)
    
    def process_document(self, file_path: str, period: str = None) -> dict:
        """
        Complete pipeline: Upload → Process → Analyze
        
        This is the main method - it does everything!
        
        Args:
            file_path: Path to document
            period: Time period (e.g., "Q3 2024")
        
        Returns:
            Complete analysis results
        
        Example:
            results = orchestrator.process_document(
                "reports/q3_2024.pdf",
                period="Q3 2024"
            )
        """
        start_time = datetime.now()
        
        logger.info("\n" + "="*70)
        logger.info(f"📄 PROCESSING: {Path(file_path).name}")
        logger.info("="*70)
        
        try:
            results = {}
            
            # STEP 1: Upload Document
            logger.info("\n[Step 1/4] Uploading document...")
            metadata = {'period': period} if period else {}
            document = self.document_agent.ingest_document(file_path, metadata)
            logger.info(f"  ✓ Document ID: {document.id}")
            
            # STEP 2: Process Document (extract text, entities, keywords)
            logger.info("\n[Step 2/4] Processing document...")
            processed = self.document_agent.process_document(document.id)
            logger.info(f"  ✓ Extracted {len(processed.entities)} entities")
            logger.info(f"  ✓ Found {len(processed.keywords)} keywords")
            
            results['document'] = {
                'id': processed.id,
                'filename': processed.filename,
                'status': processed.status,
                'entities': processed.entities[:10],  # Top 10
                'keywords': processed.keywords[:10]   # Top 10
            }
            
            # STEP 3: Analyze Document
            logger.info("\n[Step 3/4] Analyzing document...")
            
            doc_data = {
                'period': period or 'Unknown',
                'text': processed.text
            }
            
            # Run analysis
            analysis = self.analysis_agent.comprehensive_analysis(
                documents=[doc_data],
                include_risks=True,
                include_insights=True
            )
            
            results['analysis'] = {
                'sentiment': analysis.get('sentiment', {}).get('by_period', [{}])[0],
                'risks': analysis.get('risks', {}),
                'insights': analysis.get('insights', {})
            }
            
            logger.info(f"  ✓ Sentiment: {results['analysis']['sentiment'].get('sentiment', 'N/A')}")
            logger.info(f"  ✓ Risk Level: {results['analysis']['risks'].get('risk_level', 'N/A')}")
            logger.info(f"  ✓ Insights: {results['analysis']['insights'].get('total', 0)}")
            
            # STEP 4: Generate Summary
            logger.info("\n[Step 4/4] Generating summary...")
            results['summary'] = self._generate_summary(results)
            
            # Calculate processing time
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info("\n" + "="*70)
            logger.info(f"✓ PROCESSING COMPLETE ({duration:.1f}s)")
            logger.info("="*70)
            
            return results
            
        except Exception as e:
            logger.error(f"\n✗ PROCESSING FAILED: {e}")
            raise
    
    def process_multiple(self, file_paths: list, periods: list = None) -> dict:
        """
        Process multiple documents and analyze trends
        
        Args:
            file_paths: List of document paths
            periods: Optional list of periods for each document
        
        Returns:
            Multi-document analysis with trends
        
        Example:
            results = orchestrator.process_multiple(
                ["q1.pdf", "q2.pdf", "q3.pdf"],
                ["Q1 2024", "Q2 2024", "Q3 2024"]
            )
        """
        logger.info(f"\n📊 PROCESSING {len(file_paths)} DOCUMENTS")
        
        # Process each document
        documents = []
        for i, file_path in enumerate(file_paths):
            period = periods[i] if periods and i < len(periods) else None
            
            try:
                # Upload and process
                metadata = {'period': period} if period else {}
                doc = self.document_agent.ingest_document(file_path, metadata)
                processed = self.document_agent.process_document(doc.id)
                
                documents.append({
                    'id': processed.id,
                    'period': period or f'Doc {i+1}',
                    'text': processed.text
                })
                
                logger.info(f"  ✓ Processed: {Path(file_path).name}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {Path(file_path).name} - {e}")
        
        if not documents:
            return {'error': 'No documents processed successfully'}
        
        # Analyze all documents together
        logger.info("\n📈 Analyzing trends...")
        
        analysis = self.analysis_agent.comprehensive_analysis(
            documents=documents,
            include_risks=True,
            include_insights=True
        )
        
        return {
            'documents_processed': len(documents),
            'analysis': analysis,
            'summary': self._generate_multi_doc_summary(analysis)
        }
    
    def query_documents(self, query: str, doc_ids: list = None) -> dict:
        """
        Ask questions about processed documents
        
        Args:
            query: Natural language question
            doc_ids: Optional list of document IDs to search
        
        Returns:
            Answer to the question
        
        Example:
            answer = orchestrator.query_documents("What was the revenue?")
            print(answer['answer'])
        """
        logger.info(f"\n💬 Query: {query}")
        
        result = self.query_agent.query(query, doc_ids)
        
        logger.info(f"  ✓ Answer: {result['answer'][:100]}...")
        
        return result
    
    # ==================== HELPER METHODS ====================
    
    def _generate_summary(self, results: dict) -> dict:
        """Generate executive summary for single document"""
        summary = {
            'document': results['document']['filename'],
            'status': results['document']['status'],
            'key_findings': []
        }
        
        # Add sentiment
        sentiment = results['analysis']['sentiment']
        if sentiment:
            summary['key_findings'].append(
                f"Sentiment: {sentiment.get('sentiment', 'N/A')}"
            )
        
        # Add risk level
        risks = results['analysis']['risks']
        if risks:
            summary['key_findings'].append(
                f"Risk Level: {risks.get('risk_level', 'N/A')}"
            )
        
        # Add insight count
        insights = results['analysis']['insights']
        if insights:
            summary['key_findings'].append(
                f"Insights Generated: {insights.get('total', 0)}"
            )
        
        return summary
    
    def _generate_multi_doc_summary(self, analysis: dict) -> dict:
        """Generate summary for multiple documents"""
        summary = {
            'overall_health': 'GOOD',  # Default
            'key_findings': []
        }
        
        # Check trends
        if 'trends' in analysis:
            trends = analysis['trends'].get('summary', {})
            if trends.get('declining'):
                summary['overall_health'] = 'CAUTION'
                summary['key_findings'].append(
                    f"Declining metrics: {', '.join(trends['declining'])}"
                )
            if trends.get('improving'):
                summary['key_findings'].append(
                    f"Improving metrics: {', '.join(trends['improving'])}"
                )
        
        # Check risks
        if 'risks' in analysis:
            risk_level = analysis['risks'].get('risk_level')
            if risk_level in ['HIGH', 'CRITICAL']:
                summary['overall_health'] = 'POOR'
            summary['key_findings'].append(f"Risk Level: {risk_level}")
        
        return summary
    
    def get_statistics(self) -> dict:
        """Get statistics about processed documents"""
        stats = self.document_agent.get_statistics()
        
        return {
            'total_documents': stats['total'],
            'by_status': stats['by_status'],
            'query_history_count': len(self.query_agent.history)
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # Create orchestrator
    orchestrator = FinancialAnalysisOrchestrator()
    
    # Example 1: Process single document
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Document")
    print("="*70)
    
    # results = orchestrator.process_document(
    #     "data/raw/sample_report.pdf",
    #     period="Q3 2024"
    # )
    # print(f"\nSentiment: {results['analysis']['sentiment']['sentiment']}")
    # print(f"Risk Level: {results['analysis']['risks']['risk_level']}")
    
    # Example 2: Process multiple documents
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Documents")
    print("="*70)
    
    # results = orchestrator.process_multiple(
    #     ["q1.pdf", "q2.pdf", "q3.pdf"],
    #     ["Q1 2024", "Q2 2024", "Q3 2024"]
    # )
    # print(f"\nDocuments: {results['documents_processed']}")
    # print(f"Health: {results['summary']['overall_health']}")
    
    # Example 3: Query documents
    print("\n" + "="*70)
    print("EXAMPLE 3: Query")
    print("="*70)
    
    # answer = orchestrator.query_documents("What was the revenue?")
    # print(f"\nAnswer: {answer['answer']}")
    
    # Show statistics
    stats = orchestrator.get_statistics()
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total Documents: {stats['total_documents']}")
    print(f"By Status: {stats['by_status']}")
    
    print("\n✓ Orchestrator ready for use!")
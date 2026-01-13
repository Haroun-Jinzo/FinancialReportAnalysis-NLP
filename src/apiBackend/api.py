from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import os
import logging
import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

if os.name == 'nt':  # 'nt' means Windows
    pathlib.PosixPath = pathlib.WindowsPath

from agents.document_agent import DocumentAgent
from agents.analysis_agent import AnalysisAgent
from agents.query_agent import QueryAgent

app = FastAPI(
    title="Financial NLP API",
    description="Simplified API for financial document analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Initializing agents...")
document_agent = DocumentAgent(storage_path="data/documents")
analysis_agent = AnalysisAgent()
query_agent = QueryAgent(
    document_agent=document_agent,
    analysis_agent=analysis_agent
)
logger.info("✓ Agents ready")

# Requests
class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None

class AnalysisRequest(BaseModel):
    document_ids: List[str]
    metrics: Optional[List[str]] = None


class RiskAnalysisRequest(BaseModel):
    document_id: str
    detailed: bool = False

class ComprehensiveRequest(BaseModel):
    document_ids: List[str]

class MetricsRequest(BaseModel):
    document_id: str
    metric_name: Optional[str] = None
class CompareRequest(BaseModel):
    doc_id_1: str
    doc_id_2: str
    analysis_type: Optional[str] = "comprehensive"

# ==================== ENDPOINTS ====================

@app.get("/")
def root():
    return {
        "service": "Financial NLP API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/api/health")
def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "document": "active",
            "analysis": "active",
            "query": "active"
        }
    }


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    period: Optional[str] = Form(None),
    company: Optional[str] = Form(None)
):
    try:
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process
        metadata = {'title':title, 'period': period, 'company': company, 'filename': file.filename}
        doc = document_agent.ingest_document(tmp_path, metadata)
        processed = document_agent.process_document(doc.id)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "success": True,
            "message": "Document processed",
            "document": {
                "id": processed.id,
                "filename": processed.filename,
                "status": processed.status,
                "uploaded_at": processed.uploaded_at.isoformat(),
                "processed_at": processed.processed_at.isoformat() if processed.processed_at else None,
                "entities": [{"text": e['text'], "type": e['type']} for e in processed.entities[:20]],
                "keywords": processed.keywords[:20]
            }
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
def get_documents(status: Optional[str] = None):
    try:
        docs = document_agent.get_all_documents(status=status)
        return {
            "success": True,
            "count": len(docs),
            "documents": [doc.to_dict() for doc in docs]
        }
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{document_id}")
def get_document(document_id: str):
    try:
        doc = document_agent.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"success": True, "document": doc.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def query_documents(request: QueryRequest):
    try:
        result = query_agent.query(
            query_text=request.query,
            context_doc_ids=request.document_ids
        )
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/comprehensive")
def comprehensive_analysis(request: ComprehensiveRequest):
    try:
        # Get documents
        documents = []
        for doc_id in request.document_ids:
            doc = document_agent.get_document(doc_id)
            if doc and doc.status == "completed":
                documents.append({
                    'period': doc.metadata.get('period', 'Unknown'),
                    'text': doc.text
                })
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents")
        
        results = analysis_agent.comprehensive_analysis(
            documents=documents,
            include_risks=True,
            include_insights=True
        )
        
        return {"success": True, "analysis": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/trends")
def analyze_trends(request: AnalysisRequest):
    try:
        # Get documents
        documents = []
        for doc_id in request.document_ids:
            doc = document_agent.get_document(doc_id)
            if doc and doc.status == "completed":
                documents.append({
                    'period': doc.metadata.get('period', 'Unknown'),
                    'text': doc.text
                })
        
        if len(documents) < 2:
            raise HTTPException(status_code=400, detail="Need 2+ documents for trends")
        
        results = analysis_agent.analyze_trends(
            documents=documents,
            metrics=request.metrics
        )
        
        return {"success": True, "trends": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/compare")
def compare_documents(request: CompareRequest):
    try:
        doc1 = document_agent.get_document(request.doc_id_1)
        doc2 = document_agent.get_document(request.doc_id_2)
        
        if not doc1 or not doc2:
            raise HTTPException(status_code=404, detail="Document not found")
        
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
        
        results = analysis_agent.compare_documents(doc1_data, doc2_data, 'comprehensive')
        
        return {"success": True, "comparison": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compare error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/risks")
def analyze_risks(request: RiskAnalysisRequest):
    try:
        doc = document_agent.get_document(request.document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        results = analysis_agent.analyze_risks(doc.text, request.detailed)
        
        return {
            "success": True,
            "document_id": request.document_id,
            "risks": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/insights")
def generate_insights(document_ids: List[str]):
    try:
        documents = []
        for doc_id in document_ids:
            doc = document_agent.get_document(doc_id)
            if doc and doc.status == "completed":
                documents.append({
                    'period': doc.metadata.get('period', 'Unknown'),
                    'text': doc.text
                })
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents")
        
        results = analysis_agent.generate_insights(documents)
        
        return {"success": True, "insights": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metrics/extract")
def extract_metrics(request: MetricsRequest):
    try:
        from extraction.metric_extractor import MetricExtractor
        
        doc = document_agent.get_document(request.document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        extractor = MetricExtractor()
        
        if request.metric_name:
            # Specific metric
            metric = extractor.extract_metric_by_name(doc.text, request.metric_name)
            result = {
                'metric': request.metric_name,
                'value': metric.value if metric else None,
                'unit': metric.unit if metric else None,
                'period': metric.period if metric else None
            } if metric else None
        else:
            # All metrics
            all_metrics = extractor.extract_all_metrics(
                doc.text,
                doc.metadata.get('period')
            )
            result = {
                category: [
                    {
                        'name': m.name,
                        'value': m.value,
                        'unit': m.unit,
                        'period': m.period
                    }
                    for m in metrics
                ]
                for category, metrics in all_metrics.items()
            }
        
        return {
            "success": True,
            "document_id": request.document_id,
            "metrics": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
def search_documents(query: str, limit: int = 10):
    """Search documents"""
    try:
        results = document_agent.search_documents(query, limit=limit)
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": [doc.to_dict() for doc in results]
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
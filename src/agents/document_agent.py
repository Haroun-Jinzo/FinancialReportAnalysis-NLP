import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import hashlib
import json
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.document_parser import DocumentParser
from preprocessing.text_cleaner import TextCleaner
from models.ner_model import FinancialNER

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    filename: str
    path: str
    text: str = ""
    status: str = "uploaded"  # uploaded, processing, completed, failed
    uploaded_at: datetime = None
    processed_at: datetime = None
    
    # Extracted data
    entities: list = None
    keywords: list = None
    metadata: dict = None
    
    def __post_init__(self):
        if self.uploaded_at is None:
            self.uploaded_at = datetime.now()
        if self.entities is None:
            self.entities = []
        if self.keywords is None:
            self.keywords = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'text_preview': self.text[:200] if self.text else "",
            'status': self.status,
            'uploaded_at': self.uploaded_at.isoformat(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'entity_count': len(self.entities),
            'keyword_count': len(self.keywords),
            'metadata': self.metadata
        }


class DocumentAgent:
    def __init__(self, storage_path: str = "data/documents"):
        logger.info("📄 Starting Document Agent...")
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        self.parser = DocumentParser()
        self.cleaner = TextCleaner()
        self.ner = FinancialNER()
        
        # Store documents in memory
        self.documents = {}
        self._load_from_disk()
        
        logger.info(f"✓ Document Agent ready ({len(self.documents)} documents)")
    
    def ingest_document(self, file_path: str, metadata: dict = None) -> Document:
        logger.info(f"📥 Ingesting: {Path(file_path).name}")
        
        # Check if file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate unique ID
        doc_id = self._generate_id(file_path)
        
        # Check for duplicates
        if self._is_duplicate(file_path):
            logger.warning("⚠️  Duplicate document detected")
            return self._get_by_hash(self._calculate_hash(file_path))
        
        # Create document
        document = Document(
            id=doc_id,
            filename=Path(file_path).name,
            path=file_path,
            metadata=metadata or {}
        )
        
        self.documents[doc_id] = document
        self._save_to_disk()
        
        logger.info(f"✓ Document ingested: {doc_id}")
        return document
    
    def process_document(self, doc_id: str) -> Document:
        logger.info(f"⚙️  Processing: {doc_id}")
        
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")
        
        doc = self.documents[doc_id]
        doc.status = "processing"
        
        try:
            # Step 1: Parse file (PDF, Word, Excel)
            parsed = self.parser.parse(doc.path)
            doc.text = parsed['text']
            
            # Step 2: Clean text
            clean_text = self.cleaner.clean(doc.text, preserve_financial=True)
            
            # Step 3: Extract keywords
            keywords = self.cleaner.extract_keywords(clean_text, top_n=20)
            doc.keywords = [kw[0] for kw in keywords]
            
            # Step 4: Extract entities (companies, dates, money)
            entities = self.ner.extract_entities(clean_text)
            doc.entities = [
                {'text': e['text'], 'type': e['label'], 'confidence': e['score']}
                for e in entities[:50]  # Keep top 50
            ]
            
            # Done!
            doc.status = "completed"
            doc.processed_at = datetime.now()
            
            self._save_to_disk()
            
            logger.info(f"✓ Processed: {doc_id} ({len(doc.entities)} entities)")
            return doc
            
        except Exception as e:
            doc.status = "failed"
            logger.error(f"✗ Processing failed: {e}")
            raise
    
#utility methods

    def get_document(self, doc_id: str) -> Document:
        return self.documents.get(doc_id)
    
    def get_all_documents(self, status: str = None) -> list:
        docs = list(self.documents.values())
        
        if status:
            docs = [d for d in docs if d.status == status]
        
        return docs
    
    def search_documents(self, query: str, limit: int = 10) -> list:
        logger.info(f"🔍 Searching: {query}")
        
        query_lower = query.lower()
        results = []
        
        for doc in self.documents.values():
            if doc.status != "completed":
                continue
            
            score = 0
            
            # Check keywords
            if any(query_lower in kw.lower() for kw in doc.keywords):
                score += 3
            
            # Check entities
            if any(query_lower in e['text'].lower() for e in doc.entities):
                score += 2
            
            # Check text
            if query_lower in doc.text.lower():
                score += 1
            
            if score > 0:
                results.append((score, doc))
        
        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in results[:limit]]
    
    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._save_to_disk()
            logger.info(f"🗑️  Deleted: {doc_id}")
            return True
        return False
    
    def get_statistics(self) -> dict:
        total = len(self.documents)
        by_status = {}
        
        for doc in self.documents.values():
            by_status[doc.status] = by_status.get(doc.status, 0) + 1
        
        return {
            'total': total,
            'by_status': by_status
        }
    
    def _generate_id(self, filename: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"doc_{timestamp}_{hash(filename) % 10000}"
    
    def _calculate_hash(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _is_duplicate(self, file_path: str) -> bool:
        file_hash = self._calculate_hash(file_path)
        for doc in self.documents.values():
            if hasattr(doc, '_hash') and doc._hash == file_hash:
                return True
        return False
    
    def _get_by_hash(self, file_hash: str) -> Document:
        for doc in self.documents.values():
            if hasattr(doc, '_hash') and doc._hash == file_hash:
                return doc
        return None
    
    def _save_to_disk(self):
        registry_file = self.storage_path / "registry.json"
        
        data = []
        for doc in self.documents.values():
            data.append({
                'id': doc.id,
                'filename': doc.filename,
                'path': doc.path,
                'status': doc.status,
                'uploaded_at': doc.uploaded_at.isoformat(),
                'processed_at': doc.processed_at.isoformat() if doc.processed_at else None,
                'metadata': doc.metadata,
                'keywords': doc.keywords,
                'entities': doc.entities
            })
        
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_from_disk(self):
        registry_file = self.storage_path / "registry.json"
        
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)
            
            for item in data:
                doc = Document(
                    id=item['id'],
                    filename=item['filename'],
                    path=item['path'],
                    status=item['status'],
                    uploaded_at=datetime.fromisoformat(item['uploaded_at']),
                    processed_at=datetime.fromisoformat(item['processed_at']) if item.get('processed_at') else None,
                    metadata=item.get('metadata', {}),
                    keywords=item.get('keywords', []),
                    entities=item.get('entities', [])
                )
                self.documents[doc.id] = doc
            
            logger.info(f"📚 Loaded {len(self.documents)} documents from disk")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
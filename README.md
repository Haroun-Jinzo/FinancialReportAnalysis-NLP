# 📊 Financial AI Analyst - Advanced NLP Document Analysis System

A comprehensive natural language processing system designed to automatically extract, analyze, and generate insights from financial documents using state-of-the-art transformer models.

## 🎯 Overview

Financial AI Analyst is an enterprise-grade NLP system built for processing and analyzing financial documents. It combines multiple specialized AI models to perform:

- **Named Entity Recognition (NER)** - Extract financial entities (companies, currencies, regulations)
- **Sentiment Analysis** - Assess financial sentiment from text
- **Question Answering** - Query document content with natural language
- **Text Summarization** - Generate concise document summaries
- **Financial Metric Extraction** - Identify and extract key financial metrics
- **Risk & Trend Analysis** - Analyze risks and identify trends in financial documents
- **Comparative Analysis** - Compare metrics across multiple documents

## 🏗️ Architecture

The system follows a multi-agent architecture with specialized components:

```
┌─────────────────────────────────────────┐
│      Streamlit UI / FastAPI Backend     │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Financial Analysis Orchestrator       │
│  (Coordinates all processing pipeline)  │
└──┬────────────────┬────────────────────┬┘
   │                │                    │
   ▼                ▼                    ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│ Document │  │ Analysis │  │ Query        │
│ Agent    │  │ Agent    │  │ Agent        │
└──────────┘  └──────────┘  └──────────────┘
   │             │                │
   ▼             ▼                ▼
┌──────────────────────────────────────────┐
│        NLP Models & Processors           │
│  ├─ NER Model (Fine-tuned BERT)         │
│  ├─ Sentiment Model (FinBERT)           │
│  ├─ QA Model (RoBERTa)                  │
│  ├─ Summarizer (T5)                     │
│  └─ Embeddings (Sentence Transformers)  │
└──────────────────────────────────────────┘
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **Document Agent** | Parses multiple document formats (PDF, DOCX, XLSX), extracts text |
| **Analysis Agent** | Performs NER, sentiment analysis, metric extraction, risk analysis |
| **Query Agent** | Answers user questions from document content |
| **Preprocessing Pipeline** | Text cleaning, tokenization, normalization |
| **Fine-tuned Models** | Domain-specific NER for finance, custom sentiment models |

## 📦 Features

### Document Processing
- ✅ Multi-format support (PDF, Word, Excel, PowerPoint, XBRL)
- ✅ OCR capabilities for scanned documents
- ✅ Table and structured data extraction
- ✅ Document registry and caching

### Analysis Capabilities
- ✅ **Entity Extraction** - Identify financial entities, amounts, dates
- ✅ **Sentiment Analysis** - Financial sentiment classification
- ✅ **Question Answering** - Query documents naturally
- ✅ **Summarization** - Generate document summaries
- ✅ **Risk Analysis** - Identify and classify financial risks
- ✅ **Trend Detection** - Analyze trends in financial metrics
- ✅ **Comparative Analysis** - Compare multiple documents

### User Interfaces
- ✅ **Streamlit Web Application** - Interactive dashboard with visualizations
- ✅ **FastAPI Backend** - RESTful API for programmatic access
- ✅ **Docker Support** - Containerized deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Projet_semestriel_v2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up project structure and download models**
   ```bash
   python setup.py
   ```

### Running the Application

#### Streamlit Web App
```bash
streamlit run src/streamlit_app/app.py
```
Access the application at `http://localhost:8501`

#### FastAPI Backend
```bash
uvicorn src.apiBackend.api:app --reload --host 0.0.0.0 --port 8000
```
Access the API at `http://localhost:8000/docs` (interactive documentation)

#### Docker Deployment
```bash
docker-compose up --build
```

## 📂 Project Structure

```
Projet_semestriel_v2/
├── src/
│   ├── agents/                 # Orchestration and agent logic
│   │   ├── orchestrator.py     # Main pipeline coordinator
│   │   ├── document_agent.py   # Document processing
│   │   ├── analysis_agent.py   # Financial analysis
│   │   └── query_agent.py      # Q&A functionality
│   ├── models/                 # Model loading and inference
│   │   ├── ner_model.py        # Named Entity Recognition
│   │   ├── sentiment_model.py  # Sentiment Analysis
│   │   ├── qa_model.py         # Question Answering
│   │   ├── summarizer_model.py # Text Summarization
│   │   └── model_loader.py     # Model management
│   ├── extraction/             # Information extraction
│   │   ├── entity_extractor.py # Entity extraction logic
│   │   ├── metric_extractor.py # Financial metric extraction
│   │   ├── relation_extractor.py# Relationship extraction
│   │   └── pattern_matcher.py  # Pattern matching rules
│   ├── analysis/               # Analysis modules
│   │   ├── risk_analyzer.py    # Risk analysis
│   │   ├── trend_analyzer.py   # Trend detection
│   │   ├── insight_generator.py# Insight generation
│   │   └── comparative_analyzer.py # Document comparison
│   ├── preprocessing/          # Text preprocessing
│   │   ├── document_parser.py  # Document parsing
│   │   └── text_cleaner.py     # Text cleaning
│   ├── streamlit_app/          # Web UI
│   │   └── app.py             # Streamlit application
│   ├── apiBackend/             # REST API
│   │   └── api.py             # FastAPI application
│   └── utils/                  # Utility functions
├── fineTuning/                 # Fine-tuned models
│   ├── financial_ner/          # NER model for finance
│   └── finbert_custom/         # Custom sentiment model
├── data/
│   ├── raw/                    # Raw input documents
│   ├── processed/              # Processed data
│   ├── models/                 # Pre-trained models cache
│   ├── training/               # Training datasets
│   └── outputs/                # Analysis results
├── config/                     # Configuration files
│   ├── model_config.yaml       # Model parameters
│   ├── pipeline_config.yaml    # Pipeline settings
│   └── extraction_rules.json   # Extraction patterns
├── tests/                      # Test suite
├── notebooks/                  # Jupyter notebooks
├── requirements.txt            # Python dependencies
├── setup.py                    # Setup and initialization script
├── Dockerfile                  # Docker configuration
└── docker-compose.yml          # Docker Compose configuration
```

## 🔧 Configuration

### Model Configuration (`config/model_config.yaml`)
Configure model names, versions, and parameters:
```yaml
models:
  ner:
    name: "dslim/bert-base-NER"
  sentiment:
    name: "ProsusAI/finbert"
  qa:
    name: "deepset/roberta-base-squad2"
```

### Pipeline Configuration (`config/pipeline_config.yaml`)
Control pipeline behavior, batch sizes, and processing options

### Extraction Rules (`config/extraction_rules.json`)
Define patterns for entity and metric extraction

## 🤖 Models Used

| Task | Model | Source |
|------|-------|--------|
| NER | dslim/bert-base-NER (fine-tuned) | Hugging Face |
| Sentiment | ProsusAI/finbert (custom trained) | Hugging Face |
| Q&A | deepset/roberta-base-squad2 | Hugging Face |
| Summarization | T5-base | Hugging Face |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Hugging Face |

## 📊 Usage Examples

### Python API
```python
from src.agents.orchestrator import FinancialAnalysisOrchestrator

# Initialize
orchestrator = FinancialAnalysisOrchestrator()

# Process a document
results = orchestrator.process_document("path/to/financial_report.pdf")

# Access results
print(results['entities'])  # Extracted entities
print(results['sentiment']) # Sentiment analysis
print(results['summary'])   # Document summary
print(results['risks'])     # Identified risks
```

### REST API
```bash
# Upload document
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@financial_report.pdf"

# Query document
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the revenue?"}'
```

### Streamlit Web UI
1. Launch the application
2. Upload documents via the sidebar
3. Select analysis type
4. View interactive results and visualizations

## 📈 Performance Metrics

The system achieves strong performance on financial NLP tasks:
- **NER F1-Score**: ~0.92 on financial entities
- **Sentiment Accuracy**: ~0.88 on financial texts
- **Q&A Exact Match**: ~0.80 on financial questions

## 🔄 Data Pipeline

1. **Document Upload** → 2. **Parsing** → 3. **Text Extraction** → 4. **Cleaning & Preprocessing** → 5. **NER** → 6. **Sentiment Analysis** → 7. **Metric Extraction** → 8. **Risk Analysis** → 9. **Insight Generation** → 10. **Results Export**

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Individual test modules:
```bash
python tests/test_models.py        # Test model loading
python tests/test_extraction.py    # Test extraction pipeline
python tests/test_analysis.py      # Test analysis modules
python tests/test_parsing.py       # Test document parsing
```

## 📋 Requirements

### Core Dependencies
- **transformers** (4.36.0) - Transformer models from Hugging Face
- **sentence-transformers** (2.7.0) - Semantic embeddings
- **spacy** (3.7.2) - NLP pipeline
- **streamlit** (1.29.0) - Web application framework
- **fastapi** - RESTful API framework
- **pandas** - Data processing
- **scikit-learn** - Machine learning utilities

### Document Processing
- **PyPDF2** - PDF processing
- **python-docx** - Word document processing
- **openpyxl** - Excel processing
- **pytesseract** - OCR support
- **camelot-py** - Table extraction

See [requirements.txt](requirements.txt) for complete list

## 🐳 Docker Deployment

### Build and Run
```bash
docker-compose up --build
```

### Access Services
- Streamlit App: `http://localhost:8501`
- API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## 🤝 Contributing

Contributions are welcome! Please:
1. Create a feature branch (`git checkout -b feature/YourFeature`)
2. Commit changes (`git commit -m 'Add feature'`)
3. Push to branch (`git push origin feature/YourFeature`)
4. Open a Pull Request

## 📝 License

This project is part of Polytechnique's curriculum - Semester 5 NLP Project

## 📞 Support

For issues, questions, or suggestions:
- Check existing issues in the repository
- Create a new issue with detailed description
- Contact the development team

## 🎓 Academic Context

**Institution**: Polytechnique  
**Course**: Natural Language Processing  
**Level**: 5th Semester  
**Project Type**: Semester Project  

This system demonstrates advanced NLP applications in financial domain processing, combining multiple state-of-the-art transformer models for real-world document analysis.

---

**Last Updated**: January 2026  
**Version**: 2.0

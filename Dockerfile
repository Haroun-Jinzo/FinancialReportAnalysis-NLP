FROM python:3.9-slim

WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    default-jre \
    ghostscript \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip
RUN pip install --upgrade pip

# 3. Install PyTorch CPU version
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# 5. Download Spacy model
RUN python -m spacy download en_core_web_sm

# 6. Copy all source code
COPY . .

# 7. Download standard models (QA, Summarization, Embeddings)
RUN python Scripts/download_models.py || true

# 8. Create and populate model directories
RUN mkdir -p /app/data/models/ner /app/data/models/sentiment /app/data/raw /app/data/processed /app/data/documents

# 9. Copy fine-tuned models (REQUIRED - these models must exist)
COPY fineTuning/financial_ner/ /app/data/models/ner/
COPY fineTuning/finbert_custom/ /app/data/models/sentiment/

EXPOSE 8501 8000

# 10. Create startup script
RUN printf '#!/bin/bash\nuvicorn src.apiBackend.api:app --host 0.0.0.0 --port 8000 --workers 1 &\nstreamlit run src/streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0\n' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]

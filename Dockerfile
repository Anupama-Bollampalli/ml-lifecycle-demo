FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt /app/requirements.txt

# Install PyTorch CPU first (avoids downloading CUDA binaries)
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements (torch already satisfied)
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    "uvicorn[standard]==0.27.0" \
    scikit-learn==1.4.0 \
    joblib==1.3.2 \
    numpy==1.26.3 \
    pydantic==2.5.3

# Copy application code
COPY backend/ /app/backend/
COPY saved_models/ /app/saved_models/

WORKDIR /app/backend

# Pre-create saved_models dir (in case it's empty)
RUN mkdir -p /app/saved_models

EXPOSE 7863

CMD ["python", "-c", \
     "import uvicorn; uvicorn.run('main:app', host='0.0.0.0', port=7863, reload=False)"]

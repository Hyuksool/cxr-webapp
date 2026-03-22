FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .

# Install CPU-only torch first (smaller image), then other deps
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# TorchXRayVision model weights are downloaded automatically on first use
# (~100MB DenseNet checkpoint from HuggingFace/PyTorch Hub)

# Copy source code
COPY *.py ./

# Expose port
ENV PORT=8200
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8200\")}/health')"

CMD sh -c "uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 1"

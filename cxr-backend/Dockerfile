FROM python:3.12-slim

WORKDIR /app

# Install system dependencies + Node.js (for Claude CLI)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude CLI globally (required for claude_agent_sdk)
RUN npm install -g @anthropic-ai/claude-code --yes

# Copy requirements and install Python deps
# Build context is repo root, so prefix with cxr-backend/
COPY cxr-backend/requirements.txt .

# Install CPU-only torch first (smaller image), then other deps
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Set HuggingFace cache to persistent /app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# Pre-download CLIP model during build (avoids cold-start delay)
# Comment out if build time is too long; model will download on first request
RUN python3 -c "from transformers import CLIPProcessor, CLIPModel; \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    print('CLIP model cached')" || echo "CLIP pre-download failed — will download on first use"

# Copy source code
COPY cxr-backend/*.py ./

# Expose port
ENV PORT=8200
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=5 \
    CMD python3 -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8200\")}/health')"

CMD sh -c "uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 1"

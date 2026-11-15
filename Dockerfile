# Argus ML - Serving Application Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ARGUS_MODEL_STAGE=Staging

WORKDIR /app

# Install minimal OS dependencies needed for compiling C extensions / XGBoost / PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration and install Python dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy application source code and necessary directories
COPY agents/ ./agents/
COPY pipeline/ ./pipeline/
COPY tools/ ./tools/
COPY mcp_servers/ ./mcp_servers/
COPY serving/ ./serving/
COPY experiments/ ./experiments/
COPY data/ ./data/
COPY monitoring/ ./monitoring/

# Expose port for FastAPI serving
EXPOSE 8000

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start Uvicorn serving API
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Argus ML - FastAPI Serving Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ARGUS_MODEL_STAGE=Staging

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
# Install package and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

COPY agents/ ./agents/
COPY pipeline/ ./pipeline/
COPY tools/ ./tools/
COPY mcp_servers/ ./mcp_servers/
COPY serving/ ./serving/
COPY monitoring/ ./monitoring/

# Ensure directories exist even if empty
RUN mkdir -p experiments data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
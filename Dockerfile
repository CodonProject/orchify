FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ORCHIFY_LOG_LEVEL=INFO \
    ORCHIFY_RUN_STORE=/data/runs.db \
    ORCHIFY_MANAGER_SINK=/data/runs.jsonl

WORKDIR /app

COPY pyproject.toml README.md ./
COPY orchify ./orchify
RUN pip install --no-cache-dir ".[web]"

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json, urllib.request; \
print(json.load(urllib.request.urlopen('http://127.0.0.1:8010/api/health'))['ok'])"

# API keys (ORCHIFY_API_KEYS) and provider credentials (API_KEY / MODEL_ID /
# BASE_URL) come from the environment or an env-file at run time:
#   docker run --env-file .env -p 8010:8010 orchify
CMD ["orchify", "serve", "--host", "0.0.0.0", "--port", "8010"]

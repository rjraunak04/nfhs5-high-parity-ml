FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FERTILITY_MODEL_PATH=/app/models/demo_transport_model.joblib

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY models ./models
COPY data ./data

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[app]" \
    && python scripts/train_demo_model.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

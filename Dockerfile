FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UDW_OUTPUT_DIR=/data/analyses \
    UDW_MAX_WORKERS=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY 03_code ./03_code
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[api]"

VOLUME ["/data/analyses"]
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--app-dir", "03_code", "--host", "0.0.0.0", "--port", "8000"]


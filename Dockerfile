FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir "entropyos[api]"
COPY entropyos/ entropyos/

ENV ENTROPY_LOG_LEVEL=INFO
EXPOSE 8000

CMD ["uvicorn", "entropyos.api:app", "--host", "0.0.0.0", "--port", "8000"]

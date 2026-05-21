FROM python:3.11-slim

# Instalar dependencias del sistema que necesita LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src ./src

RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir pyarrow huggingface_hub

EXPOSE 8080

CMD ["uvicorn", "nido.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
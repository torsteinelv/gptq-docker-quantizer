FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir -U pip setuptools wheel uv

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY entrypoint.sh quantize.py ./
RUN chmod +x entrypoint.sh

# Vi bruker entrypoint-skriptet som start-kommando
ENTRYPOINT ["./entrypoint.sh"]

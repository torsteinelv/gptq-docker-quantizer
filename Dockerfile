# Bruk Nvidias offisielle bilde med CUDA 12.4 og byggeverktøy
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    TOKENIZERS_PARALLELISM=false

# Installer Python 3.11 og nødvendige byggeverktøy
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Gjør python3.11 til standard "python"-kommando
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Installer uv for lynraske installasjoner
RUN pip install --no-cache-dir -U pip setuptools wheel uv

COPY requirements.txt .
# Bruker uv til å installere PyTorch og resten
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY entrypoint.sh quantize.py ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Bruk Nvidias bilde for Ubuntu 24.04
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    TOKENIZERS_PARALLELISM=false

# Installer Python 3 (som er 3.12 i Ubuntu 24.04) og byggeverktøy
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Gjør python3 til standard "python"-kommando
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Installer uv (vi bruker --break-system-packages fordi Ubuntu 24.04 er streng på system-python)
RUN pip install --no-cache-dir -U pip setuptools wheel uv --break-system-packages

COPY requirements.txt .

# Bruker uv til å installere pakker system-wide
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY entrypoint.sh quantize.py ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

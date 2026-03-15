# Bruk Nvidias bilde for Ubuntu 24.04 med CUDA 13
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    TOKENIZERS_PARALLELISM=false

# 1. Installer systempakker. Vi inkluderer python3-pip og python3-venv.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Gjør python3 til standard "python"-kommando
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# 2. Installer 'uv' uten å prøve å oppgradere pip eller setuptools.
# Vi bruker --break-system-packages kun for å få lagt inn uv.
RUN pip install --no-cache-dir uv --break-system-packages

# 3. Kopier requirements
COPY requirements.txt .

# 4. Bruk uv til å installere resten. 
# Siden uv er en isolert binærfil, bryr den seg ikke om pip-konfliktene over.
RUN uv pip install --system --no-cache-dir -r requirements.txt

# 5. Kopier resten av koden
COPY entrypoint.sh quantize.py ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

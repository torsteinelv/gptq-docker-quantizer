FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    CUDA_ARCH_LIST="8.0;8.6;8.9" \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-venv \
    ninja-build \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

RUN pip install --no-cache-dir -U pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verifiser at torch faktisk finnes før GPTQModel bygges
RUN python -c "import torch; print(torch.__version__)"

RUN git clone https://github.com/ModelCloud/GPTQModel.git /tmp/GPTQModel && \
    cd /tmp/GPTQModel && \
    pip install -v . --no-build-isolation && \
    rm -rf /tmp/GPTQModel

COPY quantize.py /app/quantize.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

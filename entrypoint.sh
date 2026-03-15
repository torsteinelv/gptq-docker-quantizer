#!/bin/bash
set -e

echo "🚀 Henter absolutt nyeste versjon av GPTQModel fra GitHub..."
# Vi kloner til en temporær mappe for å ikke rote til /app
git clone https://github.com/ModelCloud/GPTQModel.git /tmp/GPTQModel
cd /tmp/GPTQModel

# Bruker --break-system-packages pga Ubuntu 24.04 og --no-build-isolation for å se PyTorch
pip install -v . --no-build-isolation --break-system-packages

# Gå tilbake til app-mappen
cd /app

echo "🧠 Starter kvantiseringsjobben..."
exec python quantize.py

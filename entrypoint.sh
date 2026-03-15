#!/bin/bash
set -e

echo "🚀 Henter absolutt nyeste versjon av GPTQModel fra GitHub..."
uv pip install --system --no-cache-dir -U git+https://github.com/ModelCloud/GPTQModel.git

echo "🧠 Starter kvantiseringsjobben..."
exec python quantize.py

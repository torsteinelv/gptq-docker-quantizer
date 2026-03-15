#!/usr/bin/env bash
set -euo pipefail

echo "Starter kvantiseringsjobben..."
exec python /app/quantize.py

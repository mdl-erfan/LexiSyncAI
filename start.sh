#!/bin/sh
set -eu

ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT INT TERM

for i in $(seq 1 60); do
    if ollama list >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
ollama list >/dev/null 2>&1

exec python -m streamlit run app/main.py \
    --server.address=0.0.0.0 \
    --server.port="${PORT:-7860}" \
    --server.headless=true

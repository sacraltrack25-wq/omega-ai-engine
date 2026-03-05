#!/bin/sh
# OMEGA Unified: Harvesters+Encoder on 8000, AI Engine on PORT

export AI_ENGINE_URL="http://localhost:${PORT:-4000}"
export ENCODER_SERVICE_URL="${ENCODER_SERVICE_URL:-http://localhost:8000/encoder}"

cd /app/harvesters && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 5
cd /app/core && exec node dist/server.js
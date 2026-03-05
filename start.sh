#!/bin/sh
# OMEGA Unified: Harvesters+Encoder on 8000, AI Engine on PORT

cd /app/harvesters && uvicorn main:app --host 0.0.0.0 --port 8000 &
cd /app/core && exec node dist/server.js
# OMEGA Unified

AI Engine + Harvesters + Encoder in one Docker container. Deploy on Render.

## Architecture

- Node AI Engine on PORT (Render traffic)
- Python Harvesters + Encoder on 8000 (internal)
- ENCODER_SERVICE_URL=http://localhost:8000/encoder

## Render

1. New в†’ Web Service
2. Connect repo: https://github.com/sacraltrack25-wq/omega-ai-engine
3. Root Directory: (leave empty)
4. Dockerfile Path: ./Dockerfile
5. Env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, AI_ENGINE_API_KEY, HF_TOKEN
   (ENCODER_SERVICE_URL is set automatically)

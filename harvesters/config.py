"""
OMEGA Harvester — конфигурация из переменных окружения.

Источники данных и куда они идут:
─────────────────────────────────────────────────────────────────────────────
  Harvester         →  AI Engine (/learn)          →  Supabase li_knowledge
  Harvester (медиа) →  Storage: media/             →  query_attachments
  Экспорт датасета  →  Storage: training-data/     →  curated_answers.export_path
  KnowledgeStore    →  Storage: knowledge-export/  →  снимок всех Li знаний
─────────────────────────────────────────────────────────────────────────────
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from harvesters dir (works when run via uvicorn from project root)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)
load_dotenv()  # fallback: cwd

# ── Supabase ───────────────────────────────────────────────────────────────────
# Откуда: harvesters/.env → SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
# Зачем: подключение к БД (harvester_jobs, li_knowledge) и Storage
SUPABASE_URL         = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# ── Supabase Storage Buckets ───────────────────────────────────────────────────
# Откуда: harvesters/.env → SUPABASE_BUCKET_*
#
#   BUCKET_MEDIA           = "media"
#     Структура: media/{user_id_or_job_id}/{network_type}/{filename}
#     Пример:   media/job-abc123/video/clip.mp4
#     Кто пишет: VideoHarvester, AudioHarvester, ImageHarvester
#     Кто читает: AI Engine (по запросу), пользователи (только свои файлы)
#
#   BUCKET_TRAINING_DATA   = "training-data"
#     Структура: training-data/{network_type}/{dataset_name}.jsonl
#     Пример:   training-data/text/wiki_science_2026-03-01.jsonl
#     Кто пишет: train_textnet.py, экспорт curated_answers
#     Кто читает: только admins
#
#   BUCKET_KNOWLEDGE_EXPORT = "knowledge-export"
#     Структура: knowledge-export/snapshot_{timestamp}.json
#     Пример:   knowledge-export/snapshot_2026-03-01T12:00:00.json
#     Кто пишет: KnowledgeStore.export_snapshot() после consolidate()
#     Кто читает: KnowledgeStore.restore_from_snapshot() при старте
BUCKET_MEDIA            = os.getenv("SUPABASE_BUCKET_MEDIA",            "media")
BUCKET_TRAINING_DATA    = os.getenv("SUPABASE_BUCKET_TRAINING_DATA",    "training-data")
BUCKET_KNOWLEDGE_EXPORT = os.getenv("SUPABASE_BUCKET_KNOWLEDGE_EXPORT", "knowledge-export")

# ── AI Engine ─────────────────────────────────────────────────────────────────
# Откуда: harvesters/.env → AI_ENGINE_URL, AI_ENGINE_API_KEY
# Зачем: POST /learn — отправка обученных векторов в Li центры
AI_ENGINE_URL     = os.getenv("AI_ENGINE_URL",     "http://localhost:4000")
AI_ENGINE_API_KEY = os.environ["AI_ENGINE_API_KEY"]

# ── Harvester settings ────────────────────────────────────────────────────────
HARVEST_INTERVAL        = int(os.getenv("HARVEST_INTERVAL",        "3600"))
MAX_WORKERS             = int(os.getenv("MAX_WORKERS",              "4"))
# 0.5 — Wikipedia/веб-тексты часто дают 0.5–0.75; 0.7 отсекал почти всё
DATA_QUALITY_THRESHOLD  = float(os.getenv("DATA_QUALITY_THRESHOLD", "0.5"))
DEDUP_THRESHOLD         = float(os.getenv("DEDUPLICATION_THRESHOLD","0.92"))

# ── Media & Export settings ───────────────────────────────────────────────────
# Файлы крупнее порога сохраняются в Storage, не передаются сырыми байтами в AI
MEDIA_STORAGE_THRESHOLD = int(os.getenv("MEDIA_STORAGE_THRESHOLD_BYTES", "524288"))  # 512 KB
# После N обученных элементов экспортировать датасет в training-data bucket
TRAINING_EXPORT_EVERY   = int(os.getenv("TRAINING_EXPORT_EVERY", "1000"))

# ── Text embedding models (sentence-transformers) ──────────────────────────────
# Два многоязычных модели для RU+EN. Векторы конкатенируются → 384+512=896-dim.
# paraphrase-multilingual-MiniLM-L12-v2  — 384-dim, быстрый
# distiluse-base-multilingual-cased-v2   — 512-dim, качественный
TEXT_EMBEDDING_MODELS = [
    m.strip() for m in os.getenv(
        "TEXT_EMBEDDING_MODELS",
        "paraphrase-multilingual-MiniLM-L12-v2,distiluse-base-multilingual-cased-v2",
    ).split(",") if m.strip()
]
if not TEXT_EMBEDDING_MODELS:
    TEXT_EMBEDDING_MODELS = ["paraphrase-multilingual-MiniLM-L12-v2", "distiluse-base-multilingual-cased-v2"]

# ── Limits ────────────────────────────────────────────────────────────────────
MAX_CONTENT_SIZE  = 5 * 1024 * 1024   # 5 MB — максимум на один item
REQUEST_TIMEOUT   = 30                 # секунд на HTTP запрос
MAX_RETRIES       = 3

# ── HTTP headers ───────────────────────────────────────────────────────────────
# Wikipedia и многие сайты блокируют запросы без User-Agent (403)
USER_AGENT = os.getenv(
    "HARVESTER_USER_AGENT",
    "OMEGA-Harvester/1.0 (Educational AI research; +https://github.com/omega-ai) Python/3",
)

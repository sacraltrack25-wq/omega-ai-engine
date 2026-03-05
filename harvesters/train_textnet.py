"""
train_textnet.py — Массовое обучение TextNet
============================================
Скачивает и скармливает качественные тексты в gX-Li-Omega движок.

Источники данных (по уровням качества):

  УРОВЕНЬ 1 — Факты и знания (Wikipedia, научные статьи)
  УРОВЕНЬ 2 — Разнообразный текст (новости, книги)
  УРОВЕНЬ 3 — Специализированные данные (код, математика)

Запуск:
  python train_textnet.py --level 1             # быстрый старт (Wikipedia)
  python train_textnet.py --level 2             # +новости, блоги
  python train_textnet.py --level 3 --gpu       # всё + GPU
  python train_textnet.py --topic "physics"     # только определённая тема
  python train_textnet.py --url https://...     # один сайт

  # Быстрое обучение — готовые датасеты:
  python train_textnet.py --lama trex          # LAMA TREx (Wikidata факты)
  python train_textnet.py --lama conceptnet    # LAMA ConceptNet (common sense)
  python train_textnet.py --dataset data.jsonl # локальный JSONL
  python train_textnet.py --dataset data.jsonl --batch 8  # параллельно 8 потоков

  # Hugging Face (любой датасет по имени):
  python train_textnet.py --hf-dataset HuggingFaceFW/fineweb --hf-column text --limit 10000 --streaming --batch 8
  python train_textnet.py --hf-dataset squad --hf-split train --limit 5000 --batch 8
  python train_textnet.py --config datasets_config.yaml --batch 8
"""

import asyncio
import argparse
import hashlib
import yaml
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

AI_ENGINE_URL = os.getenv("AI_ENGINE_URL", "http://localhost:4000")
AI_ENGINE_KEY = os.getenv("AI_ENGINE_API_KEY", "generate-a-strong-random-key")

# ── Harvest log (Supabase) ────────────────────────────────────────────────────

def _get_supabase():
    """Optional Supabase client for harvest_log. Returns None if not configured."""
    try:
        from config import SUPABASE_URL, SUPABASE_SERVICE_KEY
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception:
        return None


def _log_harvest_start(job_id: str, source_type: str, source_id: str, config: dict | None) -> None:
    supabase = _get_supabase()
    if not supabase:
        return
    try:
        supabase.table("harvest_log").upsert({
            "job_id": job_id,
            "source_type": source_type,
            "source_id": source_id,
            "config": config or {},
            "status": "running",
        }, on_conflict="job_id").execute()
    except Exception as e:
        logger.warning("harvest_log start failed: %s", e)


def _log_harvest_complete(job_id: str, items_count: int, error: str | None = None) -> None:
    supabase = _get_supabase()
    if not supabase:
        return
    try:
        supabase.table("harvest_log").update({
            "items_count": items_count,
            "status": "completed" if not error else "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }).eq("job_id", job_id).execute()
    except Exception as e:
        logger.warning("harvest_log complete failed: %s", e)


# ── Источники данных ──────────────────────────────────────────────────────────

# Уровень 1: Wikipedia API — лучший старт (структурированные факты)
WIKIPEDIA_TOPICS_RU = [
    "Искусственный интеллект", "Нейронная сеть", "Машинное обучение",
    "Физика", "Химия", "Биология", "Математика", "История",
    "Философия", "Экономика", "Программирование", "Квантовая механика",
    "Космос", "Эволюция", "Психология", "Лингвистика",
]

WIKIPEDIA_TOPICS_EN = [
    "Artificial intelligence", "Deep learning", "Neural network",
    "Physics", "Chemistry", "Biology", "Mathematics", "History",
    "Philosophy", "Economics", "Computer science", "Quantum mechanics",
    "Astronomy", "Evolution", "Psychology", "Linguistics",
    "Cognitive science", "Neuroscience", "Robotics", "Ethics",
    "Logic", "Information theory", "Complexity theory",
]

# Уровень 2: Качественные сайты
QUALITY_SITES_L2 = [
    # Наука
    "https://www.quantamagazine.org",
    "https://arstechnica.com/science",
    "https://www.newscientist.com",
    "https://phys.org",
    # Технологии
    "https://techcrunch.com",
    "https://www.wired.com",
    "https://www.technologyreview.com",
    # Образование
    "https://plato.stanford.edu",    # Stanford Encyclopedia of Philosophy
    "https://mathworld.wolfram.com", # Wolfram MathWorld
    "https://arxiv.org/list/cs.AI/recent",  # AI papers
]

# Уровень 3: Специализированные
QUALITY_SITES_L3 = [
    "https://huggingface.co/blog",
    "https://openai.com/research",
    "https://deepmind.google/research",
    "https://distill.pub",
    "https://lilianweng.github.io",
    "https://colah.github.io",
]


# ── Wikipedia fetcher ─────────────────────────────────────────────────────────

async def fetch_wikipedia(topic: str, lang: str = "en") -> str | None:
    """Fetch a Wikipedia article via the REST API."""
    import httpx
    title = topic.replace(" ", "_")
    url   = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                return data.get("extract", "")
    except Exception as e:
        logger.warning("Wikipedia fetch failed for '%s': %s", topic, e)
    return None


async def fetch_wikipedia_full(topic: str, lang: str = "en") -> str | None:
    """Fetch full Wikipedia article text via MediaWiki API."""
    import httpx
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": topic,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "exsectionformat": "plain",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, params=params)
            if r.status_code == 200:
                pages = r.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    return page.get("extract", "")
    except Exception as e:
        logger.warning("Wikipedia full fetch failed for '%s': %s", topic, e)
    return None


# ── Feed to AI Engine ─────────────────────────────────────────────────────────

async def _send_learn(chunk: str, source: str, vector: list[float]) -> bool:
    """Send one chunk to /learn."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{AI_ENGINE_URL}/learn",
                json={
                    "type":   "text",
                    "key":    hashlib.sha256(chunk.encode()).hexdigest()[:16],
                    "data":   vector,
                    "source": source,
                },
                headers={"x-api-key": AI_ENGINE_KEY},
            )
            return resp.status_code == 200
    except Exception as e:
        logger.warning("Learn failed: %s", e)
        return False


async def _send_learn_batch(items: list[tuple[str, str, list[float]]]) -> int:
    """Send batch to /learn-batch. Returns count of successful sends."""
    import httpx
    if not items:
        return 0
    payload = {
        "items": [
            {
                "type":   "text",
                "key":    hashlib.sha256(chunk.encode()).hexdigest()[:16],
                "data":   vector,
                "source": source,
            }
            for chunk, source, vector in items
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{AI_ENGINE_URL}/learn-batch",
                json=payload,
                headers={"x-api-key": AI_ENGINE_KEY},
            )
            if resp.status_code == 200:
                return len(items)
            return 0
    except Exception as e:
        logger.warning("Learn-batch failed: %s", e)
        return 0


async def feed_text(text: str, source: str, encoder, batch_size: int = 1) -> int:
    """Encode text and send to AI Engine /learn endpoint.
    batch_size > 1: parallel requests for faster ingestion."""
    import httpx

    if not text or len(text) < 50:
        return 0

    # Split into chunks of ~400 words
    words  = text.split()
    chunks = [" ".join(words[i:i+400]) for i in range(0, len(words), 350)]

    total = 0
    sem = __import__("asyncio").Semaphore(batch_size) if batch_size > 1 else None

    async def do_one(chunk: str):
        if len(chunk) < 50:
            return 0
        vector = await encoder(chunk)
        if sem:
            async with sem:
                ok = await _send_learn(chunk, source, vector)
        else:
            ok = await _send_learn(chunk, source, vector)
        return 1 if ok else 0

    tasks = [do_one(c) for c in chunks if len(c) >= 50]
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total = sum(r for r in results if isinstance(r, int))

    return total


# ── Semantic encoder ──────────────────────────────────────────────────────────

def make_encoder(batch_size: int = 32):
    """Load semantic encoders. Returns encoder with encode() and encode_batch()."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from config import TEXT_EMBEDDING_MODELS
        names = TEXT_EMBEDDING_MODELS
        models = [SentenceTransformer(n) for n in names]
        total_dim = sum(m.get_sentence_embedding_dimension() for m in models)
        logger.info("Semantic encoder loaded: %s (total %d-dim)", ", ".join(names), total_dim)

        def _norm_concat(parts_list: list) -> list[list[float]]:
            out = []
            for parts in parts_list:
                concat = np.concatenate(parts)
                norm = np.linalg.norm(concat)
                if norm > 1e-8:
                    concat = concat / norm
                out.append(concat.tolist())
            return out

        async def encode(text: str) -> list[float]:
            parts = [m.encode(text[:512], normalize_embeddings=True) for m in models]
            return _norm_concat([parts])[0]

        async def encode_batch(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            truncated = [t[:512] for t in texts]
            parts_per_model = [
                m.encode(truncated, batch_size=batch_size, normalize_embeddings=True)
                for m in models
            ]
            # parts_per_model[i] = array (n_texts, dim_i)
            by_text = [
                [parts_per_model[mi][ti] for mi in range(len(models))]
                for ti in range(len(texts))
            ]
            return _norm_concat(by_text)

        enc = encode
        enc.encode_batch = encode_batch
        return enc
    except ImportError:
        logger.warning("sentence-transformers not found — using fallback encoder")

        async def fallback(text: str) -> list[float]:
            words = text.lower().split()
            vec = [0.0] * 256
            for w in words:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16) % 256
                vec[h] += 1.0
            m = max(abs(v) for v in vec) or 1.0
            return [v / m for v in vec]

        async def fallback_batch(texts: list[str]) -> list[list[float]]:
            return [await fallback(t) for t in texts]

        enc = fallback
        enc.encode_batch = fallback_batch
        return enc


# ── Hugging Face streaming ────────────────────────────────────────────────────

def _extract_text_from_record(record: dict, column: str | None, text_from: str | list | None) -> str | None:
    """Extract text from HF record. Supports column name or text_from mapping."""
    if column:
        val = record.get(column)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list) and val:
            return " ".join(str(v) for v in val[:3]).strip()
    if text_from:
        if isinstance(text_from, str):
            return record.get(text_from, "")
        if isinstance(text_from, list):
            parts = [str(record.get(k, "")) for k in text_from]
            return " ".join(p for p in parts if p).strip()
    return record.get("text", "").strip() or None


async def train_from_hf_streaming(
    name: str,
    encoder,
    config: str | None = None,
    split: str = "train",
    column: str | None = "text",
    text_from: str | list | None = None,
    limit: int | None = None,
    streaming: bool = True,
    batch_size: int = 8,
    encode_batch_size: int = 32,
    filters: list | None = None,
) -> int:
    """Load any Hugging Face dataset with streaming and feed to OMEGA."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install: pip install datasets")
        return 0

    job_id = str(uuid.uuid4())
    source_id = f"{name}" + (f":{config}" if config else "")
    cfg = {"split": split, "column": column, "limit": limit, "streaming": streaming}
    _log_harvest_start(job_id, "huggingface", source_id, cfg)

    logger.info("=== HF %s (streaming=%s) ===", source_id, streaming)
    total = 0
    err_msg = None

    try:
        load_kw = {"split": split, "streaming": streaming}
        cols = None
        if text_from:
            cols = [text_from] if isinstance(text_from, str) else text_from
        elif column:
            cols = [column] if isinstance(column, str) else column
        if cols:
            load_kw["columns"] = cols
        if filters and streaming:
            load_kw["filters"] = filters

        if config:
            ds = load_dataset(name, config, **load_kw)
        else:
            ds = load_dataset(name, **load_kw)
        if limit:
            ds = ds.take(limit)
        if streaming:
            ds = ds.shuffle(seed=42, buffer_size=5000)

        batch_texts: list[tuple[str, str]] = []
        source_prefix = f"hf/{name}"
        if config:
            source_prefix += f"/{config}"

        for i, record in enumerate(ds):
            text = _extract_text_from_record(dict(record), column, text_from)
            if not text or len(text) < 10:
                continue
            batch_texts.append((text, source_prefix))

            if len(batch_texts) >= encode_batch_size:
                vectors = await encoder.encode_batch([t for t, _ in batch_texts])
                items = [(t, s, v) for (t, s), v in zip(batch_texts, vectors)]
                sent = await _send_learn_batch(items)
                total += sent
                batch_texts = []
                if (i + 1) % 500 == 0 or (i + 1) % encode_batch_size == 0:
                    logger.info("  HF %s: %d ingested", source_id, total)

        if batch_texts:
            vectors = await encoder.encode_batch([t for t, _ in batch_texts])
            items = [(t, s, v) for (t, s), v in zip(batch_texts, vectors)]
            total += await _send_learn_batch(items)

        logger.info("  HF %s: %d ingested", source_id, total)
    except Exception as e:
        err_msg = str(e)
        logger.error("HF streaming failed: %s", e)

    _log_harvest_complete(job_id, total, err_msg)
    return total


async def _train_from_config(config_path: str, encoder, batch_size: int) -> int:
    """Load datasets_config.yaml and process each source."""
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sources = data.get("sources", [])
    if not sources:
        logger.warning("No sources in %s", config_path)
        return 0
    total = 0
    for src in sources:
        if src.get("type") != "huggingface":
            logger.warning("Skipping unknown source type: %s", src.get("type"))
            continue
        name = src.get("name")
        if not name:
            logger.warning("Source missing 'name', skipping")
            continue
        columns = src.get("columns")
        column = columns[0] if isinstance(columns, list) and columns else src.get("column", "text")
        text_from = src.get("text_from")
        total += await train_from_hf_streaming(
            name,
            encoder,
            config=src.get("config"),
            split=src.get("split", "train"),
            column=column,
            text_from=text_from,
            limit=src.get("limit"),
            streaming=src.get("streaming", True),
            batch_size=batch_size,
            filters=src.get("filters"),
        )
    return total


# ── LAMA dataset ──────────────────────────────────────────────────────────────

def _lama_to_text(record: dict) -> str | None:
    """Convert LAMA record to factual sentence for embedding."""
    masked = record.get("masked_sentence") or ""
    obj = record.get("obj_label") or record.get("obj") or ""
    if masked and obj and "[MASK]" in masked:
        return masked.replace("[MASK]", str(obj)).strip()
    template = record.get("template") or ""
    sub = record.get("sub_label") or record.get("sub") or ""
    if template and sub and obj:
        return template.replace("[X]", sub).replace("[Y]", str(obj)).strip()
    if sub and obj:
        pred = record.get("label") or record.get("pred") or record.get("predicate_id") or ""
        return f"{sub} {pred} {obj}".strip() if pred else f"{sub} — {obj}".strip()
    return None


async def train_from_lama(config: str, encoder, batch_size: int = 4, limit: int | None = None) -> int:
    """Load LAMA from HuggingFace (facebook/lama) and feed to OMEGA."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install: pip install datasets")
        return 0

    job_id = str(uuid.uuid4())
    source_id = f"lama/{config}"
    cfg = {"limit": limit}
    _log_harvest_start(job_id, "lama", source_id, cfg)

    logger.info("=== LAMA %s (HuggingFace) ===", config)
    ds = load_dataset("facebook/lama", config, split="train")
    total = 0
    sem = asyncio.Semaphore(batch_size)

    async def send_one(record: dict):
        text = _lama_to_text(record)
        if not text or len(text) < 10:
            return 0
        vector = await encoder(text)
        async with sem:
            return 1 if await _send_learn(text, f"lama/{config}", vector) else 0

    it = iter(ds)
    if limit:
        it = __import__("itertools").islice(it, limit)
    batch = []
    for i, record in enumerate(it):
        batch.append(record)
        if len(batch) >= 32:
            tasks = [send_one(r) for r in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total += sum(r for r in results if isinstance(r, int))
            batch = []
            if (i + 1) % 200 == 0:
                logger.info("  LAMA %s: %d ingested", config, total)

    if batch:
        tasks = [send_one(r) for r in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total += sum(r for r in results if isinstance(r, int))

    logger.info("  LAMA %s: %d ingested", config, total)
    _log_harvest_complete(job_id, total, None)
    return total


async def train_from_dataset_file(path: str, encoder, batch_size: int = 4) -> int:
    """Load JSONL from file. Each line: JSON with 'text' or LAMA-style fields."""
    import json
    total = 0
    sem = asyncio.Semaphore(batch_size)

    async def send_one(obj: dict, source: str):
        text = obj.get("text") or _lama_to_text(obj)
        if not text or len(text) < 10:
            return 0
        vector = await encoder(text)
        async with sem:
            return 1 if await _send_learn(text, source, vector) else 0

    job_id = str(uuid.uuid4())
    source_id = f"dataset/{path}"
    _log_harvest_start(job_id, "jsonl", source_id, {"path": path})

    source = source_id
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    logger.info("=== Dataset %s — %d lines ===", path, len(lines))
    # Process in chunks to avoid too many concurrent tasks
    chunk_size = 100
    for i in range(0, len(lines), chunk_size):
        batch_lines = lines[i:i + chunk_size]
        objs = []
        for l in batch_lines:
            try:
                objs.append(json.loads(l))
            except json.JSONDecodeError:
                pass
        tasks = [send_one(o, source) for o in objs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total += sum(r for r in results if isinstance(r, int))
        if (i + chunk_size) % 500 == 0 or i + chunk_size >= len(lines):
            logger.info("  Dataset: %d/%d ingested", total, len(lines))
    logger.info("  Dataset: %d ingested", total)
    _log_harvest_complete(job_id, total, None)
    return total


# ── Harvesters ────────────────────────────────────────────────────────────────

async def train_from_wikipedia(topics: list[str], lang: str, encoder, desc: str, batch_size: int = 1):
    logger.info("=== %s — %d topics ===", desc, len(topics))
    ok = 0
    for topic in topics:
        text = await fetch_wikipedia_full(topic, lang)
        if text:
            count = await feed_text(text, f"wikipedia/{lang}/{topic}", encoder, batch_size)
            if count > 0:
                ok += 1
                logger.info("  ✓ [%d/%d] %s (%d chunks)", ok, len(topics), topic, count)
        await asyncio.sleep(0.3)  # polite crawling
    logger.info("  Wikipedia %s: %d/%d topics ingested", lang, ok, len(topics))
    return ok


async def train_from_url(url: str, encoder):
    """Harvest a single URL."""
    from harvesters.web_harvester import WebHarvester

    class _FakeSupabase:
        def table(self, *a): return self
        def insert(self, *a): return self
        def update(self, *a): return self
        def eq(self, *a): return self
        def execute(self): pass

    h = WebHarvester(_FakeSupabase())
    count = 0
    try:
        async for item in h.harvest(url):
            if item.quality >= 0.4:
                async with __import__("httpx").AsyncClient(timeout=20) as client:
                    resp = await client.post(
                        f"{AI_ENGINE_URL}/learn",
                        json={"type": "text", "key": item.key,
                              "data": item.data, "source": item.source},
                        headers={"x-api-key": AI_ENGINE_KEY},
                    )
                    if resp.status_code == 200:
                        count += 1
    except Exception as e:
        logger.warning("URL harvest failed for %s: %s", url, e)
    finally:
        await h.close()
    logger.info("  ✓ %s — %d items", url, count)
    return count


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Train OMEGA TextNet")
    parser.add_argument("--level",  type=int, default=1,  choices=[1, 2, 3],
                        help="Training level: 1=Wikipedia, 2=+sites, 3=+AI research")
    parser.add_argument("--topic",  type=str, default=None,
                        help="Single Wikipedia topic to train on")
    parser.add_argument("--url",    type=str, default=None,
                        help="Single URL to harvest")
    parser.add_argument("--lang",   type=str, default="both",
                        choices=["en", "ru", "both"],
                        help="Wikipedia language(s)")
    parser.add_argument("--gpu",    action="store_true",
                        help="Use GPU for encoding (requires CUDA)")
    parser.add_argument("--lama",   type=str, default=None,
                        choices=["trex", "conceptnet", "squad", "google_re"],
                        help="LAMA dataset from HuggingFace (fast factual learning)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Local JSONL file (each line: {\"text\": \"...\"} or LAMA-style)")
    parser.add_argument("--batch",  type=int, default=4,
                        help="Parallel requests for /learn (default 4, try 8 for speed)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Max records for --lama / --hf-dataset (e.g. 1000 for quick test)")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="Hugging Face dataset name (e.g. HuggingFaceFW/fineweb, squad)")
    parser.add_argument("--hf-config", type=str, default=None,
                        help="HF dataset config/subset (e.g. en for FineWeb)")
    parser.add_argument("--hf-column", type=str, default="text",
                        help="Column name for text (default: text)")
    parser.add_argument("--hf-split", type=str, default="train",
                        help="Dataset split (default: train)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming for large HF datasets")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets_config.yaml")
    parser.add_argument("--duration-hours", type=float, default=None,
                        help="Run training in a loop until N hours elapsed (for scheduled training)")
    args = parser.parse_args()

    # Check AI Engine
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{AI_ENGINE_URL}/health",
                                  headers={"x-api-key": AI_ENGINE_KEY})
            if r.status_code != 200:
                raise Exception(f"status {r.status_code}")
        logger.info("[OK] AI Engine connected: %s", AI_ENGINE_URL)
    except Exception as e:
        logger.error("✗ AI Engine not reachable at %s: %s", AI_ENGINE_URL, e)
        logger.error("  Make sure to run: pnpm --filter @omega/core start")
        sys.exit(1)

    encoder = make_encoder(batch_size=32)
    start   = time.time()
    total   = 0
    batch   = max(1, args.batch)
    duration_sec = (args.duration_hours or 0) * 3600 if args.duration_hours else 0

    # ── Duration mode: loop until time elapsed; otherwise single pass ──
    pass_num = 0
    while True:
        pass_num += 1
        if duration_sec:
            elapsed = time.time() - start
            if elapsed >= duration_sec:
                logger.info("Duration %.1f h reached, stopping.", args.duration_hours)
                break
            logger.info("=== Pass %d (%.1f h left) ===", pass_num, (duration_sec - elapsed) / 3600)

        loop_total = 0
        if args.config:
            loop_total += await _train_from_config(args.config, encoder, batch)
        elif args.hf_dataset:
            loop_total += await train_from_hf_streaming(
                args.hf_dataset,
                encoder,
                config=args.hf_config,
                split=args.hf_split,
                column=args.hf_column,
                limit=args.limit,
                streaming=args.streaming,
                batch_size=batch,
            )
        elif args.lama:
            loop_total += await train_from_lama(args.lama, encoder, batch, args.limit)
        elif args.dataset:
            loop_total += await train_from_dataset_file(args.dataset, encoder, batch)
        elif args.url:
            loop_total += await train_from_url(args.url, encoder)
        elif args.topic:
            for lang in (["en", "ru"] if args.lang == "both" else [args.lang]):
                loop_total += await train_from_wikipedia([args.topic], lang, encoder, f"Topic: {args.topic}", batch)
        else:
            if args.lang in ("en", "both"):
                loop_total += await train_from_wikipedia(
                    WIKIPEDIA_TOPICS_EN, "en", encoder, "Wikipedia EN", batch)
            if args.lang in ("ru", "both"):
                loop_total += await train_from_wikipedia(
                    WIKIPEDIA_TOPICS_RU, "ru", encoder, "Wikipedia RU", batch)
            if args.level >= 2:
                logger.info("=== Level 2 — Quality websites ===")
                for url in QUALITY_SITES_L2:
                    loop_total += await train_from_url(url, encoder)
            if args.level >= 3:
                logger.info("=== Level 3 — AI Research sites ===")
                for url in QUALITY_SITES_L3:
                    loop_total += await train_from_url(url, encoder)

        total += loop_total

        if not duration_sec:
            break
        elapsed = time.time() - start
        if elapsed >= duration_sec:
            logger.info("Duration %.1f h reached, stopping.", args.duration_hours)
            break

    elapsed = time.time() - start
    logger.info("")
    logger.info("════════════════════════════════════════")
    logger.info("Training complete!")
    logger.info("  Total items fed to TextNet: %d", total)
    logger.info("  Time: %.1f sec (%.1f min)", elapsed, elapsed / 60)
    logger.info("  AI Engine: %s", AI_ENGINE_URL)
    logger.info("════════════════════════════════════════")
    logger.info("")
    logger.info("Next step: run memory consolidation")
    logger.info("  curl -X POST %s/consolidate -H 'x-api-key: %s'",
                AI_ENGINE_URL, AI_ENGINE_KEY)


if __name__ == "__main__":
    asyncio.run(main())

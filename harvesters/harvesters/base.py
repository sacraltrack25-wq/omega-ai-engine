"""
BaseHarvester — абстрактная база для всех OMEGA харвесторов.

Поток данных:
─────────────────────────────────────────────────────────────────────────────
  Источник (URL)
    │
    ▼
  harvest() — скрапинг и парсинг
    │
    ▼
  encode()  — конвертация в вектор
    │
    ├─[маленький файл < 512KB]──────────────────────────────────────────────►
    │                           _feed_to_engine(item)
    │                           POST AI_ENGINE_URL/learn
    │                           → Li центры в RAM → li_knowledge в Supabase
    │
    └─[большой файл >= 512KB]──────────────────────────────────────────────►
                                _save_media_to_storage(item)
                                Supabase Storage: media/job_{job_id}/{network}/{filename}
                                + query_attachments таблица
                                + _feed_to_engine(item) — только вектор без raw

После N обученных элементов:
  _export_training_dataset()
  Supabase Storage: training-data/{network_type}/harvest_{timestamp}.jsonl
  Формат JSONL: {"key": ..., "raw": ..., "source": ..., "vector": [...]}
─────────────────────────────────────────────────────────────────────────────
"""
import hashlib
import io
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from datetime import datetime, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    AI_ENGINE_URL, AI_ENGINE_API_KEY, MAX_RETRIES,
    REQUEST_TIMEOUT, DATA_QUALITY_THRESHOLD,
    BUCKET_MEDIA, BUCKET_TRAINING_DATA,
    MEDIA_STORAGE_THRESHOLD, TRAINING_EXPORT_EVERY,
    USER_AGENT,
)

logger = logging.getLogger(__name__)


@dataclass
class HarvestedItem:
    """Единица собранных данных."""
    key:        str            # SHA-256 хэш (первые 16 символов)
    data:       list[float]    # Feature vector (размерность зависит от сети)
    source:     str            # Исходный URL
    network:    str            # Целевая сеть: text / image / video / audio / game
    raw:        Any = None     # Оригинальный контент (текст/bytes)
    quality:    float = 1.0    # Качество [0, 1]
    metadata:   dict  = field(default_factory=dict)


class BaseHarvester(ABC):
    """
    Абстрактный харвестор.
    Подклассы реализуют encode() и harvest().

    Суpabase client используется для:
      - Отслеживания статуса jobs (harvester_jobs таблица)
      - Сохранения медиафайлов в Storage (media bucket)
      - Экспорта датасетов (training-data bucket)
    """

    network_type: str = "text"

    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self._client  = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        self._seen:     set[str]          = set()
        self._dataset:  list[dict]        = []   # буфер для экспорта датасета
        self._job_id:   str               = ""

    # ── Абстрактный интерфейс ─────────────────────────────────────────────────

    @abstractmethod
    async def harvest(self, source: str) -> AsyncIterator[HarvestedItem]:
        """Yield HarvestedItem из источника."""
        ...

    @abstractmethod
    async def encode(self, content: Any) -> list[float]:
        """Конвертировать контент в feature vector."""
        ...

    # ── Главный метод ─────────────────────────────────────────────────────────

    async def run(self, source: str, job_id: str) -> int:
        """
        Запустить харвест источника.
        Возвращает количество обученных элементов.
        """
        self._job_id = job_id
        count = 0

        try:
            await self._update_job(job_id, "running")

            async for item in self.harvest(source):
                if item.quality < DATA_QUALITY_THRESHOLD:
                    continue
                if item.key in self._seen:
                    continue
                self._seen.add(item.key)

                # Определяем: малый файл → сразу в AI Engine
                #             большой файл → сначала в Storage, потом AI
                raw_size = len(item.raw) if isinstance(item.raw, (str, bytes)) else 0
                if raw_size >= MEDIA_STORAGE_THRESHOLD and isinstance(item.raw, bytes):
                    # Большой медиафайл → Storage
                    storage_path = await self._save_media_to_storage(item, job_id)
                    if storage_path:
                        item.metadata["storage_path"] = storage_path
                    # Отправляем только вектор (без raw bytes)
                    item_for_engine = HarvestedItem(
                        key=item.key, data=item.data, source=item.source,
                        network=item.network, raw=None,
                        quality=item.quality, metadata=item.metadata,
                    )
                    await self._feed_to_engine(item_for_engine)
                else:
                    await self._feed_to_engine(item)

                # Буферизуем для экспорта датасета
                if item.raw and isinstance(item.raw, str):
                    self._dataset.append({
                        "key":     item.key,
                        "raw":     item.raw[:2000],
                        "source":  item.source,
                        "quality": item.quality,
                    })

                count += 1

                if count % 10 == 0:
                    await self._update_job(job_id, "running", count)

                # Экспортируем датасет в training-data bucket каждые N элементов
                if len(self._dataset) >= TRAINING_EXPORT_EVERY:
                    await self._export_training_dataset()

            # Финальный экспорт если есть накопленные элементы
            if self._dataset:
                await self._export_training_dataset()

            await self._update_job(job_id, "completed", count)

        except Exception as exc:
            logger.error("Harvester %s failed: %s", self.network_type, exc)
            await self._update_job(job_id, "failed", count, str(exc))

        return count

    # ── Отправка в AI Engine ──────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(min=1, max=10))
    async def _feed_to_engine(self, item: HarvestedItem):
        """
        POST AI_ENGINE_URL/learn
        AI Engine сохраняет вектор в Li (RAM) → KnowledgeStore → li_knowledge
        """
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            payload = {
                "type":    item.network,
                "key":     item.key,
                "data":    item.data,
                "source":  item.source,
                "quality": item.quality,
            }
            if item.raw is not None:
                raw_str = item.raw if isinstance(item.raw, str) else str(item.raw)
                payload["raw"] = raw_str[:2000]

            resp = await client.post(
                f"{AI_ENGINE_URL}/learn",
                json=payload,
                headers={"x-api-key": AI_ENGINE_API_KEY},
            )
            resp.raise_for_status()

    # ── Supabase Storage: медиафайлы ──────────────────────────────────────────

    async def _save_media_to_storage(self, item: HarvestedItem, job_id: str) -> str | None:
        """
        Сохраняет большой медиафайл в Supabase Storage.

        Куда:
          Bucket: media  (SUPABASE_BUCKET_MEDIA)
          Path:   media/job_{job_id}/{network_type}/{key}.{ext}

        Зачем:
          Бинарные файлы > 512KB (видео, аудио, изображения) не передаются
          напрямую в AI Engine. Они хранятся в Storage, а AI получает только
          числовой вектор + ссылку на файл.

        После записи также обновляет query_attachments таблицу.
        """
        if not isinstance(item.raw, bytes):
            return None

        ext = self._guess_extension(item.metadata.get("content_type", ""))
        path = f"job_{job_id}/{item.network}/{item.key}{ext}"

        try:
            content_type = item.metadata.get("content_type", "application/octet-stream")
            # Supabase Python SDK — синхронный вызов Storage
            res = self.supabase.storage.from_(BUCKET_MEDIA).upload(
                path,
                item.raw,
                file_options={"content-type": content_type, "upsert": "true"},
            )
            logger.info(
                "[%s] Saved to storage: %s/%s (%d bytes)",
                self.network_type, BUCKET_MEDIA, path, len(item.raw),
            )
            return path
        except Exception as e:
            logger.warning("[%s] Storage upload failed: %s", self.network_type, e)
            return None

    @staticmethod
    def _guess_extension(content_type: str) -> str:
        mapping = {
            "image/jpeg":  ".jpg",
            "image/png":   ".png",
            "image/webp":  ".webp",
            "image/gif":   ".gif",
            "video/mp4":   ".mp4",
            "video/webm":  ".webm",
            "audio/mpeg":  ".mp3",
            "audio/wav":   ".wav",
            "audio/ogg":   ".ogg",
        }
        return mapping.get(content_type, ".bin")

    # ── Supabase Storage: экспорт датасета ────────────────────────────────────

    async def _export_training_dataset(self):
        """
        Экспортирует накопленные обучающие данные в training-data bucket.

        Куда:
          Bucket: training-data  (SUPABASE_BUCKET_TRAINING_DATA)
          Path:   training-data/{network_type}/harvest_{timestamp}.jsonl

        Формат JSONL (одна запись = одна строка):
          {"key": "abc123", "raw": "...", "source": "https://...", "quality": 0.9}

        Зачем:
          Эти файлы используются для дообучения и аудита.
          Admins могут скачать, просмотреть и пометить лучшие ответы
          (→ curated_answers таблица).
        """
        if not self._dataset:
            return

        batch = self._dataset.copy()
        self._dataset.clear()

        ts   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        path = f"{self.network_type}/harvest_{ts}_{len(batch)}items.jsonl"

        jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in batch)

        try:
            self.supabase.storage.from_(BUCKET_TRAINING_DATA).upload(
                path,
                jsonl.encode("utf-8"),
                file_options={"content-type": "application/jsonl", "upsert": "true"},
            )
            logger.info(
                "[%s] Training dataset exported → %s/%s (%d items)",
                self.network_type, BUCKET_TRAINING_DATA, path, len(batch),
            )
        except Exception as e:
            logger.warning("[%s] Training export failed: %s", self.network_type, e)
            # Возвращаем данные в буфер при ошибке
            self._dataset.extend(batch)

    # ── Supabase job tracking ─────────────────────────────────────────────────

    async def _update_job(self, job_id: str, status: str, count: int = 0, error: str | None = None):
        update: dict = {"status": status, "items_collected": count}
        if status in ("completed", "failed"):
            update["completed_at"] = datetime.utcnow().isoformat()
        if error:
            update["error"] = error
        try:
            self.supabase.table("harvester_jobs").update(update).eq("id", job_id).execute()
        except Exception as e:
            logger.warning("Failed to update job %s: %s", job_id, e)

    # ── Утилиты ───────────────────────────────────────────────────────────────

    @staticmethod
    def content_hash(content: str | bytes) -> str:
        if isinstance(content, str):
            content = content.encode()
        return hashlib.sha256(content).hexdigest()[:16]

    @staticmethod
    def normalize(vector: list[float]) -> list[float]:
        m = max(abs(v) for v in vector) or 1.0
        return [v / m for v in vector]

    async def get_url(self, url: str) -> httpx.Response:
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            return resp
        except Exception as e:
            logger.warning("GET %s failed: %s", url, e)
            raise RuntimeError(f"HTTP request failed: {e}") from e

    async def close(self):
        await self._client.aclose()

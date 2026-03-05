"""
VideoHarvester — семантическое кодирование видео через CLIP + временные признаки.

Эволюция:
  v1: pixel-level 8×8 thumbnail → 128-dim (не понимает смысл)
  v2: CLIP per-frame + temporal pooling → 512-dim семантический вектор
      "кошка прыгает" ≈ "jumping cat" в пространстве

Принцип работы:
  1. Скачивает видео через yt-dlp
  2. Извлекает до 16 ключевых кадров через ffmpeg
  3. Каждый кадр кодируется через CLIP (512-dim)
  4. Temporal pooling: взвешенное среднее кадров
     (начальные и конечные кадры получают больший вес)
  5. Итог: один 512-dim вектор описывает всё видео семантически

Требования: yt-dlp, ffmpeg, open-clip-torch, Pillow
"""
import io
import logging
import os
import tempfile
from typing import AsyncIterator

from .base import BaseHarvester, HarvestedItem

logger = logging.getLogger(__name__)

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    logger.warning("[VideoNet] yt-dlp not installed. Install: pip install yt-dlp")

# ── Загрузка CLIP (разделяем с ImageHarvester через импорт) ─────────────────
_clip_model = None
_clip_preprocess = None

def _get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            import open_clip
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            _clip_model.eval()
            logger.info("[VideoNet] CLIP ViT-B/32 loaded for video frames")
        except ImportError:
            logger.warning("[VideoNet] open-clip-torch not installed — using pixel fallback")
            _clip_model = "fallback"
    return _clip_model, _clip_preprocess


class VideoHarvester(BaseHarvester):
    """
    Загружает видео, извлекает ключевые кадры, кодирует через CLIP.
    Temporal pooling объединяет кадры в один семантический вектор.
    """
    network_type = "video"
    MAX_FRAMES   = 16          # максимум кадров для анализа
    FRAME_FPS    = 1           # 1 кадр в секунду

    async def harvest(self, source: str) -> AsyncIterator[HarvestedItem]:
        if not YTDLP_AVAILABLE:
            logger.warning("[VideoNet] yt-dlp required for video harvesting")
            return

        frames, metadata = await self._extract_frames(source)
        if not frames:
            logger.warning("[VideoNet] No frames extracted from %s", source)
            return

        key    = self.content_hash(source)
        vector = await self.encode(frames)

        yield HarvestedItem(
            key=key,
            data=vector,
            source=source,
            network=self.network_type,
            raw=metadata.get("title", source)[:500],
            quality=0.85,
            metadata={
                "frame_count": len(frames),
                "dim":         len(vector),
                **metadata,
            },
        )

    async def encode(self, frames: list[bytes]) -> list[float]:
        """
        Кодирует список кадров → единый семантический вектор.

        CLIP: каждый кадр → 512-dim
        Temporal pooling: взвешенное среднее
          - первый и последний кадры × 1.5 (важны для контекста)
          - средние кадры × 1.0
        """
        model, preprocess = _get_clip()

        if model != "fallback" and preprocess is not None:
            return await self._clip_encode(frames, model, preprocess)

        return self._temporal_histogram(frames)

    async def _clip_encode(self, frames: list[bytes], model, preprocess) -> list[float]:
        """Кодирование через CLIP с temporal pooling."""
        import torch
        from PIL import Image

        frame_vectors = []
        for frame_bytes in frames:
            try:
                img    = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                tensor = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    feat = model.encode_image(tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                frame_vectors.append(feat[0].tolist())
            except Exception as e:
                logger.debug("[VideoNet] Frame encode failed: %s", e)
                continue

        if not frame_vectors:
            return [0.0] * 512

        n = len(frame_vectors)
        dim = len(frame_vectors[0])

        # Temporal pooling с вниманием к крайним кадрам
        weights = [1.0] * n
        if n >= 2:
            weights[0]  = 1.5   # первый кадр (начало сцены)
            weights[-1] = 1.5   # последний кадр (конец сцены)
        total_weight = sum(weights)

        pooled = [0.0] * dim
        for i, vec in enumerate(frame_vectors):
            w = weights[i] / total_weight
            for j in range(dim):
                pooled[j] += vec[j] * w

        # L2 normalize
        norm = sum(v*v for v in pooled) ** 0.5
        if norm > 1e-8:
            pooled = [v / norm for v in pooled]

        return pooled

    def _temporal_histogram(self, frames: list[bytes]) -> list[float]:
        """Fallback: пространственно-временная гистограмма."""
        try:
            from PIL import Image
        except ImportError:
            dim = 128
            vec = [0.0] * dim
            for fi, frame in enumerate(frames[:16]):
                w = (fi + 1) / len(frames)
                for i, byte in enumerate(frame[:dim]):
                    vec[i % dim] += (byte / 255) * w
            return self.normalize(vec)

        dim = 128
        vec = [0.0] * dim
        n   = min(len(frames), 16)

        for f_idx, frame in enumerate(frames[:n]):
            weight = (f_idx + 1) / n
            try:
                img = Image.open(io.BytesIO(frame)).convert("RGB").resize((8, 8))
                pixels = list(img.getdata())
                for i, (r, g, b) in enumerate(pixels):
                    vec[i % dim]            += r / 255 * weight
                    vec[(i + 21) % dim]     += g / 255 * weight
                    vec[(i + 42) % dim]     += b / 255 * weight
            except Exception:
                for i, byte in enumerate(frame[:dim]):
                    vec[i % dim] += (byte / 255) * weight

        return self.normalize(vec)

    async def _extract_frames(self, url: str) -> tuple[list[bytes], dict]:
        """Скачивает видео через yt-dlp, извлекает кадры через ffmpeg."""
        frames: list[bytes] = []
        metadata: dict = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            opts = {
                "format":       "worst[ext=mp4]/worst/bestvideo[height<=480]",
                "outtmpl":      os.path.join(tmpdir, "video.%(ext)s"),
                "quiet":        True,
                "no_warnings":  True,
                "max_filesize": "100M",
                "writeinfojson": True,
                "skip_download": False,
            }
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if info:
                        metadata = {
                            "title":       info.get("title", ""),
                            "duration":    info.get("duration", 0),
                            "description": (info.get("description") or "")[:200],
                            "uploader":    info.get("uploader", ""),
                        }

                video_file = next(
                    (os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
                     if f.startswith("video") and not f.endswith(".json")),
                    None,
                )
                if not video_file:
                    return frames, metadata

                frames_dir = os.path.join(tmpdir, "frames")
                os.makedirs(frames_dir, exist_ok=True)

                # Извлечь кадры с fps=1, максимум MAX_FRAMES
                cmd = (
                    f'ffmpeg -i "{video_file}" '
                    f'-vf "fps={self.FRAME_FPS},scale=224:224:force_original_aspect_ratio=decrease" '
                    f'-vframes {self.MAX_FRAMES} '
                    f'"{frames_dir}/frame%04d.jpg" '
                    f'-hide_banner -loglevel quiet'
                )
                os.system(cmd)

                for fname in sorted(os.listdir(frames_dir))[:self.MAX_FRAMES]:
                    fpath = os.path.join(frames_dir, fname)
                    with open(fpath, "rb") as f:
                        frames.append(f.read())

            except Exception as e:
                logger.warning("[VideoNet] Extraction failed for %s: %s", url, e)

        return frames, metadata

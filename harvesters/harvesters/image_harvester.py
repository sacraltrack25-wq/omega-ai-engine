"""
ImageHarvester — семантическое кодирование изображений через CLIP.

Эволюция энкодеров:
  v1 (старый): цветовая гистограмма 128-dim → не понимает смысл
  v2 (новый):  CLIP ViT-B/32 → 512-dim семантический вектор
               "кот" и "feline" будут рядом в пространстве.

CLIP (Contrastive Language-Image Pre-training):
  - Обучен сопоставлять изображения и текст
  - Один вектор описывает И визуальный контент И смысл
  - Позволяет искать изображения по текстовому описанию и наоборот
  - Та же 512-dim модель используется для текстовых описаний изображений

Загрузка: pip install open-clip-torch (~350MB)
Fallback: Pillow гистограмма 128-dim если CLIP недоступен
"""
import io
import logging
from typing import AsyncIterator

from .base import BaseHarvester, HarvestedItem

logger = logging.getLogger(__name__)

# ── Загрузка CLIP один раз (кэш в RAM) ───────────────────────────────────────
_clip_model = None
_clip_preprocess = None
_clip_tokenizer  = None

def _get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            import open_clip
            import torch
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            _clip_model.eval()
            logger.info("[ImageNet] CLIP ViT-B/32 loaded — 512-dim semantic vectors")
        except ImportError:
            logger.warning("[ImageNet] open-clip-torch not installed — using histogram fallback")
            _clip_model = "fallback"
    return _clip_model, _clip_preprocess


class ImageHarvester(BaseHarvester):
    """
    Скрапит изображения и кодирует через CLIP → 512-dim семантические векторы.
    Работает с прямыми URL изображений или HTML страницами.
    """
    network_type = "image"

    async def harvest(self, source: str) -> AsyncIterator[HarvestedItem]:
        resp = await self.get_url(source)
        content_type = resp.headers.get("content-type", "")

        if "image/" in content_type:
            item = await self._process_image(resp.content, source, source)
            if item:
                yield item
        elif "text/html" in content_type:
            async for item in self._harvest_page_images(resp.text, source):
                yield item

    async def _harvest_page_images(self, html: str, base_url: str) -> AsyncIterator[HarvestedItem]:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        img_tags = soup.find_all("img", src=True)

        for tag in img_tags[:30]:  # увеличили с 20 до 30
            src = str(tag.get("src", ""))
            if not src.startswith("http"):
                continue
            try:
                img_resp = await self.get_url(src)
            except Exception:
                continue
            alt  = str(tag.get("alt", ""))
            item = await self._process_image(img_resp.content, src, base_url, alt=alt)
            if item:
                yield item

    async def _process_image(
        self, content: bytes, source: str, page_url: str, alt: str = ""
    ) -> HarvestedItem | None:
        if len(content) < 1000:  # слишком маленький файл
            return None

        key    = self.content_hash(content)
        vector = await self.encode(content)
        raw    = alt or source.split("/")[-1]

        return HarvestedItem(
            key=key,
            data=vector,
            source=source,
            network=self.network_type,
            raw=raw[:500] if raw else None,
            quality=0.85,
            metadata={"alt": alt, "page": page_url, "size": len(content), "dim": len(vector)},
        )

    async def encode(self, content: bytes) -> list[float]:
        """
        Кодирует изображение → 512-dim CLIP вектор.
        Fallback: улучшенная гистограмма 256-dim.
        """
        model, preprocess = _get_clip()

        if model != "fallback" and preprocess is not None:
            try:
                import torch
                from PIL import Image
                img = Image.open(io.BytesIO(content)).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0)

                with torch.no_grad():
                    features = model.encode_image(img_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

                return features[0].tolist()
            except Exception as e:
                logger.warning("[ImageNet] CLIP encode failed: %s", e)

        # Fallback: улучшенная гистограмма с пространственным разделением
        return self._spatial_histogram(content)

    def _spatial_histogram(self, content: bytes) -> list[float]:
        """
        Fallback: пространственная гистограмма 256-dim.
        Делит изображение на 4 квадранта, строит гистограмму для каждого.
        Лучше чем глобальная гистограмма — учитывает расположение цветов.
        """
        dim = 256
        vec = [0.0] * dim

        try:
            from PIL import Image
            img = Image.open(io.BytesIO(content)).convert("RGB")
            w, h = img.size
            # 4 квадранта × 64-dim = 256-dim вектор
            quadrants = [
                img.crop((0, 0, w//2, h//2)),     # верхний левый
                img.crop((w//2, 0, w, h//2)),     # верхний правый
                img.crop((0, h//2, w//2, h)),     # нижний левый
                img.crop((w//2, h//2, w, h)),     # нижний правый
            ]
            for qi, quad in enumerate(quadrants):
                quad_small = quad.resize((8, 8))
                pixels = list(quad_small.getdata())
                offset = qi * 64
                bins   = 64 // 3
                for r, g, b in pixels:
                    vec[offset + (r * bins // 256)]         += 1.0
                    vec[offset + bins + (g * bins // 256)]  += 1.0
                    vec[offset + 2*bins + (b * bins // 256)] += 1.0
            return self.normalize(vec)
        except Exception:
            # Byte histogram fallback
            for i, byte in enumerate(content[:4096]):
                vec[byte % dim] += 1.0
            return self.normalize(vec)

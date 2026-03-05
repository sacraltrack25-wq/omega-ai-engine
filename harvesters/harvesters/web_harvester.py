"""
WebHarvester — scrapes and semantically encodes text for TextNet.

Encoder: два многоязычных модели (RU+EN), векторы конкатенируются:
  - paraphrase-multilingual-MiniLM-L12-v2  → 384-dim
  - distiluse-base-multilingual-cased-v2   → 512-dim
  Итого: 896-dim семантический вектор

Falls back to TF-IDF hashing if sentence-transformers unavailable.
"""
import logging
import re
import hashlib
from typing import AsyncIterator

from bs4 import BeautifulSoup

from .base import BaseHarvester, HarvestedItem

logger = logging.getLogger(__name__)

# ── Load sentence-transformers models (cached in RAM) ──────────────────────────
_models = None

def _get_models():
    global _models
    if _models is None:
        try:
            from sentence_transformers import SentenceTransformer
            from config import TEXT_EMBEDDING_MODELS
            names = TEXT_EMBEDDING_MODELS
            _models = [SentenceTransformer(n) for n in names]
            total_dim = sum(m.get_sentence_embedding_dimension() for m in _models)
            logger.info("[TextNet] Semantic encoder loaded: %s (total %d-dim)", ", ".join(names), total_dim)
        except ImportError:
            logger.warning("[TextNet] sentence-transformers not installed — using TF-IDF fallback")
            _models = "fallback"
    return _models

STOP_WORDS = {
    "the","a","an","is","it","in","on","of","to","and","or","for",
    "with","that","this","are","was","were","be","been","have","has",
    "do","does","did","will","would","could","should","may","might",
    "at","by","from","into","through","during","before","after",
    "about","above","below","between","out","up","down","so","if",
    "but","not","no","as","than","then","when","where","who","which",
}


class WebHarvester(BaseHarvester):
    """
    Crawls web pages and encodes text using semantic sentence embeddings.
    Output: 384-dim semantic vectors for TextNet Li clusters.
    """
    network_type = "text"

    async def harvest(self, source: str) -> AsyncIterator[HarvestedItem]:
        resp = await self.get_url(source)
        logger.info("[WebHarvester] GET %s: status=%d, len=%d", source[:60], resp.status_code, len(resp.text))
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                          "form", "button", "iframe", "noscript"]):
            tag.decompose()

        # Wikipedia: контент в #mw-content-text или .mw-parser-output
        root = soup
        if "wikipedia.org" in source:
            root = soup.find(id="mw-content-text") or soup.find(class_="mw-parser-output") or soup
        elif "wikimedia.org" in source or "wiktionary.org" in source:
            root = soup.find(id="mw-content-text") or soup.find(class_="mw-parser-output") or soup

        # Extract meaningful text blocks
        blocks = root.find_all(["p", "article", "section", "h1", "h2", "h3",
                                 "h4", "li", "blockquote", "td"])

        if not blocks:
            logger.warning("[WebHarvester] No text blocks found for %s (len=%d, root=%s)", source[:60], len(resp.text), root.name if hasattr(root, 'name') else '?')

        # Group small blocks into chunks of ~256 words for better embeddings
        buffer = []
        buffer_len = 0

        for block in blocks:
            text = block.get_text(separator=" ").strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) < 30:
                continue

            words = text.split()
            buffer.append(text)
            buffer_len += len(words)

            # Flush when chunk is big enough
            if buffer_len >= 200:
                chunk = " ".join(buffer)
                item = await self._make_item(chunk, source, block.name)
                if item:
                    yield item
                buffer = []
                buffer_len = 0

        # Flush remainder
        if buffer:
            chunk = " ".join(buffer)
            item = await self._make_item(chunk, source, "text")
            if item:
                yield item

        logger.info("[WebHarvester] harvest done: %d blocks parsed", len(blocks))

    async def _make_item(self, text: str, source: str, tag: str):
        quality = self._text_quality(text)
        if quality < 0.3:
            return None

        key    = self.content_hash(text)
        vector = await self.encode(text)

        return HarvestedItem(
            key=key,
            data=vector,
            source=source,
            network=self.network_type,
            raw=text[:500],
            quality=quality,
            metadata={"tag": tag, "length": len(text), "dim": len(vector)},
        )

    async def encode(self, content: str) -> list[float]:
        """
        Semantic encoding via sentence-transformers (оба многоязычных модели).
        Конкатенирует векторы обоих моделей → L2 normalize.
        Returns 896-dim (384+512) normalized vector.
        Falls back to TF-IDF hashing (256-dim) if models unavailable.
        """
        models = _get_models()

        if models != "fallback":
            import numpy as np
            parts = []
            for model in models:
                emb = model.encode(content[:512], normalize_embeddings=True)
                parts.append(emb)
            concat = np.concatenate(parts)
            # L2 normalize combined vector
            norm = np.linalg.norm(concat)
            if norm > 1e-8:
                concat = concat / norm
            return concat.tolist()
        else:
            return self._tfidf_encode(content)

    def _tfidf_encode(self, text: str) -> list[float]:
        """Fallback: improved n-gram hashing to 256-dim vector."""
        dim   = 256
        vec   = [0.0] * dim
        text  = text.lower()[:8192]
        words = re.findall(r"\b\w+\b", text)

        # Word-level unigrams (weighted by inverse stop-word penalty)
        for word in words:
            weight = 0.3 if word in STOP_WORDS else 1.0
            h = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
            vec[h] += weight

        # Bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.md5(bigram.encode()).hexdigest(), 16) % dim
            vec[h] += 0.6

        # Trigrams
        for i in range(len(words) - 2):
            trigram = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            h = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % dim
            vec[h] += 0.4

        return self.normalize(vec)

    def _text_quality(self, text: str) -> float:
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0
        meaningful    = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        content_ratio = len(meaningful) / max(len(words), 1)
        length_score  = min(1.0, len(text) / 800)
        # Penalize repetitive text
        unique_ratio  = len(set(words)) / max(len(words), 1)
        return content_ratio * 0.5 + length_score * 0.3 + unique_ratio * 0.2

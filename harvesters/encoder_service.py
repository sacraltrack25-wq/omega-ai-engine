"""
OMEGA Encoder Service — единая точка энкодирования для core.
============================================================
Предоставляет векторы для text, image, video, audio в форматах,
совместимых с харвестерами и Li центрами.

Endpoints:
  POST /encode           — text → sentence-transformers (896-dim)
  POST /encode-clip-text — text → CLIP 512-dim (для image/video Li)
  POST /encode-clip-image — image base64 → CLIP 512-dim
  POST /encode-clip-frames — video frames base64[] → CLIP pooled 512-dim
  POST /encode-clap-text — text → CLAP (для audio Li)
  POST /encode-clap-audio — audio base64 → CLAP
  POST /transcribe — audio base64/url → text (Whisper)
  POST /caption-image — image base64/url → text (BLIP)

Запуск: uvicorn encoder_service:app --port 5001
"""
import base64
import io
import logging
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="OMEGA Encoder Service", version="1.0")

# ── Lazy-loaded models ───────────────────────────────────────────────────────

_st_models = None
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clap_model = None
_whisper_model = None
_blip_model = None


def _get_sentence_transformers():
    """Load sentence-transformers for text encoding."""
    global _st_models
    if _st_models is None:
        try:
            from sentence_transformers import SentenceTransformer
            import os
            _harvesters_dir = os.path.dirname(os.path.abspath(__file__))
            if _harvesters_dir not in sys.path:
                sys.path.insert(0, _harvesters_dir)
            from config import TEXT_EMBEDDING_MODELS
            _st_models = [SentenceTransformer(n) for n in TEXT_EMBEDDING_MODELS]
            dims = sum(m.get_sentence_embedding_dimension() for m in _st_models)
            logger.info("Loaded sentence-transformers: %s (total %d-dim)", TEXT_EMBEDDING_MODELS, dims)
        except Exception as e:
            logger.warning("sentence-transformers load failed: %s", e)
            _st_models = []
    return _st_models


def _get_clip():
    """Load CLIP ViT-B/32 for image/text encoding."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        try:
            import open_clip
            import torch
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            _clip_model.eval()
            logger.info("[CLIP] ViT-B/32 loaded — 512-dim")
        except Exception as e:
            logger.warning("[CLIP] load failed: %s", e)
            _clip_model = "fallback"
    return _clip_model, _clip_preprocess, _clip_tokenizer


def _get_clap():
    """Load CLAP for text/audio encoding."""
    global _clap_model
    if _clap_model is None:
        try:
            import laion_clap
            _clap_model = laion_clap.CLAP_Module(enable_fusion=False)
            _clap_model.load_ckpt()
            logger.info("[CLAP] loaded — text/audio embeddings")
        except Exception as e:
            logger.warning("[CLAP] load failed: %s", e)
            _clap_model = "fallback"
    return _clap_model


def _get_whisper():
    """Load Whisper for speech-to-text."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            _whisper_model = whisper.load_model("base")
            logger.info("[Whisper] base model loaded")
        except Exception as e:
            logger.warning("[Whisper] load failed: %s", e)
            _whisper_model = "fallback"
    return _whisper_model


def _get_blip():
    """Load BLIP for image captioning."""
    global _blip_model
    if _blip_model is None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            _blip_model = (
                BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
                BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base"),
            )
            logger.info("[BLIP] loaded for image captioning")
        except Exception as e:
            logger.warning("[BLIP] load failed: %s", e)
            _blip_model = "fallback"
    return _blip_model


# ── Request/Response models ───────────────────────────────────────────────────

class EncodeTextRequest(BaseModel):
    text: str


class EncodeImageRequest(BaseModel):
    data: str  # base64
    url: Optional[str] = None  # alternative: fetch from URL


class EncodeFramesRequest(BaseModel):
    frames: list[str]  # base64 list


class EncodeAudioRequest(BaseModel):
    data: Optional[str] = None  # base64
    url: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_concat(parts: list) -> list[float]:
    import numpy as np
    concat = np.concatenate(parts)
    norm = np.linalg.norm(concat)
    if norm > 1e-8:
        concat = concat / norm
    return concat.tolist()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "OMEGA Encoder Service",
        "version": "1.0",
        "endpoints": [
            "POST /encode",
            "POST /encode-clip-text",
            "POST /encode-clip-image",
            "POST /encode-clip-frames",
            "POST /encode-clap-text",
            "POST /encode-clap-audio",
            "POST /transcribe",
            "POST /caption-image",
        ],
    }


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/encode")
async def encode_text(req: EncodeTextRequest):
    """Text → sentence-transformers vector (896-dim by default)."""
    models = _get_sentence_transformers()
    if not models:
        raise HTTPException(503, "sentence-transformers not available")

    try:
        import numpy as np
        text = req.text[:512]
        parts = [m.encode(text, normalize_embeddings=True) for m in models]
        vec = _norm_concat(parts)
        return {"vector": vec, "dim": len(vec)}
    except Exception as e:
        logger.exception("encode failed")
        raise HTTPException(500, str(e))


@app.post("/encode-clip-text")
async def encode_clip_text(req: EncodeTextRequest):
    """Text → CLIP 512-dim (for image/video Li cross-modal search)."""
    model, preprocess, tokenizer = _get_clip()
    if model == "fallback":
        raise HTTPException(503, "CLIP not available")

    try:
        import torch
        tokens = tokenizer([req.text[:77]])  # CLIP max tokens
        with torch.no_grad():
            feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return {"vector": feat[0].tolist(), "dim": len(feat[0].tolist())}
    except Exception as e:
        logger.exception("encode-clip-text failed")
        raise HTTPException(500, str(e))


@app.post("/encode-clip-image")
async def encode_clip_image(req: EncodeImageRequest):
    """Image base64 or URL → CLIP 512-dim."""
    model, preprocess, _ = _get_clip()
    if model == "fallback":
        raise HTTPException(503, "CLIP not available")

    content = None
    if req.data:
        try:
            content = base64.b64decode(req.data)
        except Exception as e:
            raise HTTPException(400, f"Invalid base64: {e}")
    elif req.url:
        try:
            import httpx
            import os
            _hd = os.path.dirname(os.path.abspath(__file__))
            if _hd not in sys.path:
                sys.path.insert(0, _hd)
            try:
                from config import REQUEST_TIMEOUT, USER_AGENT
            except ImportError:
                REQUEST_TIMEOUT = 30
                USER_AGENT = "OMEGA-Encoder/1.0"
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.get(req.url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
                content = r.content
        except Exception as e:
            raise HTTPException(400, f"Failed to fetch URL: {e}")
    else:
        raise HTTPException(400, "data or url required")

    if not content or len(content) < 100:
        raise HTTPException(400, "Image too small or empty")

    try:
        import torch
        from PIL import Image
        img = Image.open(io.BytesIO(content)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return {"vector": feat[0].tolist(), "dim": len(feat[0].tolist())}
    except Exception as e:
        logger.exception("encode-clip-image failed")
        raise HTTPException(500, str(e))


@app.post("/encode-clip-frames")
async def encode_clip_frames(req: EncodeFramesRequest):
    """Video frames base64[] → CLIP temporal pooled 512-dim."""
    model, preprocess, _ = _get_clip()
    if model == "fallback":
        raise HTTPException(503, "CLIP not available")

    if not req.frames:
        raise HTTPException(400, "frames required")

    try:
        import torch
        from PIL import Image

        frame_vectors = []
        for b64 in req.frames[:16]:
            try:
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                tensor = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    feat = model.encode_image(tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                frame_vectors.append(feat[0].tolist())
            except Exception as e:
                logger.debug("Frame decode failed: %s", e)
                continue

        if not frame_vectors:
            return {"vector": [0.0] * 512, "dim": 512}

        n = len(frame_vectors)
        dim = len(frame_vectors[0])
        weights = [1.0] * n
        if n >= 2:
            weights[0] = 1.5
            weights[-1] = 1.5
        total_w = sum(weights)

        pooled = [0.0] * dim
        for i, vec in enumerate(frame_vectors):
            w = weights[i] / total_w
            for j in range(dim):
                pooled[j] += vec[j] * w

        norm = sum(v * v for v in pooled) ** 0.5
        if norm > 1e-8:
            pooled = [v / norm for v in pooled]

        return {"vector": pooled, "dim": dim}
    except Exception as e:
        logger.exception("encode-clip-frames failed")
        raise HTTPException(500, str(e))


@app.post("/encode-clap-text")
async def encode_clap_text(req: EncodeTextRequest):
    """Text → CLAP vector (for audio Li cross-modal search)."""
    clap = _get_clap()
    if clap == "fallback":
        raise HTTPException(503, "CLAP not available")

    try:
        emb = clap.get_text_embedding([req.text[:512]], use_tensor=False)
        vec = emb[0].tolist() if hasattr(emb[0], "tolist") else list(emb[0])
        return {"vector": vec, "dim": len(vec)}
    except Exception as e:
        logger.exception("encode-clap-text failed")
        raise HTTPException(500, str(e))


@app.post("/encode-clap-audio")
async def encode_clap_audio(req: EncodeAudioRequest):
    """Audio base64 or URL → CLAP vector."""
    clap = _get_clap()
    if clap == "fallback":
        raise HTTPException(503, "CLAP not available")

    content = await _get_audio_content_async(req)
    if not content:
        raise HTTPException(400, "data or url required")

    try:
        import tempfile
        import os

        # CLAP expects file path; convert to wav if needed via pydub
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(content)
            raw_path = f.name

        path = raw_path
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(io.BytesIO(content))
            audio = audio.set_frame_rate(48000).set_channels(1)
            wav_path = raw_path + ".wav"
            audio.export(wav_path, format="wav")
            path = wav_path
        except Exception:
            path = raw_path

        try:
            emb = clap.get_audio_embedding_from_filelist(x=[path], use_tensor=False)
            vec = emb[0].tolist() if hasattr(emb[0], "tolist") else list(emb[0])
            return {"vector": vec, "dim": len(vec)}
        finally:
            import os
            to_remove = [p for p in [path, raw_path] if p and os.path.exists(p)]
            for p in set(to_remove):
                try:
                    os.unlink(p)
                except Exception:
                    pass
    except Exception as e:
        logger.exception("encode-clap-audio failed")
        raise HTTPException(500, str(e))


async def _get_audio_content_async(req: EncodeAudioRequest) -> Optional[bytes]:
    """Extract audio bytes from request."""
    if req.data:
        try:
            return base64.b64decode(req.data)
        except Exception:
            return None
    if req.url:
        try:
            import httpx
            import os
            _hd = os.path.dirname(os.path.abspath(__file__))
            if _hd not in sys.path:
                sys.path.insert(0, _hd)
            try:
                from config import REQUEST_TIMEOUT, USER_AGENT
            except ImportError:
                REQUEST_TIMEOUT, USER_AGENT = 30, "OMEGA-Encoder/1.0"
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.get(req.url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
                return r.content
        except Exception:
            return None
    return None


@app.post("/transcribe")
async def transcribe(req: EncodeAudioRequest):
    """Audio base64 or URL → text (Whisper)."""
    model = _get_whisper()
    if model == "fallback":
        raise HTTPException(503, "Whisper not available")

    content = None
    if req.data:
        try:
            content = base64.b64decode(req.data)
        except Exception:
            raise HTTPException(400, "Invalid base64")
    elif req.url:
        try:
            import httpx
            import os
            _hd = os.path.dirname(os.path.abspath(__file__))
            if _hd not in sys.path:
                sys.path.insert(0, _hd)
            try:
                from config import REQUEST_TIMEOUT, USER_AGENT
            except ImportError:
                REQUEST_TIMEOUT, USER_AGENT = 30, "OMEGA-Encoder/1.0"
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(req.url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
                content = r.content
        except Exception as e:
            raise HTTPException(400, f"Failed to fetch URL: {e}")
    else:
        raise HTTPException(400, "data or url required")

    if not content or len(content) < 100:
        raise HTTPException(400, "Audio too small")

    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            result = model.transcribe(path, language=None, fp16=False)
            text = (result.get("text") or "").strip()
            return {"text": text}
        finally:
            import os
            os.unlink(path)
    except Exception as e:
        logger.exception("transcribe failed")
        raise HTTPException(500, str(e))


@app.post("/caption-image")
async def caption_image(req: EncodeImageRequest):
    """Image base64 or URL → text caption (BLIP)."""
    blip = _get_blip()
    if blip == "fallback":
        raise HTTPException(503, "BLIP not available")

    content = None
    if req.data:
        try:
            content = base64.b64decode(req.data)
        except Exception:
            raise HTTPException(400, "Invalid base64")
    elif req.url:
        try:
            import httpx
            import os
            _hd = os.path.dirname(os.path.abspath(__file__))
            if _hd not in sys.path:
                sys.path.insert(0, _hd)
            try:
                from config import REQUEST_TIMEOUT, USER_AGENT
            except ImportError:
                REQUEST_TIMEOUT, USER_AGENT = 30, "OMEGA-Encoder/1.0"
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.get(req.url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
                content = r.content
        except Exception as e:
            raise HTTPException(400, f"Failed to fetch URL: {e}")
    else:
        raise HTTPException(400, "data or url required")

    if not content or len(content) < 100:
        raise HTTPException(400, "Image too small")

    try:
        from PIL import Image
        import torch
        processor, model = blip
        img = Image.open(io.BytesIO(content)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50)
        text = processor.decode(out[0], skip_special_tokens=True).strip()
        return {"text": text}
    except Exception as e:
        logger.exception("caption-image failed")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)

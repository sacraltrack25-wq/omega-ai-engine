"""AudioHarvester — encodes audio files for AudioNet.

Encoding: CLAP (when available) for text↔audio cross-modal search.
Fallback: 128-dim FFT frequency-band vector.
"""
import io
import logging
import math
import os
import tempfile
from typing import AsyncIterator

from .base import BaseHarvester, HarvestedItem

logger = logging.getLogger(__name__)

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

_clap_model = None


def _get_clap():
    """Load CLAP for text/audio embeddings. Shared with encoder_service."""
    global _clap_model
    if _clap_model is None:
        try:
            import laion_clap
            _clap_model = laion_clap.CLAP_Module(enable_fusion=False)
            _clap_model.load_ckpt()
            logger.info("[AudioHarvester] CLAP loaded — semantic audio vectors")
        except Exception as e:
            logger.warning("[AudioHarvester] CLAP not available: %s", e)
            _clap_model = "fallback"
    return _clap_model


class AudioHarvester(BaseHarvester):
    """Download and encode audio content for AudioNet."""
    network_type = "audio"

    AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}

    async def harvest(self, source: str) -> AsyncIterator[HarvestedItem]:
        if any(source.lower().endswith(ext) for ext in self.AUDIO_EXTS):
            resp = await self.get_url(source)
            key    = self.content_hash(resp.content)
            vector = await self.encode(resp.content)
            yield HarvestedItem(
                key=key, data=vector, source=source,
                network=self.network_type, quality=0.85,
                raw=source.split("/")[-1][:500],
                metadata={"size": len(resp.content)},
            )

    async def encode(self, content: bytes) -> list[float]:
        """Encode audio → CLAP vector (for cross-modal) or 128-dim FFT fallback."""
        clap = _get_clap()
        if clap != "fallback":
            try:
                return await self._encode_clap(content)
            except Exception as e:
                logger.warning("CLAP encode failed: %s", e)

        return self._encode_fft(content)

    async def _encode_clap(self, content: bytes) -> list[float]:
        """CLAP encoding — requires 48kHz wav."""
        clap = _get_clap()
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(content)
            raw_path = f.name

        path = raw_path
        try:
            if PYDUB_AVAILABLE:
                audio = AudioSegment.from_file(io.BytesIO(content))
                audio = audio.set_frame_rate(48000).set_channels(1)
                wav_path = raw_path + ".wav"
                audio.export(wav_path, format="wav")
                path = wav_path

            emb = clap.get_audio_embedding_from_filelist(x=[path], use_tensor=False)
            vec = emb[0].tolist() if hasattr(emb[0], "tolist") else list(emb[0])
            return self.normalize(vec)
        finally:
            for p in [path, raw_path]:
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

    def _encode_fft(self, content: bytes) -> list[float]:
        """Fallback: 128-dim FFT frequency-band vector."""
        dim = 128
        vec = [0.0] * dim

        if PYDUB_AVAILABLE:
            try:
                audio  = AudioSegment.from_file(io.BytesIO(content))
                samples = audio.get_array_of_samples()
                n = min(len(samples), 8192)
                bands = dim

                for k in range(bands):
                    real = sum(samples[i] / 32768 * math.cos(2 * math.pi * k * i / n)
                               for i in range(n))
                    imag = sum(samples[i] / 32768 * math.sin(2 * math.pi * k * i / n)
                               for i in range(n))
                    vec[k] = math.sqrt(real**2 + imag**2) / n
                return self.normalize(vec)
            except Exception as e:
                logger.warning("pydub encode failed: %s", e)

        for i, byte in enumerate(content[:4096]):
            vec[byte % dim] += 1.0
        return self.normalize(vec)

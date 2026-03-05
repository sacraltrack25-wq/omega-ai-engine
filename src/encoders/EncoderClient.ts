/**
 * EncoderClient — HTTP client for OMEGA Encoder Service
 *
 * Core calls the encoder service to get vectors for text, image, video, audio.
 * Fallback: n-gram hash for text when service is unavailable.
 */

const ENCODER_URL = process.env.ENCODER_SERVICE_URL ?? "http://localhost:5001";
const TIMEOUT_MS = 30_000;

async function post<T>(path: string, body: unknown): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(`${ENCODER_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Encoder ${path}: ${res.status} ${err}`);
    }

    return (await res.json()) as T;
  } catch (e) {
    clearTimeout(timeout);
    throw e;
  }
}

/** Text → sentence-transformers vector (896-dim). Used for TextNet. */
export async function encodeText(text: string): Promise<number[]> {
  try {
    const r = await post<{ vector: number[] }>("/encode", { text });
    return r.vector;
  } catch (e) {
    console.warn("[EncoderClient] encode failed, using n-gram fallback:", String(e));
    return nGramFallback(text);
  }
}

/** Text → CLIP 512-dim. For image/video Li cross-modal search. */
export async function encodeClipText(text: string): Promise<number[]> {
  const r = await post<{ vector: number[] }>("/encode-clip-text", { text });
  return r.vector;
}

/** Image base64 → CLIP 512-dim. */
export async function encodeClipImage(data: string): Promise<number[]> {
  const r = await post<{ vector: number[] }>("/encode-clip-image", { data });
  return r.vector;
}

/** Image URL → CLIP 512-dim. */
export async function encodeClipImageFromUrl(url: string): Promise<number[]> {
  const r = await post<{ vector: number[] }>("/encode-clip-image", { url });
  return r.vector;
}

/** Video frames base64[] → CLIP pooled 512-dim. */
export async function encodeClipFrames(frames: string[]): Promise<number[]> {
  const r = await post<{ vector: number[] }>("/encode-clip-frames", { frames });
  return r.vector;
}

/** Text → CLAP vector. For audio Li cross-modal search. */
export async function encodeClapText(text: string): Promise<number[]> {
  const r = await post<{ vector: number[] }>("/encode-clap-text", { text });
  return r.vector;
}

/** Audio base64 → CLAP vector. */
export async function encodeClapAudio(data: string): Promise<number[]> {
  const r = await post<{ vector: number[] }>("/encode-clap-audio", { data });
  return r.vector;
}

/** Audio base64 or URL → transcribed text. */
export async function transcribe(data?: string, url?: string): Promise<string> {
  const r = await post<{ text: string }>("/transcribe", { data, url });
  return r.text ?? "";
}

/** Image base64 or URL → caption text. */
export async function captionImage(data?: string, url?: string): Promise<string> {
  const r = await post<{ text: string }>("/caption-image", { data, url });
  return r.text ?? "";
}

/** Fallback when encoder service is down — n-gram hash 128-dim. */
function nGramFallback(text: string, vecDim = 128): number[] {
  const t = text.slice(0, 4096);
  const vec = new Array<number>(vecDim).fill(0);
  const bytes = Buffer.from(t, "utf8");

  for (const byte of bytes) {
    vec[byte % vecDim] += 1;
  }
  for (let i = 0; i < bytes.length - 1; i++) {
    const h = (bytes[i] * 31 + bytes[i + 1]) % vecDim;
    vec[h] += 0.5;
  }

  const max = Math.max(...vec, 1);
  return vec.map((v) => v / max);
}

/** Check if encoder service is reachable. */
export async function isEncoderAvailable(): Promise<boolean> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 2000);
    const res = await fetch(`${ENCODER_URL}/health`, { signal: ctrl.signal });
    clearTimeout(t);
    return res.ok;
  } catch {
    return false;
  }
}

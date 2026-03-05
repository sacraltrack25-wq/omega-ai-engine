/**
 * AI Engine HTTP Server
 * Exposes the gX-Li-Ω system via REST API
 * Port: 4000 (configurable via PORT env)
 */

import express    from "express";
import cors       from "cors";
import "dotenv/config";

import { mirrorAI } from "./MirrorAI";
import { knowledgeStore } from "./persistence/KnowledgeStore";
import { NetworkType } from "./centers/Li";

const app  = express();
const PORT = Number(process.env.PORT ?? 4000);

app.use(cors());
app.use(express.json({ limit: "50mb" }));

// ── Public routes (no auth, for health checks / Render) ──────────────────────
app.get("/", (_req, res) => {
  res.json({
    name:    "OMEGA AI Engine",
    version: "1.0",
    status:  "running",
    docs:    {
      health:    "GET /health",
      status:    "GET /status",
      query:     "POST /query",
      learn:     "POST /learn",
      learnBatch: "POST /learn-batch",
      consolidate: "POST /consolidate",
      dumpLi:     "POST /dump-li",
    },
  });
});

app.get("/health", (_req, res) => {
  res.json({ ok: true, ready: mirrorAI.ready });
});

// ── Auth middleware ────────────────────────────────────────────────────────────
const API_KEY = process.env.AI_ENGINE_API_KEY ?? "generate-a-strong-random-key";
app.use((req, res, next) => {
  const key = req.headers["x-api-key"];
  if (!API_KEY || key !== API_KEY) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }
  next();
});

// ── Protected routes ──────────────────────────────────────────────────────────
app.get("/status", (_req, res) => {
  res.json(mirrorAI.status());
});

/** POST /query — run inference, apply filters */
app.post("/query", async (req, res) => {
  const { type, input, context, user_plan, multimodal } = req.body as {
    type:       NetworkType;
    input:      unknown;
    context?:   Record<string, unknown>;
    user_plan?: string;
    multimodal?: boolean;
  };

  if (!type || input === undefined) {
    res.status(400).json({ error: "type and input are required" });
    return;
  }

  try {
    const truth = await mirrorAI.query(
      type,
      input,
      context,
      user_plan ?? "free",
      { multimodal: !!multimodal },
    );
    res.json(truth);
  } catch (err: unknown) {
    res.status(500).json({ error: String(err) });
  }
});

/** POST /learn — feed training data */
app.post("/learn", async (req, res) => {
  const { type, key, data, source, raw, quality } = req.body as {
    type:     NetworkType;
    key:      string;
    data:     number[];
    source:   string;
    raw?:     string;
    quality?: number;
  };

  if (!type || !key || !Array.isArray(data)) {
    res.status(400).json({ error: "type, key, and data[] are required" });
    return;
  }

  await mirrorAI.learn(type, key, data, source ?? "api", raw, quality);
  res.json({ ok: true });
});

/** POST /learn-batch — feed multiple training items (reduces HTTP overhead) */
app.post("/learn-batch", async (req, res) => {
  const { items } = req.body as {
    items: Array<{ type: NetworkType; key: string; data: number[]; source: string; raw?: string; quality?: number }>;
  };

  if (!Array.isArray(items) || items.length === 0) {
    res.status(400).json({ error: "items[] array is required and must not be empty" });
    return;
  }

  for (const item of items) {
    if (!item.type || !item.key || !Array.isArray(item.data)) continue;
    await mirrorAI.learn(
      item.type,
      item.key,
      item.data,
      item.source ?? "api",
      item.raw,
      item.quality ?? 0.7,
    );
  }
  await knowledgeStore.flushAll();
  res.json({ ok: true, count: items.length });
});

/** POST /consolidate — memory consolidation + flush knowledge to Supabase */
app.post("/consolidate", async (_req, res) => {
  const result = await mirrorAI.consolidate();
  res.json({ ok: true, pruned: result });
});

/** POST /dump-li — dump all Li RAM knowledge to Supabase (for data collected before Supabase was configured) */
app.post("/dump-li", async (_req, res) => {
  try {
    const result = await mirrorAI.dumpLiToSupabase();
    res.json({ ok: true, ...result });
  } catch (err) {
    console.error("[OMEGA] dump-li failed:", err);
    res.status(500).json({ ok: false, error: String(err) });
  }
});

/** POST /filters/reload — hot-reload filters from Supabase */
app.post("/filters/reload", async (_req, res) => {
  const count = await mirrorAI.reloadFilters();
  res.json({ ok: true, filters_loaded: count });
});

// ── Graceful shutdown ─────────────────────────────────────────────────────────

process.on("SIGTERM", async () => {
  console.log("[OMEGA] Shutting down, flushing knowledge...");
  await mirrorAI.shutdown();
  process.exit(0);
});

process.on("SIGINT", async () => {
  console.log("[OMEGA] Shutting down, flushing knowledge...");
  await mirrorAI.shutdown();
  process.exit(0);
});

// ── Boot ──────────────────────────────────────────────────────────────────────

(async () => {
  await mirrorAI.init();
  app.listen(PORT);
})();

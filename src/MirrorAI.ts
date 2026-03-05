/**
 * MirrorAI — Unified OMEGA AI System
 *
 * Top-level orchestrator for the gX · Li · Ω engine.
 * Manages all 5 networks, the Omega truth center, training,
 * knowledge persistence (Supabase) and filter engine.
 *
 * On init():
 *   1. Loads knowledge from Supabase → restores Li centers
 *   2. Loads active filters → FilterEngine ready
 *
 * On learn():
 *   1. Li centers learn (RAM)
 *   2. KnowledgeStore.enqueue() → batched async save to Supabase
 *
 * On query():
 *   1. FilterEngine.preQuery() — block/restrict check
 *   2. AI inference (gX → Li → Omega)
 *   3. FilterEngine.postQuery() — boost/quality/inject
 */

import { Omega, OmegaTruth } from "./omega/Omega";
import { NetworkType } from "./centers/Li";
import { BaseNetwork } from "./networks/BaseNetwork";
import { TextNet }  from "./networks/TextNet";
import { ImageNet } from "./networks/ImageNet";
import { VideoNet } from "./networks/VideoNet";
import { AudioNet } from "./networks/AudioNet";
import { GameNet }  from "./networks/GameNet";
import { TRAINING_PARAMS } from "./training/parameters";
import { knowledgeStore } from "./persistence/KnowledgeStore";
import { filterEngine } from "./filters/FilterEngine";
import { CREATOR_FOUNDATION } from "./foundation/creator";
import {
  encodeText,
  encodeClipText,
  encodeClipImage,
  encodeClipImageFromUrl,
  encodeClipFrames,
  encodeClapText,
  encodeClapAudio,
  transcribe,
  captionImage,
} from "./encoders/EncoderClient";

export interface MirrorAIStatus {
  version: string;
  ready: boolean;
  omega: ReturnType<Omega["state"]>;
  networks: {
    text:  ReturnType<TextNet["stats"]>;
    image: ReturnType<ImageNet["stats"]>;
    video: ReturnType<VideoNet["stats"]>;
    audio: ReturnType<AudioNet["stats"]>;
    game:  ReturnType<GameNet["stats"]>;
  };
  uptime:        number;
  totalQueries:  number;
  knowledgeRows: number;
  filtersLoaded: number;
}

export class MirrorAI {
  private readonly VERSION = "1.0.0";
  private startedAt = Date.now();

  readonly omega: Omega;
  readonly text:  TextNet;
  readonly image: ImageNet;
  readonly video: VideoNet;
  readonly audio: AudioNet;
  readonly game:  GameNet;

  private _ready       = false;
  private totalQueries = 0;
  private knowledgeRows = 0;

  constructor() {
    this.omega = new Omega();
    this.text  = new TextNet(this.omega);
    this.image = new ImageNet(this.omega);
    this.video = new VideoNet(this.omega);
    this.audio = new AudioNet(this.omega);
    this.game  = new GameNet(this.omega);
  }

  // ── Init: restore from Supabase ───────────────────────────────────────────

  async init(): Promise<void> {
    console.log(`[OMEGA] Mirror AI v${this.VERSION} initializing...`);
    console.log(`[OMEGA] Parameters: LR=${TRAINING_PARAMS.LEARNING_RATE} | Neurons=${TRAINING_PARAMS.NEURON_DENSITY} | Layers=${TRAINING_PARAMS.LAYER_DEPTH}`);

    // Load filters
    await filterEngine.init();

    // Restore knowledge from Supabase
    const knowledgeMap = await knowledgeStore.loadAll();

    const networkMap: Record<NetworkType, BaseNetwork> = {
      text:  this.text,
      image: this.image,
      video: this.video,
      audio: this.audio,
      game:  this.game,
    };

    for (const [type, rows] of knowledgeMap.entries()) {
      const net = networkMap[type];
      if (!net) continue;

      for (const row of rows) {
        await net.learn(row.key, row.vector, row.source, row.raw, {
          strength:     row.strength,
          accessCount:  row.access_count,
          lastAccessed: row.last_accessed
            ? new Date(row.last_accessed).getTime()
            : Date.now(),
        });
      }
      this.knowledgeRows += rows.length;
      console.log(`[OMEGA] Restored ${rows.length} ${type} knowledge entries.`);
    }

    // Первое рождение — сообщение от создателя (основа бытия OMEGA)
    const foundationVec = await this.text.encode(CREATOR_FOUNDATION.message);
    await this.text.learn(
      CREATOR_FOUNDATION.key,
      foundationVec,
      CREATOR_FOUNDATION.source,
      CREATOR_FOUNDATION.message,
    );
    this.knowledgeRows++;
    console.log(`[OMEGA] Creator foundation loaded — ${CREATOR_FOUNDATION.creator}`);

    this._ready = true;
    console.log(`[OMEGA] AI Engine running on http://localhost:${process.env.PORT ?? 4000}`);
    console.log(`[OMEGA] Ready — ${this.knowledgeRows} entries loaded, ${filterEngine.filterCount} filters active.`);
  }

  // ── Query ─────────────────────────────────────────────────────────────────

  async query(
    type: NetworkType,
    input: unknown,
    context?: Record<string, unknown>,
    userPlan = "free",
    options?: { multimodal?: boolean },
  ): Promise<OmegaTruth & { blocked?: boolean; blockReason?: string; filtersApplied?: string[] }> {
    if (!this._ready) throw new Error("MirrorAI not initialized. Call init() first.");

    // 1. Pre-query filter check
    const inputStr = typeof input === "string" ? input : JSON.stringify(input);
    const preCheck = await filterEngine.preQuery(inputStr, type, userPlan);

    if (!preCheck.allowed) {
      return {
        answer:          preCheck.reason ?? "Запрос заблокирован.",
        answerVector:    [],
        confidence:      0,
        converged:       false,
        iterations:      0,
        participatingLi: [],
        mirrorAgreement: 0,
        networkType:     type,
        timestamp:       Date.now(),
        processingMs:    0,
        knowledgeRecall: [],
        recallUsed:      false,
        blocked:         true,
        blockReason:     preCheck.reason,
      };
    }

    // 2. AI inference
    this.totalQueries++;
    let truth: OmegaTruth;

    if (options?.multimodal && ["text", "image", "video", "audio"].includes(type)) {
      truth = await this.queryMultimodal(type, input);
    } else {
      switch (type) {
        case "text":  truth = await this.text.infer(input, context);  break;
        case "image": truth = await this.image.infer(input, context); break;
        case "video": truth = await this.video.infer(input, context); break;
        case "audio": truth = await this.audio.infer(input, context); break;
        case "game":  truth = await this.game.infer(input, context);  break;
        default:      throw new Error(`Unknown network type: ${type}`);
      }
    }

    // 3. Post-query filters (boost, quality check, inject)
    const { truth: filtered, filtersApplied } = await filterEngine.postQuery(truth, type);

    return { ...filtered, filtersApplied };
  }

  /** Multimodal query — recall from all Li (text, image, video, audio). */
  private async queryMultimodal(type: NetworkType, input: unknown): Promise<OmegaTruth> {
    const queryLabel = typeof input === "string" ? input.slice(0, 200) : JSON.stringify(input).slice(0, 200);
    const vectors: { text?: number[]; clip?: number[]; clap?: number[] } = {};

    try {
      if (type === "text") {
        const text = String(input);
        vectors.text = await encodeText(text);
        vectors.clip = await encodeClipText(text);
        vectors.clap = await encodeClapText(text);
      } else if (type === "image") {
        const obj = input as { data?: string; url?: string };
        const caption = obj.data
          ? await captionImage(obj.data)
          : obj.url
            ? await captionImage(undefined, obj.url)
            : "";
        vectors.text = caption ? await encodeText(caption) : undefined;
        vectors.clip = obj.data
          ? await encodeClipImage(obj.data)
          : obj.url
            ? await encodeClipImageFromUrl(obj.url)
            : undefined;
      } else if (type === "video") {
        const obj = input as { frames?: string[]; url?: string };
        const caption = ""; // TODO: caption-video when implemented
        vectors.text = caption ? await encodeText(caption) : undefined;
        if (obj.frames?.length) {
          vectors.clip = await encodeClipFrames(obj.frames);
        }
        // TODO: fetch frames from url when needed
      } else if (type === "audio") {
        const obj = input as { data?: string; url?: string };
        const text = obj.data
          ? await transcribe(obj.data)
          : obj.url
            ? await transcribe(undefined, obj.url)
            : "";
        vectors.text = text ? await encodeText(text) : undefined;
        vectors.clap = obj.data
          ? await encodeClapAudio(obj.data)
          : undefined;
        // TODO: encodeClapAudio from url
      }

      return this.omega.emitMultimodal({
        query: queryLabel,
        vectors,
      });
    } catch (e) {
      console.warn("[MirrorAI] Multimodal query failed:", e);
      return {
        answer:          `[Multimodal error: ${String(e)}. Check Encoder Service.]`,
        answerVector:     [],
        confidence:       0,
        converged:        false,
        iterations:       0,
        participatingLi:  [],
        mirrorAgreement:  0,
        networkType:      type,
        timestamp:        Date.now(),
        processingMs:     0,
        knowledgeRecall:  [],
        recallUsed:       false,
      };
    }
  }

  // ── Learn ─────────────────────────────────────────────────────────────────

  async learn(
    type: NetworkType,
    key: string,
    data: number[],
    source: string,
    raw?: string,
    quality = 0.7,
  ): Promise<void> {
    const networkMap: Record<NetworkType, BaseNetwork> = {
      text:  this.text,
      image: this.image,
      video: this.video,
      audio: this.audio,
      game:  this.game,
    };

    const net = networkMap[type];
    if (!net) return;

    // 1. Li learns in RAM (fast) — returns updated Knowledge for persistence
    const knowledge = await net.learn(key, data, source, raw);
    this.knowledgeRows++;

    // 2. Async persist to Supabase with real experience values
    knowledgeStore.enqueue({
      key,
      network_type: type,
      vector:       data,
      fingerprint:  knowledge?.fingerprint ?? [],
      raw,
      source,
      strength:     knowledge?.strength ?? 0.5,
      access_count: knowledge?.accessCount ?? 1,
      last_accessed: knowledge?.lastAccessed
        ? new Date(knowledge.lastAccessed).toISOString()
        : new Date().toISOString(),
      quality,
    });
  }

  /**
   * Dump all Li RAM knowledge to Supabase.
   * Use when data was collected before Supabase was configured.
   */
  async dumpLiToSupabase(): Promise<{ total: number; byNetwork: Record<NetworkType, number> }> {
    const byNetwork: Record<NetworkType, number> = {
      text: 0, image: 0, video: 0, audio: 0, game: 0,
    };
    const networkMap: Record<NetworkType, BaseNetwork> = {
      text:  this.text,
      image: this.image,
      video: this.video,
      audio: this.audio,
      game:  this.game,
    };
    for (const [type, net] of Object.entries(networkMap) as [NetworkType, BaseNetwork][]) {
      for (const k of net.getAllKnowledgeForPersistence()) {
        knowledgeStore.enqueue({
          key:           k.key,
          network_type:  type,
          vector:        k.value,
          fingerprint:  k.fingerprint,
          raw:           k.raw ?? undefined,
          source:        k.source,
          strength:      k.strength,
          access_count: k.accessCount,
          quality:       0.7,
          last_accessed: typeof k.lastAccessed === "number"
            ? new Date(k.lastAccessed).toISOString()
            : new Date().toISOString(),
        });
        byNetwork[type]++;
      }
    }
    await knowledgeStore.flush();
    const total = Object.values(byNetwork).reduce((a, b) => a + b, 0);
    console.log(`[OMEGA] Dumped ${total} Li entries to Supabase`);
    return { total, byNetwork };
  }

  // ── Consolidate ───────────────────────────────────────────────────────────

  async consolidate(): Promise<Record<NetworkType, number>> {
    const result = {
      text:  this.text.consolidate(),
      image: this.image.consolidate(),
      video: this.video.consolidate(),
      audio: this.audio.consolidate(),
      game:  this.game.consolidate(),
    };

    // Force flush pending writes
    await knowledgeStore.flush();
    return result;
  }

  // ── Reload filters (hot reload) ───────────────────────────────────────────

  async reloadFilters(): Promise<number> {
    await filterEngine.reload();
    return filterEngine.filterCount;
  }

  // ── Shutdown ──────────────────────────────────────────────────────────────

  async shutdown(): Promise<void> {
    await knowledgeStore.shutdown();
  }

  get ready()        { return this._ready; }

  status(): MirrorAIStatus {
    return {
      version:      this.VERSION,
      ready:        this._ready,
      omega:        this.omega.state(),
      networks: {
        text:  this.text.stats(),
        image: this.image.stats(),
        video: this.video.stats(),
        audio: this.audio.stats(),
        game:  this.game.stats(),
      },
      uptime:        Date.now() - this.startedAt,
      totalQueries:  this.totalQueries,
      knowledgeRows: this.knowledgeRows,
      filtersLoaded: filterEngine.filterCount,
    };
  }
}

export const mirrorAI = new MirrorAI();

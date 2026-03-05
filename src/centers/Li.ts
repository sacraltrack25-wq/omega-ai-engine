/**
 * Li — Processing Center (очаг обработки данных)
 *
 * Li — это мозговой очаг. gX — нейроны внутри него.
 *
 * Mirror Principle:
 *   let Li1  →  primary processing center   (MUTABLE, grows with data)
 *   let Li2  →  mirror processing center    (MUTABLE, mirrors Li1)
 *
 * Как работает резонанс:
 *   1. Входной вектор проходит через слои gX нейронов
 *   2. Каждый нейрон: если input бит == gX1 → score +1 (резонанс)
 *                     если input бит == gX2 → score -1 (анти-резонанс)
 *   3. Суммарный резонанс слоя = Σ(score)/N  →  [-1..+1]
 *   4. Выход слоя (activation pattern) = бинарный отпечаток состояния
 *   5. Этот отпечаток = уникальная "подпись" того что Li думает о данных
 *
 * Knowledge recall через gX-отпечатки:
 *   При обучении → gX нейроны создают fingerprint вектора → хранится в Knowledge
 *   При запросе  → gX нейроны создают fingerprint запроса → сравниваем отпечатки
 *   Совпадение отпечатков = gX РЕЗОНАНС → Li нашёл связанное знание
 *
 * Omega взвешивает резонансы всех Li центров и выдаёт истину.
 */

import { NeuronLayer, Bit, LayerResult } from "../neurons/gX";
import { TRAINING_PARAMS } from "../training/parameters";

// ─── Types ────────────────────────────────────────────────────────────────────

export type NetworkType = "text" | "image" | "video" | "audio" | "game";

export interface LiConfig {
  id: string;
  type: NetworkType;
  layerDepth: number;
  neuronsPerLayer: number;
}

export interface ProcessingResult {
  liId: string;
  confidence: number;
  output: number[];
  resonance: number;
  processed: boolean;
  timestamp: number;
}

export interface Knowledge {
  key: string;
  value: number[];          // original feature vector from harvester
  fingerprint: number[];    // gX activation pattern (binary, 0/1) — the resonance signature
  source: string;
  strength: number;
  lastAccessed: number;
  accessCount: number;
  raw?: string;             // original text for human-readable recall
}

export interface LearnOptions {
  strength?: number;
  accessCount?: number;
  lastAccessed?: number;
}

export interface KnowledgeMatch {
  key: string;
  /**
   * Combined score: gX resonance similarity × knowledge strength
   * 1.0 = perfect gX resonance match
   * 0.0 = no match
   */
  score: number;
  resonanceScore: number;   // pure gX fingerprint similarity [0..1]
  source: string;
  raw?: string;
  liId: string;
}

export interface LiStatus {
  id: string;
  type: NetworkType;
  layers: number;
  neurons: number;
  knowledgeSize: number;
  avgResonance: number;
  processingCount: number;
  hasMirror: boolean;
  mirrorId: string | null;
}

// ─── Processing Center ────────────────────────────────────────────────────────

export class ProcessingCenter {
  readonly id: string;
  readonly type: NetworkType;

  private _mirrorCenter: ProcessingCenter | null = null;

  // Mutable state — grows with data (let Li)
  private layers: NeuronLayer[] = [];
  private knowledge: Map<string, Knowledge> = new Map();
  private processingCount = 0;
  private totalResonance  = 0;

  private readonly config: LiConfig;

  constructor(config: LiConfig) {
    this.id     = config.id;
    this.type   = config.type;
    this.config = config;
    this.initLayers();
  }

  private initLayers() {
    for (let d = 0; d < this.config.layerDepth; d++) {
      this.layers.push(
        new NeuronLayer(`${this.id}_L${d}`, d, this.config.neuronsPerLayer),
      );
    }
  }

  // ── Mirror connection ──────────────────────────────────────────────────────

  connectMirror(mirror: ProcessingCenter) {
    this._mirrorCenter = mirror;
    if (mirror._mirrorCenter !== this) {
      mirror._mirrorCenter = this;
    }
  }

  get mirrorCenter() { return this._mirrorCenter; }

  // ── Core processing ────────────────────────────────────────────────────────

  /**
   * Process an input vector through all gX layers.
   * gX нейроны резонируют с входными битами.
   * Выход каждого слоя подаётся на вход следующему.
   */
  async process(
    inputVector: number[],
    _context?: Record<string, unknown>,
  ): Promise<ProcessingResult> {
    if (inputVector.length === 0) {
      return { liId: this.id, confidence: 0, output: [], resonance: 0, processed: false, timestamp: Date.now() };
    }

    const { fingerprint, resonance, output } = this.runThroughGX(inputVector);
    const confidence = this.calcConfidence(resonance, output);

    this.processingCount++;
    this.totalResonance += resonance;
    this.considerGrowth(resonance);

    return {
      liId:      this.id,
      confidence,
      output,
      resonance,
      processed: true,
      timestamp: Date.now(),
    };
  }

  // ── Knowledge / Learning ───────────────────────────────────────────────────

  /** Readonly access to knowledge entry by key (for persistence) */
  getKnowledge(key: string): Knowledge | undefined {
    return this.knowledge.get(key);
  }

  /** Get all knowledge for persistence (dump Li RAM → Supabase) */
  getAllKnowledgeForPersistence(): Knowledge[] {
    return Array.from(this.knowledge.values());
  }

  /**
   * Absorb new data.
   * gX нейроны создают fingerprint входного вектора — "отпечаток резонанса".
   * Этот отпечаток хранится вместе с данными и используется при recall.
   * @param restore — when provided, set strength/accessCount/lastAccessed (restore from DB)
   * @returns the Knowledge entry after update
   */
  async learn(
    key: string,
    data: number[],
    source: string,
    raw?: string,
    restore?: LearnOptions,
  ): Promise<Knowledge> {
    const { fingerprint } = this.runThroughGX(data);

    const existing = this.knowledge.get(key);
    if (existing) {
      if (restore) {
        existing.strength     = restore.strength ?? existing.strength;
        existing.accessCount  = restore.accessCount ?? existing.accessCount;
        existing.lastAccessed = restore.lastAccessed ?? existing.lastAccessed;
        existing.value        = data;
        existing.fingerprint  = fingerprint;
        if (raw) existing.raw = raw;
      } else {
        existing.strength     = Math.min(1, existing.strength + 0.1);
        existing.accessCount++;
        existing.lastAccessed = Date.now();
        existing.value        = existing.value.map((v, i) => v * 0.9 + (data[i] ?? 0) * 0.1);
        existing.fingerprint  = fingerprint;
        if (raw && !existing.raw) existing.raw = raw;
      }
    } else {
      const entry: Knowledge = {
        key,
        value: data,
        fingerprint,
        source,
        strength:     restore?.strength ?? 0.5,
        lastAccessed: restore?.lastAccessed ?? Date.now(),
        accessCount:  restore?.accessCount ?? 1,
        raw,
      };
      this.knowledge.set(key, entry);

      if (this.knowledge.size % TRAINING_PARAMS.LI_LAYER_SPAWN_THRESHOLD === 0) {
        this.grow();
      }
    }
    return this.knowledge.get(key)!;
  }

  /**
   * Recall: найти топ-K знаний которые резонируют с queryVector.
   *
   * Механизм:
   *   1. Прогоняем queryVector через gX нейроны → получаем query fingerprint
   *   2. Сравниваем query fingerprint с сохранёнными fingerprints через gX resonance score
   *   3. gX resonance = доля совпавших битов (Hamming similarity)
   *   4. Итоговый score = resonance × (0.5 + strength × 0.5)
   *
   * Это настоящий gX-резонанс — не математическая формула, а нейронный отклик.
   */
  recall(queryVector: number[], topK = 3): KnowledgeMatch[] {
    if (this.knowledge.size === 0) return [];

    // Compute gX fingerprint for the query
    const { fingerprint: queryFingerprint } = this.runThroughGX(queryVector);

    const results: KnowledgeMatch[] = [];

    for (const entry of this.knowledge.values()) {
      const resonanceScore = this.gxResonanceSimilarity(queryFingerprint, entry.fingerprint);
      const score = resonanceScore * (0.5 + entry.strength * 0.5);

      results.push({
        key:           entry.key,
        score,
        resonanceScore,
        source:        entry.source,
        raw:           entry.raw,
        liId:          this.id,
      });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  /** Consolidate memory — prune weak knowledge, reinforce strong */
  consolidate(): number {
    let pruned = 0;
    for (const [key, entry] of this.knowledge) {
      if (entry.strength < 0.1 && entry.accessCount < 2) {
        this.knowledge.delete(key);
        pruned++;
      }
    }
    return pruned;
  }

  // ── Growth ────────────────────────────────────────────────────────────────

  private grow() {
    const rate = TRAINING_PARAMS.LI_GROWTH_RATE;

    for (const layer of this.layers) {
      layer.grow(Math.max(1, Math.floor(layer.size * rate)));
    }

    const expectedLayers = Math.ceil(
      this.knowledge.size / TRAINING_PARAMS.LI_LAYER_SPAWN_THRESHOLD,
    );
    while (this.layers.length < expectedLayers) {
      this.layers.push(
        new NeuronLayer(
          `${this.id}_L${this.layers.length}`,
          this.layers.length,
          this.config.neuronsPerLayer,
        ),
      );
    }
  }

  private considerGrowth(resonance: number) {
    if (resonance < 0.3 && this.processingCount % 50 === 0) {
      this.grow();
    }
  }

  // ── Core gX resonance engine ──────────────────────────────────────────────

  /**
   * Run a vector through all gX neuron layers.
   *
   * Улучшение над предыдущей версией:
   *   - Использует weightedResonance (взвешенное по важности нейронов)
   *   - Fingerprint = непрерывные оценки нейронов (не просто 0/1)
   *     → более информативные совпадения при recall
   *   - Resonance = взвешенное среднее с экспоненциальным затуханием
   *     по слоям: последние слои важнее (ближе к выходу)
   */
  private runThroughGX(inputVector: number[]): {
    fingerprint: number[];
    resonance: number;
    output: number[];
  } {
    let currentInput = this.vectorToBinary(inputVector);
    let totalWeightedRes = 0;
    let totalLayerWeight = 0;
    let lastResult: LayerResult | null = null;

    for (let i = 0; i < this.layers.length; i++) {
      const result = this.layers[i].process(currentInput);
      // Экспоненциальное взвешивание слоёв: последние слои важнее
      const layerImportance = Math.exp(i / Math.max(1, this.layers.length - 1));
      totalWeightedRes += result.weightedResonance * layerImportance;
      totalLayerWeight += layerImportance;
      lastResult = result;
      // Activation pattern передаётся в следующий слой
      currentInput = result.activations.map(a => (a ? 1 : 0) as Bit);
    }

    const resonance = totalLayerWeight > 0 ? totalWeightedRes / totalLayerWeight : 0;

    // Fingerprint: используем непрерывные scores последнего слоя
    // ([-weight..+weight] для каждого нейрона)
    // Более богатое представление чем бинарное 0/1
    const fingerprint = lastResult ? lastResult.scores : [];
    const output      = lastResult ? lastResult.activations.map(a => (a ? 1 : 0)) : [];

    return { fingerprint, resonance, output };
  }

  /**
   * gX Resonance Similarity — cosine similarity на непрерывных fingerprints.
   *
   * Fingerprint теперь = scores[] нейронов (float, не 0/1).
   * Cosine similarity: насколько оба вектора "смотрят в одну сторону".
   *
   * 1.0 = идеальное совпадение паттернов активации
   * 0.0 = ортогональные паттерны (нет связи)
   * <0  = противоположные паттерны (anti-resonance)
   *
   * Нормируем в [0..1] через (cosine + 1) / 2.
   *
   * Это значительно точнее бинарного подсчёта совпадений —
   * нейроны с большим весом (важные) вносят больший вклад в сходство.
   */
  private gxResonanceSimilarity(a: number[], b: number[]): number {
    const len = Math.min(a.length, b.length);
    if (len === 0) return 0;

    let dot  = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < len; i++) {
      dot   += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    if (denom < 1e-8) return 0;

    // Нормируем cosine [-1..1] → [0..1]
    return (dot / denom + 1) / 2;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private vectorToBinary(vector: number[]): Bit[] {
    const max = Math.max(...vector, 1e-8);
    return vector.map(v => (v / max > 0.5 ? 1 : 0) as Bit);
  }

  private calcConfidence(resonance: number, output: number[]): number {
    const activationRate = output.length > 0
      ? output.filter(v => v > 0).length / output.length
      : 0;
    return Math.max(0, Math.min(1, (resonance + 1) / 2 * 0.6 + activationRate * 0.4));
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  get layerCount()     { return this.layers.length; }
  get knowledgeSize()  { return this.knowledge.size; }
  get neuronCount()    { return this.layers.reduce((s, l) => s + l.size, 0); }
  get avgResonance()   { return this.processingCount > 0 ? this.totalResonance / this.processingCount : 0; }

  status(): LiStatus {
    return {
      id:              this.id,
      type:            this.type,
      layers:          this.layerCount,
      neurons:         this.neuronCount,
      knowledgeSize:   this.knowledgeSize,
      avgResonance:    this.avgResonance,
      processingCount: this.processingCount,
      hasMirror:       this._mirrorCenter !== null,
      mirrorId:        this._mirrorCenter?.id ?? null,
    };
  }
}

// ─── Li Cluster (mirror pair + additional cores) ──────────────────────────────

export interface ClusterResult {
  clusterId: string;
  primaryResult:  ProcessingResult;
  mirrorResult:   ProcessingResult;
  consensusOutput: number[];
  consensusConfidence: number;
  mirrorAgreement: number;
  timestamp: number;
}

export class LiCluster {
  readonly id: string;
  readonly type: NetworkType;

  Li1: ProcessingCenter;
  Li2: ProcessingCenter;

  private extraCores: Array<[ProcessingCenter, ProcessingCenter]> = [];

  constructor(id: string, type: NetworkType, coreCount = 4) {
    this.id   = id;
    this.type = type;

    const baseConfig: LiConfig = {
      id:              `${id}_Li1`,
      type,
      layerDepth:      TRAINING_PARAMS.LAYER_DEPTH,
      neuronsPerLayer: TRAINING_PARAMS.NEURON_DENSITY,
    };

    this.Li1 = new ProcessingCenter(baseConfig);
    this.Li2 = new ProcessingCenter({ ...baseConfig, id: `${id}_Li2` });
    this.Li1.connectMirror(this.Li2);

    const extraPairs = Math.max(0, Math.floor(coreCount / 2) - 1);
    for (let i = 0; i < extraPairs; i++) {
      const n  = (i + 2) * 2;
      const c1 = new ProcessingCenter({ ...baseConfig, id: `${id}_Li${n - 1}` });
      const c2 = new ProcessingCenter({ ...baseConfig, id: `${id}_Li${n}` });
      c1.connectMirror(c2);
      this.extraCores.push([c1, c2]);
    }
  }

  /**
   * Process query through mirror pair — Li1 and Li2 resonate independently,
   * then we take consensus. Mirror disagreement reveals uncertainty.
   */
  async process(inputVector: number[], context?: Record<string, unknown>): Promise<ClusterResult> {
    const [primary, mirror] = await Promise.all([
      this.Li1.process(inputVector, context),
      this.Li2.process(inputVector, context),
    ]);

    const agreement = this.calcAgreement(primary.output, mirror.output);
    const consensus = this.buildConsensus(primary, mirror, agreement);

    return {
      clusterId:           this.id,
      primaryResult:       primary,
      mirrorResult:        mirror,
      consensusOutput:     consensus.output,
      consensusConfidence: consensus.confidence,
      mirrorAgreement:     agreement,
      timestamp:           Date.now(),
    };
  }

  /** Mirror learning — Li1 and Li2 learn the same data from opposite perspectives */
  async learn(
    key: string,
    data: number[],
    source: string,
    raw?: string,
    restore?: LearnOptions,
  ): Promise<Knowledge> {
    await Promise.all([
      this.Li1.learn(key, data, source, raw, restore),
      this.Li2.learn(key, data, source, raw, restore),
      ...this.extraCores.flat().map(c => c.learn(key, data, source, raw, restore)),
    ]);
    return this.Li1.getKnowledge(key)!;
  }

  /**
   * Recall via gX resonance across all centers in cluster.
   * Li1 and Li2 may find different memories — we take best score per key.
   * This captures the "two perspectives" aspect of mirror learning.
   */
  recall(queryVector: number[], topK = 3): KnowledgeMatch[] {
    const allMatches = this.allCenters.flatMap(c => c.recall(queryVector, topK));

    const best = new Map<string, KnowledgeMatch>();
    for (const m of allMatches) {
      const prev = best.get(m.key);
      if (!prev || m.score > prev.score) best.set(m.key, m);
    }

    return Array.from(best.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  private calcAgreement(a: number[], b: number[]): number {
    const len = Math.min(a.length, b.length);
    if (len === 0) return 0;
    let matches = 0;
    for (let i = 0; i < len; i++) {
      if ((a[i] > 0.5) === (b[i] > 0.5)) matches++;
    }
    return matches / len;
  }

  private buildConsensus(
    p: ProcessingResult,
    m: ProcessingResult,
    agreement: number,
  ): { output: number[]; confidence: number } {
    const len = Math.max(p.output.length, m.output.length);
    const output: number[] = [];
    for (let i = 0; i < len; i++) {
      const pw    = p.confidence;
      const mw    = m.confidence;
      const total = pw + mw || 1;
      output.push(((p.output[i] ?? 0) * pw + (m.output[i] ?? 0) * mw) / total);
    }
    const confidence = (p.confidence + m.confidence) / 2 * (0.5 + agreement * 0.5);
    return { output, confidence };
  }

  get allCenters(): ProcessingCenter[] {
    return [this.Li1, this.Li2, ...this.extraCores.flat()];
  }

  statuses(): LiStatus[] {
    return this.allCenters.map(c => c.status());
  }
}

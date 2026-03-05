/**
 * Ω Omega — Truth Center (Центр Истины)
 *
 * let Omega — самозеркалящийся, самовалидирующийся (MUTABLE)
 *
 * Полный цикл обработки запроса:
 *
 *  ┌────────────┐    gX резонанс    ┌────────────┐
 *  │  Query     │ ─────────────────▶│  Li1 / Li2 │  ← mirror pair
 *  │  Vector    │                   │  (cluster) │
 *  └────────────┘                   └─────┬──────┘
 *                                         │ ClusterResult
 *                                         │ (consensus, mirrorAgreement)
 *                                         ▼
 *                                  ┌────────────┐
 *                                  │   Omega    │  Ω(x) → Ω(Ω(x)) → ...
 *                                  │  synthesize│  until |Ω(Ω(x)) - Ω(x)| < ε
 *                                  │  + mirror  │  (convergence = truth)
 *                                  └─────┬──────┘
 *                                        │ candidate vector
 *                                        │
 *                                        ▼
 *                          ┌─────────────────────────┐
 *                          │  Knowledge Recall (gX)  │
 *                          │  gX fingerprint match   │
 *                          │  query + candidate both │
 *                          └─────────────────────────┘
 *                                        │ KnowledgeMatch[]
 *                                        ▼
 *                                  ┌──────────┐
 *                                  │  decode  │
 *                                  │  answer  │  → const OmegaTruth
 *                                  └──────────┘
 *
 * Omega взвешивает результаты Li центров:
 *   weight(Li) = consensusConfidence × (0.5 + mirrorAgreement × 0.5)
 *   Высокое mirrorAgreement (Li1 ↔ Li2 согласны) = бо́льший вес в ответе
 *
 * Knowledge Recall (gX-based):
 *   Не cosine similarity — а сравнение gX activation fingerprints.
 *   resonanceScore = % gX нейронов с одинаковым ответом на два вектора.
 *   1.0 = полный резонанс = одинаковое воспоминание
 */

import { ClusterResult, LiCluster, NetworkType, KnowledgeMatch } from "../centers/Li";
import { TRAINING_PARAMS } from "../training/parameters";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface OmegaInput {
  query: string;
  queryVector: number[];
  networkType: NetworkType;
  context?: Record<string, unknown>;
}

/** Input for multimodal mode — vectors for each network type. */
export interface OmegaMultimodalInput {
  query: string;
  vectors: {
    text?: number[];
    clip?: number[];  // image + video Li
    clap?: number[];  // audio Li
  };
  context?: Record<string, unknown>;
}

export interface OmegaTruth {
  // CONST — the final, validated answer
  readonly answer: string;
  readonly answerVector: number[];
  readonly confidence: number;
  readonly converged: boolean;
  readonly iterations: number;
  readonly participatingLi: string[];
  readonly mirrorAgreement: number;
  readonly networkType: NetworkType;
  readonly timestamp: number;
  readonly processingMs: number;
  // Knowledge recall — top matches from Li Knowledge Maps
  readonly knowledgeRecall: KnowledgeMatch[];
  readonly recallUsed: boolean;   // true when answer came from Knowledge Map
}

export interface OmegaState {
  id: string;
  activeNetworks: NetworkType[];
  totalQueries: number;
  avgConfidence: number;
  avgIterations: number;
  convergeRate: number;
}

// ─── Omega ────────────────────────────────────────────────────────────────────

export class Omega {
  // let Omega — auto-mirroring (validates itself)
  private clusters: Map<NetworkType, LiCluster[]> = new Map();
  private queryCount       = 0;
  private totalConfidence  = 0;
  private totalIterations  = 0;
  private convergedCount   = 0;

  // Omega's own mirror state (used in self-validation)
  private mirrorBuffer: number[] = [];

  constructor() {
    const types: NetworkType[] = ["text", "image", "video", "audio", "game"];
    for (const t of types) {
      this.clusters.set(t, []);
    }
  }

  // ── Cluster registry ──────────────────────────────────────────────────────

  registerCluster(cluster: LiCluster) {
    const list = this.clusters.get(cluster.type) ?? [];
    list.push(cluster);
    this.clusters.set(cluster.type, list);
  }

  // ── Core: emit truth ─────────────────────────────────────────────────────

  /**
   * Process a query and return the verified truth (Ω output).
   *
   * Algorithm:
   * 1. Route query to the correct Li clusters by networkType
   * 2. Collect ClusterResults (Li1 ↔ Li2 consensus, gX resonance)
   * 3. Synthesize candidate answer vector
   * 4. Self-validate: re-run synthesis on the candidate (mirror loop)
   * 5. Repeat until convergence or max iterations
   * 6. Recall: search Li Knowledge Maps for raw text matching answer vector
   * 7. Return const OmegaTruth
   */
  async emit(input: OmegaInput): Promise<OmegaTruth> {
    const start = Date.now();

    const activeClusters = this.clusters.get(input.networkType) ?? [];
    if (activeClusters.length < TRAINING_PARAMS.LI_PARTICIPATION_MIN) {
      return this.lowConfidenceResponse(input, start);
    }

    // ── Phase 1: Gather Li results (gX neurons activate) ──────────────────
    const clusterResults = await Promise.all(
      activeClusters.map(c => c.process(input.queryVector, input.context)),
    );

    const participatingLi = clusterResults.flatMap(r => [r.clusterId + "_Li1", r.clusterId + "_Li2"]);

    // ── Phase 2: Build candidate truth vector ──────────────────────────────
    let candidate = this.synthesize(clusterResults);
    let iterations = 0;
    let converged  = false;

    // ── Phase 3: Self-validation loop (Omega mirrors itself) ───────────────
    for (let i = 0; i < TRAINING_PARAMS.OMEGA_MAX_ITERATIONS; i++) {
      iterations++;
      const reEval = await this.selfMirror(candidate, input.networkType, activeClusters, input.context);
      const delta  = this.vectorDelta(candidate, reEval);

      if (delta < (1 - TRAINING_PARAMS.OMEGA_CONVERGENCE_THRESHOLD)) {
        converged = true;
        candidate = reEval;
        break;
      }
      candidate = reEval;
    }

    // ── Phase 4: Confidence score ──────────────────────────────────────────
    const avgMirrorAgreement = clusterResults.reduce((s, r) => s + r.mirrorAgreement, 0) / clusterResults.length;
    const clusterConfidence  = clusterResults.reduce((s, r) => s + r.consensusConfidence, 0) / clusterResults.length;
    const convergenceFactor  = converged ? 1 : 0.6;
    const confidence         = Math.min(1, clusterConfidence * convergenceFactor * (0.7 + avgMirrorAgreement * 0.3));

    // ── Phase 5: Knowledge recall from Li — find matching raw text ─────────
    // Search both queryVector (what was asked) and candidate (what Omega thinks)
    const recallFromQuery    = activeClusters.flatMap(c => c.recall(input.queryVector, 3));
    const recallFromCandidate = activeClusters.flatMap(c => c.recall(candidate, 3));

    // Merge and deduplicate by key, keep best score
    const recallMap = new Map<string, KnowledgeMatch>();
    for (const m of [...recallFromQuery, ...recallFromCandidate]) {
      const prev = recallMap.get(m.key);
      if (!prev || m.score > prev.score) recallMap.set(m.key, m);
    }
    const knowledgeRecall = Array.from(recallMap.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    // ── Phase 6: Decode to answer string ──────────────────────────────────
    const { answer, recallUsed } = this.decode(
      candidate,
      input.networkType,
      input.query,
      knowledgeRecall,
    );

    // ── Update stats ───────────────────────────────────────────────────────
    this.queryCount++;
    this.totalConfidence  += confidence;
    this.totalIterations  += iterations;
    this.convergedCount   += converged ? 1 : 0;

    // Store in mirror buffer for next Omega self-check
    this.mirrorBuffer = candidate;

    const truth: OmegaTruth = {
      answer,
      answerVector:     candidate,
      confidence,
      converged,
      iterations,
      participatingLi,
      mirrorAgreement:  avgMirrorAgreement,
      networkType:      input.networkType,
      timestamp:        Date.now(),
      processingMs:     Date.now() - start,
      knowledgeRecall,
      recallUsed,
    };

    return truth;
  }

  /**
   * emitMultimodal — recall from all Li (text, image, video, audio) and merge.
   * Used when multimodal: true. Vectors must be pre-computed by caller.
   */
  async emitMultimodal(input: OmegaMultimodalInput): Promise<OmegaTruth> {
    const start = Date.now();
    const weights = this.parseMultimodalWeights();
    const recallMap = new Map<string, KnowledgeMatch>();

    const textClusters = this.clusters.get("text") ?? [];
    const imageClusters = this.clusters.get("image") ?? [];
    const videoClusters = this.clusters.get("video") ?? [];
    const audioClusters = this.clusters.get("audio") ?? [];

    const topK = 3;

    if (input.vectors.text && textClusters.length > 0) {
      const matches = textClusters.flatMap(c => c.recall(input.vectors.text!, topK));
      for (const m of matches) {
        const weightedScore = m.score * weights.text;
        const prev = recallMap.get(m.key);
        if (!prev || weightedScore > prev.score) {
          recallMap.set(m.key, { ...m, score: weightedScore });
        }
      }
    }

    if (input.vectors.clip && (imageClusters.length > 0 || videoClusters.length > 0)) {
      const imgMatches = imageClusters.flatMap(c => c.recall(input.vectors.clip!, topK));
      const vidMatches = videoClusters.flatMap(c => c.recall(input.vectors.clip!, topK));
      for (const m of [...imgMatches, ...vidMatches]) {
        const w = (m.liId.includes("image") ? weights.image : weights.video) * m.score;
        const prev = recallMap.get(m.key);
        if (!prev || w > prev.score) {
          recallMap.set(m.key, { ...m, score: w });
        }
      }
    }

    if (input.vectors.clap && audioClusters.length > 0) {
      const matches = audioClusters.flatMap(c => c.recall(input.vectors.clap!, topK));
      for (const m of matches) {
        const w = weights.audio * m.score;
        const prev = recallMap.get(m.key);
        if (!prev || w > prev.score) {
          recallMap.set(m.key, { ...m, score: w });
        }
      }
    }

    const knowledgeRecall = Array.from(recallMap.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    const { answer, recallUsed } = this.decode(
      [],
      "text",
      input.query,
      knowledgeRecall,
    );

    const participatingLi = [
      ...textClusters.map(c => c.id + "_Li1"),
      ...imageClusters.map(c => c.id + "_Li1"),
      ...videoClusters.map(c => c.id + "_Li1"),
      ...audioClusters.map(c => c.id + "_Li1"),
    ];

    return {
      answer,
      answerVector:     [],
      confidence:       knowledgeRecall[0] ? Math.min(1, knowledgeRecall[0].score * 1.2) : 0,
      converged:        true,
      iterations:       1,
      participatingLi,
      mirrorAgreement:  0.8,
      networkType:      "text",
      timestamp:        Date.now(),
      processingMs:     Date.now() - start,
      knowledgeRecall,
      recallUsed,
    };
  }

  private parseMultimodalWeights(): { text: number; image: number; video: number; audio: number } {
    try {
      const raw = process.env.OMEGA_MULTIMODAL_WEIGHTS;
      if (raw) {
        const w = JSON.parse(raw) as Record<string, number>;
        return {
          text:  w.text ?? 0.4,
          image: w.image ?? 0.2,
          video: w.video ?? 0.2,
          audio: w.audio ?? 0.2,
        };
      }
    } catch {
      /* ignore */
    }
    return { text: 0.4, image: 0.2, video: 0.2, audio: 0.2 };
  }

  // ── Self-mirror: re-run synthesis on candidate ────────────────────────────

  private async selfMirror(
    candidate: number[],
    type: NetworkType,
    clusters: LiCluster[],
    context?: Record<string, unknown>,
  ): Promise<number[]> {
    const reResults = await Promise.all(
      clusters.map(c => c.process(candidate, context)),
    );
    return this.synthesize(reResults);
  }

  /**
   * Synthesis: Omega взвешивает результаты Li через softmax.
   *
   * Улучшение над линейным взвешиванием:
   *   Вместо w = conf × agreement
   *   Используем softmax(logits) где logit = conf × (0.5 + agreement × 0.5)
   *
   * Softmax гарантирует:
   *   - Веса в сумме = 1 (нормированное распределение вероятностей)
   *   - Лучший кластер получает экспоненциально больше веса
   *   - Слабые кластеры почти не влияют на финальный ответ
   *   → более "чёткие" и уверенные ответы
   *
   * Temperature T управляет остротой выбора:
   *   T → 0: winner takes all (самый уверенный Li доминирует)
   *   T → ∞: равномерное взвешивание (демократия кластеров)
   *   T = 1.0: стандартный softmax (баланс)
   */
  private synthesize(results: ClusterResult[], temperature = 1.0): number[] {
    if (results.length === 0) return [];

    // Вычисляем logits для каждого кластера
    const logits = results.map(r =>
      r.consensusConfidence * (0.5 + r.mirrorAgreement * 0.5),
    );

    // Softmax: exp(logit / T) / Σ exp(logit / T)
    const maxLogit = Math.max(...logits);  // numerical stability trick
    const expWeights = logits.map(l => Math.exp((l - maxLogit) / temperature));
    const sumExp = expWeights.reduce((s, e) => s + e, 0);
    const weights = expWeights.map(e => e / sumExp);

    const maxLen = Math.max(...results.map(r => r.consensusOutput.length));
    const output = new Array<number>(maxLen).fill(0);

    for (let ci = 0; ci < results.length; ci++) {
      const w = weights[ci];
      for (let i = 0; i < results[ci].consensusOutput.length; i++) {
        output[i] += (results[ci].consensusOutput[i] ?? 0) * w;
      }
    }

    return output;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private vectorDelta(a: number[], b: number[]): number {
    const len = Math.min(a.length, b.length);
    if (len === 0) return 1;
    let sum = 0;
    for (let i = 0; i < len; i++) sum += Math.abs(a[i] - b[i]);
    return sum / len;
  }

  /**
   * Decode: превращает вектор-истину в читаемый ответ.
   *
   * Omega выбирает ответ взвешивая gX resonanceScore:
   *   1. bestWithRaw: recall match с raw текстом и resonanceScore > threshold
   *      → возвращаем сохранённый текст (реальный ответ)
   *   2. Match найден но нет raw → источник + score
   *   3. Li пустые → инструкция по обучению
   *   4. Нет близких знаний → прозрачный ответ
   */
  private decode(
    vector: number[],
    _type: NetworkType,
    query: string,
    recall: KnowledgeMatch[],
  ): { answer: string; recallUsed: boolean } {
    // gX resonance threshold — при resonanceScore > 0.55 считаем совпадение достаточным
    // (>50% нейронов дали одинаковый ответ на оба вектора)
    const RESONANCE_THRESHOLD = 0.55;
    const SCORE_THRESHOLD = 0.3;

    // Best match with raw text — Omega выбирает по combined score (resonance × strength)
    const bestWithRaw = recall.find(
      m => m.raw && (m.resonanceScore >= RESONANCE_THRESHOLD || m.score >= SCORE_THRESHOLD),
    );
    if (bestWithRaw?.raw) {
      return { answer: bestWithRaw.raw, recallUsed: true };
    }

    // Match found but no raw text (старые данные до поля raw)
    const bestMatch = recall.find(
      m => m.resonanceScore >= RESONANCE_THRESHOLD || m.score >= SCORE_THRESHOLD,
    );
    if (bestMatch) {
      return {
        answer: [
          `[gX Резонанс: ${(bestMatch.resonanceScore * 100).toFixed(0)}%`,
          `score=${bestMatch.score.toFixed(2)}`,
          `источник: ${bestMatch.source}]`,
          `Текст не сохранён — перезапусти харвесторы чтобы обновить базу знаний.`,
        ].join(" | "),
        recallUsed: true,
      };
    }

    // Li пустые
    const magnitude = vector.reduce((s, v) => s + v, 0) / Math.max(vector.length, 1);
    if (magnitude < 0.01) {
      return {
        answer: `[Li-центры пусты. Запусти: python train_textnet.py --level 1]`,
        recallUsed: false,
      };
    }

    // Данные есть, но gX резонанс слабый для этого запроса
    const topScore = recall[0]?.resonanceScore ?? 0;
    return {
      answer: [
        `[gX резонанс слабый (${(topScore * 100).toFixed(0)}%) для запроса "${query}".`,
        `Нужно больше обучающих данных по этой теме. Запусти харвесторы.]`,
      ].join(" "),
      recallUsed: false,
    };
  }

  private lowConfidenceResponse(input: OmegaInput, start: number): OmegaTruth {
    return {
      answer:           "Недостаточно активных Li-центров. Добавь данные через харвесторы.",
      answerVector:     [],
      confidence:       0,
      converged:        false,
      iterations:       0,
      participatingLi:  [],
      mirrorAgreement:  0,
      networkType:      input.networkType,
      timestamp:        Date.now(),
      processingMs:     Date.now() - start,
      knowledgeRecall:  [],
      recallUsed:       false,
    };
  }

  // ── State ─────────────────────────────────────────────────────────────────

  state(): OmegaState {
    const active = Array.from(this.clusters.entries())
      .filter(([, list]) => list.length > 0)
      .map(([type]) => type);

    return {
      id:              "omega",
      activeNetworks:  active,
      totalQueries:    this.queryCount,
      avgConfidence:   this.queryCount > 0 ? this.totalConfidence  / this.queryCount : 0,
      avgIterations:   this.queryCount > 0 ? this.totalIterations  / this.queryCount : 0,
      convergeRate:    this.queryCount > 0 ? this.convergedCount   / this.queryCount : 0,
    };
  }
}

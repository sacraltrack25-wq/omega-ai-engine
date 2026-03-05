/**
 * BaseNetwork — Abstract base for all 5 OMEGA neural networks
 * (TextNet, ImageNet, VideoNet, AudioNet, GameNet)
 */

import { LiCluster, NetworkType, KnowledgeMatch, Knowledge, LearnOptions } from "../centers/Li";
import { Omega, OmegaInput, OmegaTruth } from "../omega/Omega";
import { TRAINING_PARAMS } from "../training/parameters";

export interface NetworkConfig {
  id: string;
  type: NetworkType;
  clusterCount: number;
  coreCount: number;
}

export interface NetworkStats {
  id: string;
  type: NetworkType;
  clusters: number;
  totalNeurons: number;
  totalKnowledge: number;
  isReady: boolean;
}

export abstract class BaseNetwork {
  readonly id: string;
  readonly type: NetworkType;

  protected clusters: LiCluster[] = [];
  protected omega: Omega;
  protected isReady = false;

  constructor(config: NetworkConfig, omega: Omega) {
    this.id    = config.id;
    this.type  = config.type;
    this.omega = omega;
    this.initClusters(config);
  }

  private initClusters(config: NetworkConfig) {
    for (let i = 0; i < config.clusterCount; i++) {
      const cluster = new LiCluster(
        `${config.id}_C${i}`,
        config.type,
        config.coreCount,
      );
      this.clusters.push(cluster);
      this.omega.registerCluster(cluster);
    }
    this.isReady = true;
  }

  /**
   * Encode a raw input into a numeric vector.
   * Each network implements its own strategy.
   */
  abstract encode(input: unknown): Promise<number[]>;

  /**
   * Decode an Omega truth vector into a typed output.
   */
  abstract decode(truth: OmegaTruth): Promise<unknown>;

  /**
   * Main inference: encode → emit → decode
   */
  async infer(rawInput: unknown, context?: Record<string, unknown>): Promise<OmegaTruth> {
    const queryVector = await this.encode(rawInput);
    const query = typeof rawInput === "string" ? rawInput : JSON.stringify(rawInput).slice(0, 200);

    const omegaInput: OmegaInput = {
      query,
      queryVector,
      networkType: this.type,
      context,
    };

    return this.omega.emit(omegaInput);
  }

  /** Learn from a labelled example — raw is the original text/content */
  async learn(
    key: string,
    data: number[],
    source: string,
    raw?: string,
    restore?: LearnOptions,
  ): Promise<Knowledge | undefined> {
    const results = await Promise.all(
      this.clusters.map(c => c.learn(key, data, source, raw, restore)),
    );
    return results[0];
  }

  /**
   * Recall: search all Li clusters for knowledge matching queryVector.
   * Returns top-K matches sorted by cosine similarity × strength.
   * This is the core retrieval mechanism — turns stored vectors back into text.
   */
  recall(queryVector: number[], topK = 5): KnowledgeMatch[] {
    const allMatches = this.clusters.flatMap(c => c.recall(queryVector, topK));

    // Deduplicate by key, keep highest score
    const best = new Map<string, KnowledgeMatch>();
    for (const m of allMatches) {
      const prev = best.get(m.key);
      if (!prev || m.score > prev.score) best.set(m.key, m);
    }

    return Array.from(best.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /** Get all knowledge for persistence (dump Li RAM → Supabase). Dedupes across mirror pairs. */
  getAllKnowledgeForPersistence(): Knowledge[] {
    const seen = new Set<string>();
    const result: Knowledge[] = [];
    for (const cluster of this.clusters) {
      for (const center of cluster.allCenters) {
        for (const k of center.getAllKnowledgeForPersistence()) {
          if (seen.has(k.key)) continue;
          seen.add(k.key);
          result.push(k);
        }
      }
    }
    return result;
  }

  /** Network-wide memory consolidation */
  consolidate(): number {
    let pruned = 0;
    for (const cluster of this.clusters) {
      for (const center of cluster.allCenters) {
        pruned += center.consolidate();
      }
    }
    return pruned;
  }

  stats(): NetworkStats {
    let totalNeurons  = 0;
    let totalKnowledge = 0;
    for (const c of this.clusters) {
      for (const center of c.allCenters) {
        totalNeurons   += center.neuronCount;
        totalKnowledge += center.knowledgeSize;
      }
    }
    return {
      id: this.id,
      type: this.type,
      clusters: this.clusters.length,
      totalNeurons,
      totalKnowledge,
      isReady: this.isReady,
    };
  }
}

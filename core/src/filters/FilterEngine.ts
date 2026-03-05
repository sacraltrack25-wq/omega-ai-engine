/**
 * FilterEngine — гибкий управляемый фильтр запросов/ответов.
 *
 * Применяется в двух точках:
 *   1. PRE-QUERY  — до обработки AI (блок запросов, ограничения сети)
 *   2. POST-QUERY — после получения ответа (буст источников, минимальное качество)
 *
 * Фильтры загружаются из Supabase при старте и кэшируются.
 * Горячая перезагрузка: вызови FilterEngine.reload()
 *
 * Типы фильтров:
 *   content_block    — блок запросов по ключевым словам
 *   source_block     — игнорировать знания из указанных источников
 *   source_boost     — повышать score матчей из указанных источников
 *   quality_min      — минимальный resonanceScore для ответа
 *   network_restrict — сеть доступна только для определённого плана
 *   response_inject  — добавить текст к ответу
 */

import { FilterRow, knowledgeStore } from "../persistence/KnowledgeStore";
import { KnowledgeMatch } from "../centers/Li";
import { OmegaTruth } from "../omega/Omega";
import { NetworkType } from "../centers/Li";

export interface PreQueryResult {
  allowed:   boolean;
  reason?:   string;
  filterId?: string;
}

export interface PostQueryResult {
  truth:         OmegaTruth;
  filtersApplied: string[];  // names of applied filters
}

export class FilterEngine {
  private filters: FilterRow[] = [];
  private lastLoaded: number = 0;
  private readonly CACHE_TTL_MS = 60_000;  // reload every minute

  // ── Load / Reload ─────────────────────────────────────────────────────────

  async init(): Promise<void> {
    await this.reload();
  }

  async reload(): Promise<void> {
    this.filters   = await knowledgeStore.loadFilters();
    this.lastLoaded = Date.now();
    console.log(`[FilterEngine] Loaded ${this.filters.length} active filters.`);
  }

  private async ensureFresh(): Promise<void> {
    if (Date.now() - this.lastLoaded > this.CACHE_TTL_MS) {
      await this.reload();
    }
  }

  // ── PRE-QUERY filter ──────────────────────────────────────────────────────

  async preQuery(
    input: string,
    networkType: NetworkType,
    userPlan = "free",
  ): Promise<PreQueryResult> {
    await this.ensureFresh();

    const applicable = this.filters.filter(
      f => f.network_type === null || f.network_type === networkType,
    );

    for (const filter of applicable) {
      // ── content_block ──
      if (filter.type === "content_block") {
        const cfg = filter.config as { keywords?: string[]; match?: string };
        const keywords = cfg.keywords ?? [];
        const lower    = input.toLowerCase();
        const hits     = keywords.filter(kw => lower.includes(kw.toLowerCase()));

        const blocked = cfg.match === "all"
          ? hits.length === keywords.length && keywords.length > 0
          : hits.length > 0;

        if (blocked) {
          return {
            allowed:   false,
            reason:    `Запрос заблокирован фильтром "${filter.name}" (ключевые слова: ${hits.join(", ")})`,
            filterId:  filter.id,
          };
        }
      }

      // ── network_restrict ──
      if (filter.type === "network_restrict") {
        const cfg = filter.config as { network?: string; min_plan?: string };
        if (cfg.network === networkType) {
          const planRank: Record<string, number> = {
            free: 0, pro: 1, pro_unlimited: 2,
          };
          const userRank = planRank[userPlan] ?? 0;
          const minRank  = planRank[cfg.min_plan ?? "free"] ?? 0;
          if (userRank < minRank) {
            return {
              allowed:   false,
              reason:    `Сеть "${networkType}" требует план "${cfg.min_plan}". Твой план: ${userPlan}.`,
              filterId:  filter.id,
            };
          }
        }
      }
    }

    return { allowed: true };
  }

  // ── POST-QUERY filter ─────────────────────────────────────────────────────

  async postQuery(
    truth: OmegaTruth,
    networkType: NetworkType,
  ): Promise<PostQueryResult> {
    await this.ensureFresh();

    let modified     = { ...truth };
    const applied: string[] = [];

    const applicable = this.filters.filter(
      f => f.network_type === null || f.network_type === networkType,
    );

    for (const filter of applicable) {
      // ── quality_min ──
      if (filter.type === "quality_min") {
        const cfg = filter.config as { min_resonance?: number; fallback?: string };
        const minR = cfg.min_resonance ?? 0.3;
        const bestScore = truth.knowledgeRecall?.[0]?.resonanceScore ?? 0;

        if (bestScore < minR && truth.recallUsed) {
          applied.push(filter.name);
          if (cfg.fallback === "no_answer") {
            modified = {
              ...modified,
              answer: `[gX резонанс ${(bestScore * 100).toFixed(0)}% ниже минимума ${(minR * 100).toFixed(0)}%. Требуется более точное обучение по этой теме.]`,
              recallUsed: false,
            };
          }
          // "best_effort" — оставляем ответ как есть
        }
      }

      // ── source_boost ──
      if (filter.type === "source_boost" && truth.knowledgeRecall?.length) {
        const cfg = filter.config as { sources?: string[]; multiplier?: number };
        const sources    = cfg.sources ?? [];
        const multiplier = cfg.multiplier ?? 1.3;

        const boosted = truth.knowledgeRecall.map(m => {
          const isBoosted = sources.some(s => m.source.toLowerCase().includes(s.toLowerCase()));
          return isBoosted
            ? { ...m, score: Math.min(1, m.score * multiplier) }
            : m;
        }).sort((a, b) => b.score - a.score);

        if (JSON.stringify(boosted) !== JSON.stringify(truth.knowledgeRecall)) {
          applied.push(filter.name);
          modified = { ...modified, knowledgeRecall: boosted };
          // Re-select best answer if boosting changed the top result
          const bestBoosted = boosted.find(m => m.raw && m.score >= 0.3);
          if (bestBoosted?.raw && bestBoosted.source !== truth.knowledgeRecall[0]?.source) {
            modified = { ...modified, answer: bestBoosted.raw, recallUsed: true };
          }
        }
      }

      // ── source_block ──
      if (filter.type === "source_block" && truth.knowledgeRecall?.length) {
        const cfg = filter.config as { sources?: string[] };
        const blocked = cfg.sources ?? [];

        const filtered = truth.knowledgeRecall.filter(
          m => !blocked.some(s => m.source.toLowerCase().includes(s.toLowerCase())),
        );

        if (filtered.length !== truth.knowledgeRecall.length) {
          applied.push(filter.name);
          modified = { ...modified, knowledgeRecall: filtered };
        }
      }

      // ── response_inject ──
      if (filter.type === "response_inject") {
        const cfg = filter.config as { text?: string; position?: string };
        if (cfg.text && truth.answer && !truth.answer.startsWith("[")) {
          applied.push(filter.name);
          if (cfg.position === "before") {
            modified = { ...modified, answer: `${cfg.text}\n\n${modified.answer}` };
          } else {
            modified = { ...modified, answer: `${modified.answer}\n\n${cfg.text}` };
          }
        }
      }
    }

    return { truth: modified, filtersApplied: applied };
  }

  // ── Boost recall by knowledge strength ───────────────────────────────────

  /**
   * Apply Supabase knowledge strength to recall scores.
   * Entries that have been positively rated get score boost.
   * Called after recall, before Omega decode.
   */
  applyStrengthBoost(recall: KnowledgeMatch[]): KnowledgeMatch[] {
    // This is a simple boost — in a full implementation we'd
    // look up each key in li_knowledge to get its boost_score.
    // For now we rely on the strength already embedded in Li centers.
    return recall;
  }

  get filterCount() { return this.filters.length; }
}

export const filterEngine = new FilterEngine();

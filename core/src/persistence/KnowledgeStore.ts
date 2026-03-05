/**
 * KnowledgeStore — Supabase persistence layer for Li knowledge.
 *
 * Цикл данных (откуда и куда идёт каждый тип данных):
 * ─────────────────────────────────────────────────────────────────────────────
 *  Харвестер → POST /learn → Li (RAM)
 *    └─ KnowledgeStore.enqueue()
 *         └─ Supabase table: li_knowledge
 *              (key, network_type, vector, fingerprint, raw, source, strength)
 *
 *  AI Engine startup → KnowledgeStore.loadAll()
 *    └─ Supabase table: li_knowledge  →  Li центры в RAM
 *
 *  Консолидация (каждые N запросов) → KnowledgeStore.exportSnapshot()
 *    └─ Supabase Storage: knowledge-export/snapshot_{timestamp}.json
 *         Содержит: ВСЕ li_knowledge строки в одном JSON файле
 *         Назначение: быстрый restore без постраничной загрузки из БД
 *
 *  FilterEngine.init() → KnowledgeStore.loadFilters()
 *    └─ Supabase table: filters
 *
 * Env vars:
 *   SUPABASE_URL или NEXT_PUBLIC_SUPABASE_URL — URL проекта
 *   SUPABASE_SERVICE_ROLE_KEY                — service role ключ
 *   SUPABASE_BUCKET_KNOWLEDGE_EXPORT         — имя bucket (default: "knowledge-export")
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { NetworkType } from "../centers/Li";

export interface KnowledgeRow {
  key:           string;
  network_type:  NetworkType;
  vector:        number[];
  fingerprint:   number[];
  raw?:          string;
  source:        string;
  strength:      number;
  access_count:  number;
  quality:       number;
  last_accessed?: string;  // ISO timestamp
}

export class KnowledgeStore {
  private client: SupabaseClient | null = null;
  private enabled = false;

  // Bucket name: env SUPABASE_BUCKET_KNOWLEDGE_EXPORT → default "knowledge-export"
  private readonly exportBucket: string;

  // Write buffer — flush in batches
  private writeBuffer: KnowledgeRow[] = [];
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly FLUSH_INTERVAL_MS = 5000;   // flush every 5 seconds
  private readonly BATCH_SIZE        = 100;    // max rows per upsert

  constructor() {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL
             ?? process.env.SUPABASE_URL;
    const key = process.env.SUPABASE_SERVICE_ROLE_KEY;

    this.exportBucket = process.env.SUPABASE_BUCKET_KNOWLEDGE_EXPORT ?? "knowledge-export";

    if (url && key) {
      this.client  = createClient(url, key);
      this.enabled = true;
    } else {
      console.warn("[KnowledgeStore] Supabase not configured — running without persistence.");
      console.warn("[KnowledgeStore] Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY to enable.");
    }
  }

  // ── Write ─────────────────────────────────────────────────────────────────

  /**
   * Queue a knowledge entry for saving.
   * Non-blocking — immediately returns; data is flushed in background.
   */
  enqueue(row: KnowledgeRow): void {
    if (!this.enabled) return;

    // Overwrite if same key+network already queued
    const existing = this.writeBuffer.findIndex(
      r => r.key === row.key && r.network_type === row.network_type,
    );
    if (existing >= 0) {
      this.writeBuffer[existing] = row;
    } else {
      this.writeBuffer.push(row);
    }

    if (this.writeBuffer.length >= this.BATCH_SIZE) {
      this.flush();
    } else {
      this.scheduleFlush();
    }
  }

  private scheduleFlush() {
    if (this.flushTimer) return;
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null;
      this.flush();
    }, this.FLUSH_INTERVAL_MS);
  }

  /**
   * Flush all buffered rows to Supabase.
   * Use after learn-batch to persist immediately.
   */
  async flushAll(): Promise<void> {
    while (this.writeBuffer.length > 0) {
      await this.flush();
    }
  }

  async flush(): Promise<void> {
    if (!this.enabled || !this.client || this.writeBuffer.length === 0) return;

    const batch = this.writeBuffer.splice(0, this.BATCH_SIZE);

    try {
      const rows = batch.map(r => ({
        key:           r.key,
        network_type:  r.network_type,
        vector:        r.vector,
        fingerprint:   r.fingerprint,
        raw:           r.raw ?? null,
        source:        r.source,
        strength:      r.strength,
        access_count:  r.access_count,
        quality:       r.quality,
        last_accessed: r.last_accessed ?? new Date().toISOString(),
      }));

      const { error } = await this.client
        .from("li_knowledge")
        .upsert(rows, { onConflict: "key,network_type" });

      if (error) {
        console.error("[KnowledgeStore] Flush error:", error.message);
        // Put failed rows back in buffer
        this.writeBuffer.unshift(...batch);
      }
    } catch (err) {
      console.error("[KnowledgeStore] Flush exception:", err);
      this.writeBuffer.unshift(...batch);
    }
  }

  // ── Read / Restore ────────────────────────────────────────────────────────

  /**
   * Load all knowledge from Supabase.
   * Called once at AI Engine startup to restore Li centers.
   * Returns rows grouped by network_type.
   */
  async loadAll(): Promise<Map<NetworkType, KnowledgeRow[]>> {
    const result = new Map<NetworkType, KnowledgeRow[]>();
    if (!this.enabled || !this.client) return result;

    let offset = 0;
    const PAGE  = 1000;

    console.log("[KnowledgeStore] Loading knowledge from Supabase...");

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { data, error } = await this.client
        .from("li_knowledge")
        .select("key, network_type, vector, fingerprint, raw, source, strength, access_count, quality, last_accessed")
        .order("strength", { ascending: false })  // load strongest first
        .range(offset, offset + PAGE - 1);

      if (error) {
        console.error("[KnowledgeStore] Load error:", error.message);
        break;
      }

      if (!data || data.length === 0) break;

      for (const row of data) {
        const type = row.network_type as NetworkType;
        const list = result.get(type) ?? [];
        list.push({
          key:           row.key,
          network_type:  type,
          vector:        row.vector as number[],
          fingerprint:   row.fingerprint as number[],
          raw:           row.raw ?? undefined,
          source:        row.source,
          strength:      row.strength,
          access_count:  row.access_count,
          quality:       row.quality,
          last_accessed: (row as { last_accessed?: string }).last_accessed ?? undefined,
        });
        result.set(type, list);
      }

      offset += PAGE;
      if (data.length < PAGE) break;
    }

    const total = Array.from(result.values()).reduce((s, a) => s + a.length, 0);
    console.log(`[KnowledgeStore] Loaded ${total} knowledge entries from Supabase.`);

    return result;
  }

  /**
   * Load active filters from Supabase.
   * Used by FilterEngine at startup.
   */
  async loadFilters(networkType?: string): Promise<FilterRow[]> {
    if (!this.enabled || !this.client) return [];

    let q = this.client
      .from("filters")
      .select("id, name, type, config, priority, network_type")
      .eq("is_active", true)
      .order("priority");

    if (networkType) {
      q = (q as typeof q).or(`network_type.is.null,network_type.eq.${networkType}`);
    }

    const { data, error } = await q;
    if (error) {
      console.error("[KnowledgeStore] Load filters error:", error.message);
      return [];
    }
    return (data ?? []) as FilterRow[];
  }

  // ── Snapshot Export ───────────────────────────────────────────────────────

  /**
   * Экспортирует все знания из li_knowledge в Storage bucket.
   *
   * Куда пишет:
   *   Supabase Storage: knowledge-export/snapshot_{timestamp}.json
   *
   * Зачем:
   *   Быстрый restore AI Engine без постраничного чтения тысяч строк из БД.
   *   Один JSON файл → один запрос к Storage вместо 1000+ к БД.
   *
   * Когда вызывается:
   *   - После MirrorAI.consolidate() (каждые N запросов)
   *   - При graceful shutdown (SIGTERM/SIGINT)
   *
   * Формат файла:
   *   { version: 1, exported_at: ISO, rows: KnowledgeRow[] }
   */
  async exportSnapshot(): Promise<string | null> {
    if (!this.enabled || !this.client) return null;

    // Сначала сбросить буфер чтобы снимок был актуальным
    await this.flush();

    const { data, error } = await this.client
      .from("li_knowledge")
      .select("key, network_type, vector, fingerprint, raw, source, strength, access_count, quality, last_accessed")
      .order("strength", { ascending: false });

    if (error || !data) {
      console.error("[KnowledgeStore] Snapshot read error:", error?.message);
      return null;
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const path      = `snapshot_${timestamp}.json`;
    const payload   = JSON.stringify({ version: 1, exported_at: new Date().toISOString(), rows: data });

    const { error: uploadError } = await this.client.storage
      .from(this.exportBucket)
      .upload(path, payload, { contentType: "application/json", upsert: true });

    if (uploadError) {
      console.error("[KnowledgeStore] Snapshot upload error:", uploadError.message);
      return null;
    }

    console.log(`[KnowledgeStore] Snapshot saved → ${this.exportBucket}/${path} (${data.length} rows)`);
    return path;
  }

  /**
   * Восстанавливает знания из последнего снимка в Storage.
   * Быстрее loadAll() для больших баз знаний.
   * Если снимка нет — fallback на loadAll().
   */
  async restoreFromSnapshot(): Promise<Map<NetworkType, KnowledgeRow[]> | null> {
    if (!this.enabled || !this.client) return null;

    // Список снимков: берём самый свежий
    const { data: files, error } = await this.client.storage
      .from(this.exportBucket)
      .list("", { limit: 50, sortBy: { column: "name", order: "desc" } });

    if (error || !files || files.length === 0) return null;

    const latest = files.find(f => f.name.startsWith("snapshot_") && f.name.endsWith(".json"));
    if (!latest) return null;

    const { data: blob, error: dlError } = await this.client.storage
      .from(this.exportBucket)
      .download(latest.name);

    if (dlError || !blob) {
      console.warn("[KnowledgeStore] Snapshot download failed:", dlError?.message);
      return null;
    }

    try {
      const text   = await blob.text();
      const parsed = JSON.parse(text) as { version: number; rows: KnowledgeRow[] };
      const result = new Map<NetworkType, KnowledgeRow[]>();

      for (const row of parsed.rows) {
        const list = result.get(row.network_type) ?? [];
        list.push(row);
        result.set(row.network_type, list);
      }

      const total = parsed.rows.length;
      console.log(`[KnowledgeStore] Restored ${total} rows from snapshot ${latest.name}`);
      return result;
    } catch (e) {
      console.warn("[KnowledgeStore] Snapshot parse failed:", e);
      return null;
    }
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────

  async shutdown(): Promise<void> {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    await this.flushAll();
    // Сохранить финальный снимок при выключении
    await this.exportSnapshot();
    console.log("[KnowledgeStore] Flushed and shut down.");
  }
}

export interface FilterRow {
  id:           string;
  name:         string;
  type:         string;
  config:       Record<string, unknown>;
  priority:     number;
  network_type: string | null;
}

// Singleton
export const knowledgeStore = new KnowledgeStore();

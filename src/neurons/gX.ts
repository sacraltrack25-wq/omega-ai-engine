/**
 * gX — Mirror Neuron  (upgraded: weighted resonance + Hebbian adaptation)
 *
 * Фундаментальная единица OMEGA AI.
 * 1 нейрон = зеркальная пара gX1 ↔ gX2.
 *
 * Mirror Principle:
 *   const gX1  →  primary state  (IMMUTABLE binary seed)
 *   const gX2  →  mirror state   (IMMUTABLE, always gX1 XOR 1)
 *
 * Улучшения над простым бинарным нейроном:
 *   1. WEIGHT  — важность нейрона [0.1 .. 3.0]
 *      Нейроны которые часто попадают в резонанс — сильнее влияют на ответ.
 *      Нейроны которые постоянно ошибаются — слабеют.
 *      → Hebbian learning: "fire together, wire together"
 *
 *   2. ASYMMETRIC score — первичный резонанс сильнее чем анти-резонанс:
 *      primary:  +weight       (нашли совпадение)
 *      mirror:   -weight * 0.1 (мягкое подавление при несовпадении)
 *      Это ключевое: нейрон НЕ наказывает резко за ошибку —
 *      он просто немного сомневается.
 *
 *   3. TEMPERATURE — глобальный параметр остроты выбора:
 *      T → 0: нейрон очень уверен (бинарный режим)
 *      T → ∞: нейрон мягкий, принимает неточные совпадения
 *
 * Резонанс слоя теперь float, не bool[] — это позволяет Li
 * хранить непрерывные fingerprints вместо бинарных.
 */

export type Bit = 0 | 1;

export interface NeuronState {
  id:             string;
  gX1:            Bit;
  gX2:            Bit;
  weight:         number;
  layer:          number;
  position:       number;
  activations:    number;
  resonanceScore: number;
}

export interface NeuronResponse {
  primary:  boolean;   // точное совпадение с gX1
  mirror:   boolean;   // совпадение с gX2
  score:    number;    // взвешенный вклад: +weight | -weight*0.1
  weight:   number;    // текущий вес нейрона
}

// ─── Mirror Neuron ─────────────────────────────────────────────────────────────

export class MirrorNeuron {
  // CONST — бинарные зеркальные состояния. Никогда не меняются.
  readonly id:       string;
  readonly gX1:      Bit;
  readonly gX2:      Bit;
  readonly layer:    number;
  readonly position: number;

  // Mutable: адаптивный вес (Hebbian)
  private _weight:         number;
  private _activations:    number = 0;
  private _resonance:      number = 0;
  private _consecutiveHits = 0;   // streak of correct activations

  private static readonly MIN_WEIGHT  = 0.1;
  private static readonly MAX_WEIGHT  = 3.0;
  private static readonly LEARN_UP    = 1.008;  // вес растёт при попадании
  private static readonly LEARN_DOWN  = 0.996;  // вес падает при промахе
  private static readonly STREAK_BONUS = 0.01;  // бонус за серию попаданий

  constructor(id: string, value: Bit, layer = 0, position = 0) {
    this.id       = id;
    this.gX1      = value;
    this.gX2      = (value ^ 1) as Bit;
    this.layer    = layer;
    this.position = position;
    // Инициализируем вес с небольшим случайным разбросом
    this._weight  = 0.8 + Math.random() * 0.4;   // [0.8 .. 1.2]
  }

  /**
   * Полный зеркальный ответ на входной бит.
   * Возвращает взвешенный вклад нейрона в резонанс слоя.
   *
   * Asymmetric scoring:
   *   primary resonance:  +weight       (сильный положительный сигнал)
   *   mirror anti-res:    -weight * 0.1 (слабое подавление)
   *
   * Это ключевое отличие от трансформеров: нейрон НЕ наказывает жёстко
   * за несовпадение — он просто слегка сомневается.
   */
  respond(input: Bit): NeuronResponse {
    const primary = input === this.gX1;
    const mirror  = input === this.gX2;

    if (primary) {
      this._activations++;
      this._consecutiveHits++;
      // Hebbian: нейрон который часто резонирует — усиливается
      this._weight = Math.min(
        MirrorNeuron.MAX_WEIGHT,
        this._weight * MirrorNeuron.LEARN_UP + this._consecutiveHits * MirrorNeuron.STREAK_BONUS,
      );
      this._resonance += this._weight;
    } else {
      this._consecutiveHits = 0;
      // Мягкое ослабление при промахе
      this._weight    = Math.max(MirrorNeuron.MIN_WEIGHT, this._weight * MirrorNeuron.LEARN_DOWN);
      this._resonance -= this._weight * 0.1;
    }

    return {
      primary,
      mirror,
      score:  primary ? this._weight : -this._weight * 0.1,
      weight: this._weight,
    };
  }

  /** Простая активация без обучения веса (для inference без обратной связи) */
  activate(input: Bit): boolean {
    return input === this.gX1;
  }

  /** Принудительно усилить нейрон (внешний feedback 👍) */
  strengthen(factor = 1.05): void {
    this._weight = Math.min(MirrorNeuron.MAX_WEIGHT, this._weight * factor);
  }

  /** Принудительно ослабить нейрон (внешний feedback 👎) */
  weaken(factor = 0.95): void {
    this._weight = Math.max(MirrorNeuron.MIN_WEIGHT, this._weight * factor);
  }

  get activations()    { return this._activations; }
  get resonanceScore() { return this._resonance; }
  get weight()         { return this._weight; }

  snapshot(): NeuronState {
    return {
      id:             this.id,
      gX1:            this.gX1,
      gX2:            this.gX2,
      weight:         this._weight,
      layer:          this.layer,
      position:       this.position,
      activations:    this._activations,
      resonanceScore: this._resonance,
    };
  }
}


// ─── Neuron Layer ──────────────────────────────────────────────────────────────

export interface LayerResult {
  activations:        boolean[];
  mirrorActivations:  boolean[];
  scores:             number[];    // непрерывные взвешенные оценки каждого нейрона
  resonance:          number;      // среднее взвешенное по всему слою [-1, 1]
  weightedResonance:  number;      // сумма score / сумма weight (нормировано)
  primaryRate:        number;
  mirrorRate:         number;
  totalWeight:        number;      // сумма весов — показывает "мощность" слоя
}

export class NeuronLayer {
  readonly id:    string;
  readonly depth: number;
  private neurons: MirrorNeuron[] = [];

  constructor(id: string, depth: number, size: number) {
    this.id    = id;
    this.depth = depth;
    this.seed(size);
  }

  private seed(count: number) {
    for (let i = 0; i < count; i++) {
      const bit = (Math.random() > 0.5 ? 1 : 0) as Bit;
      this.neurons.push(new MirrorNeuron(`${this.id}_gX${i}`, bit, this.depth, i));
    }
  }

  /**
   * Forward pass — все нейроны отвечают на входные биты.
   *
   * Ключевое улучшение: resonance теперь weightedResonance:
   *   Σ(score_i) / Σ(weight_i)
   * Это как attention: нейроны с большим весом доминируют.
   * Простой average был бы менее информативен.
   */
  process(inputs: Bit[]): LayerResult {
    const len                  = Math.min(inputs.length, this.neurons.length);
    const activations:        boolean[] = [];
    const mirrorActivations:  boolean[] = [];
    const scores:             number[]  = [];

    let totalScore  = 0;
    let totalWeight = 0;

    for (let i = 0; i < len; i++) {
      const r = this.neurons[i].respond(inputs[i]);
      activations.push(r.primary);
      mirrorActivations.push(r.mirror);
      scores.push(r.score);
      totalScore  += r.score;
      totalWeight += r.weight;
    }

    const resonance         = len > 0 ? totalScore  / len         : 0;
    const weightedResonance = totalWeight > 0 ? totalScore / totalWeight : 0;
    const primaryFires      = activations.filter(Boolean).length;
    const mirrorFires       = mirrorActivations.filter(Boolean).length;

    return {
      activations,
      mirrorActivations,
      scores,
      resonance,
      weightedResonance,   // ← это основная метрика качества
      primaryRate:   len > 0 ? primaryFires / len : 0,
      mirrorRate:    len > 0 ? mirrorFires  / len : 0,
      totalWeight,
    };
  }

  /**
   * Grow — добавить новые нейроны.
   * Новые нейроны стартуют со средним весом существующих
   * (не с нуля — быстрее адаптируются).
   */
  grow(count: number) {
    const base = this.neurons.length;
    const avgWeight = base > 0
      ? this.neurons.reduce((s, n) => s + n.weight, 0) / base
      : 1.0;

    for (let i = 0; i < count; i++) {
      const bit    = (Math.random() > 0.5 ? 1 : 0) as Bit;
      const neuron = new MirrorNeuron(`${this.id}_gX${base + i}`, bit, this.depth, base + i);
      // Инициализируем вес близко к среднему (с небольшим шумом)
      const noise = (Math.random() - 0.5) * 0.2;
      if (neuron.weight !== avgWeight + noise) {
        // Force initial weight via strengthen/weaken
        const ratio = (avgWeight + noise) / neuron.weight;
        if (ratio > 1) neuron.strengthen(ratio);
        else if (ratio < 1) neuron.weaken(1 / ratio);
      }
      this.neurons.push(neuron);
    }
  }

  /** Применить внешний feedback ко всем нейронам которые активировались */
  applyFeedback(activations: boolean[], positive: boolean) {
    for (let i = 0; i < Math.min(activations.length, this.neurons.length); i++) {
      if (activations[i]) {
        if (positive) this.neurons[i].strengthen();
        else          this.neurons[i].weaken();
      }
    }
  }

  get size() { return this.neurons.length; }

  snapshots(): NeuronState[] {
    return this.neurons.map(n => n.snapshot());
  }
}

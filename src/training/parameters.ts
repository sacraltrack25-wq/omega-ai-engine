/**
 * Training Parameters — OMEGA AI
 *
 * All regulators, criteria, and depth settings for the gX-Li-Ω engine.
 * Parameters are loaded from environment variables with sensible defaults.
 */

// ─── Helper ───────────────────────────────────────────────────────────────────

function env(key: string, fallback: number): number {
  const v = process.env[key];
  return v !== undefined && !isNaN(Number(v)) ? Number(v) : fallback;
}
function envBool(key: string, fallback: boolean): boolean {
  const v = process.env[key];
  if (v === undefined) return fallback;
  return v === "true" || v === "1";
}

// ─── Parameter Definitions ────────────────────────────────────────────────────

export interface TrainingParameters {
  // ── Learning ────────────────────────────────────────────────────────────────
  /** How fast the network adapts to new data [0.0001–0.1] */
  LEARNING_RATE: number;
  /** Inertia of gradient updates [0–1] */
  MOMENTUM: number;
  /** Slow decay of older weights [0–0.01] */
  WEIGHT_DECAY: number;
  /** Noise injection to prevent overfitting [0–0.5] */
  DROPOUT_RATE: number;
  /** L2 regularization strength [0–0.001] */
  REGULARIZATION_STRENGTH: number;
  /** Samples per training step */
  BATCH_SIZE: number;
  /** Gradual ramp-up steps before full learning rate */
  WARMUP_STEPS: number;

  // ── Mirror Principle ────────────────────────────────────────────────────────
  /** How tightly gX1 and gX2 stay synchronized [0–1] */
  MIRROR_SYNC_RATE: number;
  /** Max allowed divergence between mirror pairs before re-sync [0–0.5] */
  MIRROR_DIVERGENCE_TOLERANCE: number;
  /** Require both mirror Li centers to agree before output */
  MIRROR_AGREEMENT_REQUIRED: boolean;

  // ── gX Neuron Architecture ──────────────────────────────────────────────────
  /** Number of gX neurons per layer */
  NEURON_DENSITY: number;
  /** Number of gX layers per Li center */
  LAYER_DEPTH: number;

  // ── Li Processing Centers ───────────────────────────────────────────────────
  /** Number of Li core pairs per cluster */
  LI_CORE_COUNT: number;
  /** Growth fraction when Li needs to expand [0–0.5] */
  LI_GROWTH_RATE: number;
  /** Cosine similarity threshold to merge redundant Li centers [0–1] */
  LI_MERGE_THRESHOLD: number;
  /** Knowledge entries per Li before a new layer is added */
  LI_LAYER_SPAWN_THRESHOLD: number;
  /** Minimum Li centers that must participate in a query */
  LI_PARTICIPATION_MIN: number;

  // ── Omega Truth Center ──────────────────────────────────────────────────────
  /** Confidence required to accept Omega output as truth [0–1] */
  OMEGA_CONVERGENCE_THRESHOLD: number;
  /** Max self-validation iterations */
  OMEGA_MAX_ITERATIONS: number;
  /** Depth of Omega's self-mirroring validation loop */
  OMEGA_SELF_VALIDATION_DEPTH: number;
  /** Minimum confidence for Omega to emit a final answer [0–1] */
  TRUTH_CONFIDENCE_THRESHOLD: number;

  // ── Memory & Storage ────────────────────────────────────────────────────────
  /** Working memory (knowledge cache) entries */
  CACHE_SIZE: number;
  /** Steps between memory consolidation sweeps */
  MEMORY_CONSOLIDATION_CYCLE: number;

  // ── Data & Harvesters ───────────────────────────────────────────────────────
  /** Minimum quality score to accept harvested data [0–1] */
  DATA_QUALITY_THRESHOLD: number;
  /** Cosine similarity above which data is considered duplicate [0–1] */
  DEDUPLICATION_THRESHOLD: number;

  // ── Generative/Creative ─────────────────────────────────────────────────────
  /** Balance between determinism (0) and creativity (1) */
  ENTROPY_REGULATION: number;
  /** Minimum pattern signal strength to learn [0–1] */
  PATTERN_SENSITIVITY: number;
  /** Confidence above which an input is flagged as anomalous [0–1] */
  ANOMALY_DETECTION_THRESHOLD: number;

  // ── Cross-Network ───────────────────────────────────────────────────────────
  /** Fraction of learned features shared across networks [0–1] */
  CROSS_NETWORK_TRANSFER: number;

  // ── Network-Specific ────────────────────────────────────────────────────────
  /** TextNet: token context window */
  CONTEXT_WINDOW: number;
  /** VideoNet / AudioNet: temporal look-back window (frames / samples) */
  TEMPORAL_DEPTH: number;
  /** ImageNet / VideoNet / GameNet: spatial detail level [1–8] */
  SPATIAL_RESOLUTION: number;
  /** AudioNet: number of frequency bands */
  FREQUENCY_RESOLUTION: number;
  /** GameNet: physics + visual realism level [1–5] */
  REALISM_LEVEL: number;
}

// ─── TRAINING_PARAMS singleton ────────────────────────────────────────────────

export const TRAINING_PARAMS: TrainingParameters = {
  // Learning
  LEARNING_RATE:              env("LEARNING_RATE",              0.001),
  MOMENTUM:                   env("MOMENTUM",                   0.9),
  WEIGHT_DECAY:               env("WEIGHT_DECAY",               0.01),
  DROPOUT_RATE:               env("DROPOUT_RATE",               0.1),
  REGULARIZATION_STRENGTH:    env("REGULARIZATION_STRENGTH",    0.0001),
  BATCH_SIZE:                 env("BATCH_SIZE",                  32),
  WARMUP_STEPS:               env("WARMUP_STEPS",               500),

  // Mirror Principle
  MIRROR_SYNC_RATE:             env("MIRROR_SYNC_RATE",              0.95),
  MIRROR_DIVERGENCE_TOLERANCE:  env("MIRROR_DIVERGENCE_TOLERANCE",   0.15),
  MIRROR_AGREEMENT_REQUIRED:    envBool("MIRROR_AGREEMENT_REQUIRED", true),

  // gX Architecture
  NEURON_DENSITY: env("NEURON_DENSITY", 128),
  LAYER_DEPTH:    env("LAYER_DEPTH",    6),

  // Li Centers
  LI_CORE_COUNT:             env("LI_CORE_COUNT",              8),
  LI_GROWTH_RATE:            env("LI_GROWTH_RATE",             0.1),
  LI_MERGE_THRESHOLD:        env("LI_MERGE_THRESHOLD",         0.85),
  LI_LAYER_SPAWN_THRESHOLD:  env("LI_LAYER_SPAWN_THRESHOLD",   1000),
  LI_PARTICIPATION_MIN:      env("LI_PARTICIPATION_MIN",       2),

  // Omega
  OMEGA_CONVERGENCE_THRESHOLD: env("OMEGA_CONVERGENCE_THRESHOLD", 0.95),
  OMEGA_MAX_ITERATIONS:        env("OMEGA_MAX_ITERATIONS",        10),
  OMEGA_SELF_VALIDATION_DEPTH: env("OMEGA_SELF_VALIDATION_DEPTH", 3),
  TRUTH_CONFIDENCE_THRESHOLD:  env("TRUTH_CONFIDENCE_THRESHOLD",  0.75),

  // Memory
  CACHE_SIZE:                  env("CACHE_SIZE",                 10000),
  MEMORY_CONSOLIDATION_CYCLE:  env("MEMORY_CONSOLIDATION_CYCLE", 1000),

  // Data
  DATA_QUALITY_THRESHOLD:  env("DATA_QUALITY_THRESHOLD",  0.7),
  DEDUPLICATION_THRESHOLD: env("DEDUPLICATION_THRESHOLD", 0.92),

  // Creative
  ENTROPY_REGULATION:          env("ENTROPY_REGULATION",          0.5),
  PATTERN_SENSITIVITY:         env("PATTERN_SENSITIVITY",         0.3),
  ANOMALY_DETECTION_THRESHOLD: env("ANOMALY_DETECTION_THRESHOLD", 0.85),

  // Cross-network
  CROSS_NETWORK_TRANSFER: env("CROSS_NETWORK_TRANSFER", 0.3),

  // Network-specific
  CONTEXT_WINDOW:      env("CONTEXT_WINDOW",      4096),
  TEMPORAL_DEPTH:      env("TEMPORAL_DEPTH",       64),
  SPATIAL_RESOLUTION:  env("SPATIAL_RESOLUTION",   4),
  FREQUENCY_RESOLUTION:env("FREQUENCY_RESOLUTION", 256),
  REALISM_LEVEL:       env("REALISM_LEVEL",         3),
};

export default TRAINING_PARAMS;

// ─── Parameter metadata (for Admin UI) ───────────────────────────────────────

export interface ParamMeta {
  key: keyof TrainingParameters;
  label: string;
  description: string;
  min: number;
  max: number;
  step: number;
  group: string;
}

export const PARAM_META: ParamMeta[] = [
  { key: "LEARNING_RATE",              label: "Learning Rate",             description: "How fast the network adapts to new data",                  min: 0.00001, max: 0.1,  step: 0.0001,  group: "Learning"       },
  { key: "MOMENTUM",                   label: "Momentum",                  description: "Inertia of gradient updates",                              min: 0,       max: 1,    step: 0.01,    group: "Learning"       },
  { key: "WEIGHT_DECAY",               label: "Weight Decay",              description: "Slow decay of older weights",                              min: 0,       max: 0.01, step: 0.0001,  group: "Learning"       },
  { key: "DROPOUT_RATE",               label: "Dropout Rate",              description: "Noise injection to prevent overfitting",                   min: 0,       max: 0.5,  step: 0.01,    group: "Learning"       },
  { key: "BATCH_SIZE",                 label: "Batch Size",                description: "Samples per training step",                                min: 1,       max: 512,  step: 1,       group: "Learning"       },
  { key: "WARMUP_STEPS",               label: "Warmup Steps",              description: "Gradual ramp-up before full learning rate",                min: 0,       max: 5000, step: 10,      group: "Learning"       },
  { key: "MIRROR_SYNC_RATE",           label: "Mirror Sync Rate",          description: "How tightly mirror pairs stay synchronized",               min: 0,       max: 1,    step: 0.01,    group: "Mirror"         },
  { key: "MIRROR_DIVERGENCE_TOLERANCE",label: "Mirror Divergence Tol.",    description: "Max divergence between mirror pairs before re-sync",       min: 0,       max: 0.5,  step: 0.01,    group: "Mirror"         },
  { key: "NEURON_DENSITY",             label: "Neuron Density",            description: "Number of gX neurons per layer",                           min: 16,      max: 2048, step: 16,      group: "Architecture"   },
  { key: "LAYER_DEPTH",                label: "Layer Depth",               description: "Number of gX layers per Li center",                        min: 1,       max: 32,   step: 1,       group: "Architecture"   },
  { key: "LI_CORE_COUNT",              label: "Li Core Count",             description: "Number of Li core pairs per cluster",                      min: 2,       max: 64,   step: 2,       group: "Li Centers"     },
  { key: "LI_GROWTH_RATE",             label: "Li Growth Rate",            description: "Growth fraction when Li needs to expand",                  min: 0,       max: 0.5,  step: 0.01,    group: "Li Centers"     },
  { key: "LI_MERGE_THRESHOLD",         label: "Li Merge Threshold",        description: "Similarity threshold to merge redundant Li centers",       min: 0.5,     max: 1,    step: 0.01,    group: "Li Centers"     },
  { key: "OMEGA_CONVERGENCE_THRESHOLD",label: "Ω Convergence Threshold",   description: "Confidence required to accept truth",                      min: 0.5,     max: 1,    step: 0.01,    group: "Omega"          },
  { key: "OMEGA_MAX_ITERATIONS",       label: "Ω Max Iterations",          description: "Maximum self-validation iterations",                       min: 1,       max: 50,   step: 1,       group: "Omega"          },
  { key: "OMEGA_SELF_VALIDATION_DEPTH",label: "Ω Validation Depth",        description: "Depth of self-mirroring validation loop",                  min: 1,       max: 10,   step: 1,       group: "Omega"          },
  { key: "TRUTH_CONFIDENCE_THRESHOLD", label: "Truth Confidence Threshold",description: "Minimum confidence to emit a final answer",                min: 0.5,     max: 1,    step: 0.01,    group: "Omega"          },
  { key: "ENTROPY_REGULATION",         label: "Entropy (Creativity)",      description: "Balance between determinism (0) and creativity (1)",       min: 0,       max: 1,    step: 0.01,    group: "Creative"       },
  { key: "PATTERN_SENSITIVITY",        label: "Pattern Sensitivity",       description: "Minimum pattern signal strength to learn",                 min: 0,       max: 1,    step: 0.01,    group: "Creative"       },
  { key: "CROSS_NETWORK_TRANSFER",     label: "Cross-Network Transfer",    description: "Fraction of features shared across networks",              min: 0,       max: 1,    step: 0.01,    group: "Cross-Network"  },
  { key: "CONTEXT_WINDOW",             label: "Context Window (Text)",     description: "Token context window for TextNet",                         min: 256,     max: 32768,step: 256,     group: "Network-Specific"},
  { key: "TEMPORAL_DEPTH",             label: "Temporal Depth",            description: "Look-back window for Video/Audio (frames/samples)",        min: 8,       max: 512,  step: 8,       group: "Network-Specific"},
  { key: "SPATIAL_RESOLUTION",         label: "Spatial Resolution",        description: "Detail level for Image/Video/Game [1–8]",                  min: 1,       max: 8,    step: 1,       group: "Network-Specific"},
  { key: "FREQUENCY_RESOLUTION",       label: "Frequency Resolution",      description: "Number of audio frequency bands",                          min: 32,      max: 1024, step: 32,      group: "Network-Specific"},
  { key: "REALISM_LEVEL",              label: "Realism Level (Game)",      description: "Physics + visual realism depth for GameNet [1–5]",         min: 1,       max: 5,    step: 1,       group: "Network-Specific"},
];

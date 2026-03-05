/**
 * GameNet — Ultra-Realistic Game AI Network
 *
 * Handles: NPC behavior, physics simulation assistance,
 *          procedural content generation, difficulty adaptation,
 *          real-time strategy, pathfinding, player modeling.
 *
 * Realism levels 1–5 control physics + visual AI depth.
 * Encoding: game state vector (position, velocity, health, inventory, etc.)
 */

import { Omega, OmegaTruth } from "../omega/Omega";
import { BaseNetwork, NetworkConfig } from "./BaseNetwork";
import { TRAINING_PARAMS } from "../training/parameters";

export interface GameState {
  // Agent/NPC state
  position:      [number, number, number];
  velocity:      [number, number, number];
  rotation:      [number, number, number];
  health:        number;   // [0, 1]
  stamina:       number;   // [0, 1]

  // Environment
  nearbyEntities: number[][];  // Each entity: [x, y, z, type, health]
  terrainData:    number[];    // Local terrain heightmap (flattened)
  lightLevel:     number;      // [0, 1]
  timeOfDay:      number;      // [0, 1] 0=midnight, 0.5=noon

  // Game logic
  objectiveVector: number[];   // Encoded goal
  inventory:       number[];   // Item counts (normalized)
  threatLevel:     number;     // [0, 1]

  // Meta
  playerSkillEstimate: number; // [0, 1]
  gameMode: "survival" | "combat" | "exploration" | "creative";
  realistLevel?: number;
}

export interface GameAction {
  type: "move" | "attack" | "interact" | "use-item" | "wait" | "spawn-event" | "generate-terrain";
  direction?:    [number, number, number];
  targetEntity?: number;
  itemId?:       number;
  confidence:    number;
  raw:           OmegaTruth;
}

export class GameNet extends BaseNetwork {
  private readonly vecDim      = TRAINING_PARAMS.NEURON_DENSITY;
  private readonly realism     = TRAINING_PARAMS.REALISM_LEVEL;
  private readonly spatialRes  = TRAINING_PARAMS.SPATIAL_RESOLUTION;

  constructor(omega: Omega) {
    const config: NetworkConfig = {
      id:           "gamenet",
      type:         "game",
      clusterCount: 5,
      coreCount:    TRAINING_PARAMS.LI_CORE_COUNT,
    };
    super(config, omega);
  }

  async encode(input: unknown): Promise<number[]> {
    const gs  = input as GameState;
    const vec = new Array<number>(this.vecDim).fill(0);

    if (!gs) return vec;

    // Positional encoding
    const [px, py, pz] = gs.position ?? [0, 0, 0];
    const [vx, vy, vz] = gs.velocity ?? [0, 0, 0];
    const [rx, ry, rz] = gs.rotation ?? [0, 0, 0];

    vec[0] = Math.tanh(px / 100);
    vec[1] = Math.tanh(py / 100);
    vec[2] = Math.tanh(pz / 100);
    vec[3] = Math.tanh(vx / 10);
    vec[4] = Math.tanh(vy / 10);
    vec[5] = Math.tanh(vz / 10);
    vec[6] = rx / (2 * Math.PI);
    vec[7] = ry / (2 * Math.PI);
    vec[8] = rz / (2 * Math.PI);

    // Vital stats
    vec[9]  = gs.health   ?? 1;
    vec[10] = gs.stamina  ?? 1;
    vec[11] = gs.lightLevel  ?? 0.5;
    vec[12] = gs.timeOfDay   ?? 0.5;
    vec[13] = gs.threatLevel ?? 0;
    vec[14] = gs.playerSkillEstimate ?? 0.5;

    // Nearby entities (up to 8)
    const entities = (gs.nearbyEntities ?? []).slice(0, 8);
    for (let i = 0; i < entities.length; i++) {
      const base = 15 + i * 5;
      for (let j = 0; j < 5 && base + j < this.vecDim; j++) {
        vec[base + j] = Math.tanh((entities[i][j] ?? 0) / 100);
      }
    }

    // Terrain (realism-scaled)
    const terrainSlice = (gs.terrainData ?? []).slice(0, this.spatialRes * 8);
    for (let i = 0; i < terrainSlice.length; i++) {
      const idx = this.vecDim - terrainSlice.length + i;
      if (idx >= 0) vec[idx] = Math.tanh(terrainSlice[i] / 255);
    }

    // Objective + inventory
    const obj = (gs.objectiveVector ?? []).slice(0, 8);
    for (let i = 0; i < obj.length; i++) {
      const idx = Math.floor(this.vecDim * 0.7) + i;
      if (idx < this.vecDim) vec[idx] = Math.tanh(obj[i]);
    }

    // Realism scaling — higher realism = more physics detail injected
    const realismFactor = (gs.realistLevel ?? this.realism) / 5;
    for (let i = 0; i < this.vecDim; i++) {
      vec[i] *= (0.6 + realismFactor * 0.4);
    }

    return vec;
  }

  async decode(truth: OmegaTruth): Promise<GameAction> {
    const v = truth.answerVector;
    const actionTypes: GameAction["type"][] = ["move", "attack", "interact", "use-item", "wait", "spawn-event", "generate-terrain"];

    let bestIdx   = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < actionTypes.length; i++) {
      const score = v[i] ?? 0;
      if (score > bestScore) { bestScore = score; bestIdx = i; }
    }

    return {
      type:      actionTypes[bestIdx],
      direction: [v[7] ?? 0, v[8] ?? 0, v[9] ?? 0],
      confidence: truth.confidence,
      raw:       truth,
    };
  }

  /** Decide the best NPC action for a given game state */
  async decideAction(state: GameState): Promise<GameAction> {
    const truth = await this.infer(state, { gameMode: state.gameMode });
    return this.decode(truth);
  }

  /** Generate procedural terrain parameters */
  async generateTerrain(seed: number, biome: string): Promise<OmegaTruth> {
    return this.infer(
      { objectiveVector: [seed / 1e9], gameMode: "creative", terrainData: [], position: [0,0,0], velocity: [0,0,0], rotation: [0,0,0], health: 1, stamina: 1, nearbyEntities: [], lightLevel: 1, timeOfDay: 0.5, inventory: [], threatLevel: 0, playerSkillEstimate: 0.5 } as GameState,
      { task: "generate-terrain", biome },
    );
  }
}

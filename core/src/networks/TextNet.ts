/**
 * TextNet — Language & Reasoning Network
 *
 * Handles: text generation, question answering, summarization,
 *          translation, code, reasoning, conversation.
 *
 * Encoding: sentence-transformers via Encoder Service (896-dim).
 * Fallback: n-gram hash when encoder service unavailable.
 */

import { Omega, OmegaTruth } from "../omega/Omega";
import { BaseNetwork, NetworkConfig } from "./BaseNetwork";
import { TRAINING_PARAMS } from "../training/parameters";
import { encodeText } from "../encoders/EncoderClient";

export class TextNet extends BaseNetwork {
  private readonly vocabSize = 65536;   // 16-bit token space
  private readonly vecDim    = TRAINING_PARAMS.NEURON_DENSITY;

  constructor(omega: Omega) {
    const config: NetworkConfig = {
      id:           "textnet",
      type:         "text",
      clusterCount: 3,
      coreCount:    TRAINING_PARAMS.LI_CORE_COUNT,
    };
    super(config, omega);
  }

  /** Encode text → semantic vector via Encoder Service (fallback: n-gram) */
  async encode(input: unknown): Promise<number[]> {
    const text = String(input).slice(0, TRAINING_PARAMS.CONTEXT_WINDOW);
    return encodeText(text);
  }

  async decode(truth: OmegaTruth): Promise<{ text: string; confidence: number }> {
    return {
      text:       truth.answer,
      confidence: truth.confidence,
    };
  }

  async chat(userMessage: string, history: string[] = []): Promise<OmegaTruth> {
    const context = history.slice(-10).join("\n");
    return this.infer(`${context}\nUser: ${userMessage}\nAI:`, { history });
  }

  async summarize(text: string): Promise<OmegaTruth> {
    return this.infer(`Summarize: ${text}`, { task: "summarize" });
  }
}

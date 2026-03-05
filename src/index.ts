/**
 * OMEGA AI Engine — Public API
 */

export { MirrorAI, mirrorAI } from "./MirrorAI";
export { Omega }               from "./omega/Omega";
export { MirrorNeuron, NeuronLayer } from "./neurons/gX";
export { ProcessingCenter, LiCluster } from "./centers/Li";
export { TextNet }  from "./networks/TextNet";
export { ImageNet } from "./networks/ImageNet";
export { VideoNet } from "./networks/VideoNet";
export { AudioNet } from "./networks/AudioNet";
export { GameNet }  from "./networks/GameNet";
export { TRAINING_PARAMS, PARAM_META } from "./training/parameters";

export type { Bit, NeuronState, LayerResult }  from "./neurons/gX";
export type { NetworkType, LiStatus }           from "./centers/Li";
export type { OmegaInput, OmegaTruth }          from "./omega/Omega";
export type { TrainingParameters, ParamMeta }   from "./training/parameters";
export type { MirrorAIStatus }                  from "./MirrorAI";

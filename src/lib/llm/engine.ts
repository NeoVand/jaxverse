/**
 * The transformer engine seam — ported from LLMVibes, whose M0 spike proved
 * the contract:
 * - Training runs in a Web Worker on jax-js/WebGPU (measured FASTER than the
 *   main thread there, with the UI staying responsive).
 * - Everything must run jitted (eager dot-backward materializes [M,K,N] and
 *   dies at real vocab sizes — jax-js issue #150).
 * - Embeddings are oneHot-matmul until gather backward lands upstream (#79).
 * - Sync per step (loss readout) — pipelining is currently slower (#151).
 * - No-WebGPU browsers get prose fallbacks, not WASM training.
 *
 * jaxverse additions: `settokens` (swap the corpus in place — this is what
 * fine-tuning is) and `rlstep` (a real REINFORCE/GRPO-style policy-gradient
 * update from verifier rewards — RLVR, not a simulation).
 */

export interface ModelConfig {
	/** Display name for the run ('scribe', 'rook', …) — no behavior. */
	name?: string;
	nLayer: number;
	nEmbd: number;
	nHead: number;
	blockSize: number;
	vocab: number;
}

export interface TrainStepMetrics {
	step: number;
	loss: number;
	/** Wall-clock ms for this step (jit-compile time excluded after step 0). */
	stepMs: number;
	tokensPerSec: number;
	/** Optional per-run extras (legal-move rate, constraint pass rate, …). */
	extra?: Record<string, number>;
}

export interface SampleResult {
	/** Token ids of the completion (prompt excluded). */
	tokens: number[];
	text: string;
	/** Per-token data for the token-stream inspector. */
	perToken?: PerTokenInfo[];
}

/** One token's worth of inspector data — the universal token-stream inspector
 * (per-token loss/entropy heatmaps, chs. 2→13) is built entirely on this. */
export interface PerTokenInfo {
	id: number;
	text: string;
	/** Negative log-likelihood under the current model, when computed. */
	loss?: number;
	/** Distribution entropy at this position, when computed. */
	entropy?: number;
	/** Top-k alternatives (id, probability), when computed. */
	topk?: Array<[number, number]>;
}

export type EngineCapability = 'webgpu' | 'playback-only';

/**
 * The async world-engine the playgrounds drive. One instance per mounted
 * playground; heavy state lives GPU-side in the Worker.
 */
export interface LlmEngine {
	readonly capability: EngineCapability;

	/** Create or load a model; resolves when weights are resident. */
	init(config: ModelConfig, checkpoint?: ArrayBuffer): Promise<void>;

	/** Run n training steps; metrics stream via onMetrics. */
	train(steps: number, onMetrics: (m: TrainStepMetrics) => void): Promise<void>;

	/** Pause/abort an in-flight train() (resolves its promise early). */
	stop(): Promise<void>;

	/** Sample a completion from the current weights. */
	sample(
		promptTokens: number[],
		opts?: { temperature?: number; topK?: number; maxTokens?: number }
	): Promise<SampleResult>;

	/** Per-token loss/entropy for given text — feeds the inspector heatmap. */
	inspect(tokens: number[]): Promise<PerTokenInfo[]>;

	/** Serialize current weights (for OPFS checkpoints / stat sheets). */
	exportCheckpoint(): Promise<ArrayBuffer>;

	dispose(): Promise<void>;
}

/** Feature-detect what this browser can offer, choosing the engine tier. */
export async function detectCapability(): Promise<EngineCapability> {
	if (typeof navigator !== 'undefined' && navigator.gpu) {
		try {
			const adapter = await navigator.gpu.requestAdapter();
			if (adapter) return 'webgpu';
		} catch {
			// fall through
		}
	}
	return 'playback-only';
}

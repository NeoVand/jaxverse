/**
 * The MLP engine seam — one small, general trainer serves chapters 1–4:
 * curve fitting (mse), 2-D classification (xent), MNIST (xent), and the
 * autoencoder (mse against the input).
 *
 * Contract carried over from LLMVibes' proven jax-js idioms:
 * - Training runs in a Web Worker (faster there, UI stays at 60fps).
 * - Everything is jitted; jit signatures are cached per input shape, so all
 *   variable-size calls are padded to fixed chunk sizes.
 * - The engine prefers WebGPU but will fall back to jax-js's cpu/wasm
 *   backends: the early-chapter models are small enough to train anywhere.
 */

export type Activation = 'tanh' | 'relu' | 'gelu' | 'silu';
export type LossKind = 'mse' | 'xent';

export interface MlpConfig {
	/** Layer widths, input first, output last — e.g. [2, 8, 8, 2]. */
	layers: number[];
	activation: Activation;
	loss: LossKind;
	seed?: number;
	lr?: number;
	batchSize?: number;
	/** Fraction of rows held out for validation (taken from the tail). */
	valFraction?: number;
}

export interface DataSpec {
	/** Row-major features, n × inDim. */
	x: Float32Array;
	/** Float targets (n × outDim) for mse, or int labels (n) for xent. */
	y: Float32Array | Int32Array;
	n: number;
}

export interface TrainMetrics {
	step: number;
	loss: number;
	stepMs: number;
}

export interface EvalResult {
	loss: number;
	/** Classification only. */
	accuracy?: number;
}

export interface LayerWeights {
	/** Row-major inDim × outDim. */
	w: Float32Array;
	b: Float32Array;
	inDim: number;
	outDim: number;
}

/** Which device the worker actually got. */
export type DeviceKind = 'webgpu' | 'cpu';

export async function detectWebGPU(): Promise<boolean> {
	if (typeof navigator === 'undefined' || !navigator.gpu) return false;
	try {
		return (await navigator.gpu.requestAdapter()) !== null;
	} catch {
		return false;
	}
}

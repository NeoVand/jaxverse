// The async seam between the plates and the GPU.
//
// Same shape as the language engine in $lib/llm: one Worker per lab, a promise
// per request, metrics streamed on a side channel. Diffusion needs one thing
// the language model did not — a sampler that reports every intermediate step,
// because the intermediate steps are what two of these plates are about.

import type { Condition } from './runtime';
import type { Objective } from './model';

export interface DiffusionInit {
	objective: Objective;
	/** Start from the shipped checkpoint instead of random weights. */
	checkpoint?: ArrayBuffer;
	batch?: number;
	lr?: number;
	seed?: number;
}

export interface TrainStepMetrics {
	step: number;
	loss: number;
	/** Wall-clock ms for this step, jit compilation excluded after step 0. */
	stepMs: number;
	imagesPerSec: number;
}

export interface SampleOptions {
	steps: number;
	a: Condition;
	b?: Condition;
	guidanceA: number;
	guidanceB?: number;
	seed: number;
	/** 0 deterministic, 1 ancestral. Only meaningful for the `eps` model. */
	eta?: number;
	/** Ask for every intermediate frame; used by the trajectory plates. */
	trace?: 'state' | 'endpoint';
	/** Draw from the shipped checkpoint rather than the live weights. */
	fromShipped?: boolean;
}

export interface SampleResult {
	/** [count · channels · res · res] in roughly [-1, 1]. */
	pixels: Float32Array;
	count: number;
	/** Present when `trace` was set: one entry per denoising step. */
	frames?: Float32Array[];
	ms: number;
}

export type Phase = 'idle' | 'loading' | 'ready' | 'training' | 'sampling' | 'error' | 'no-webgpu';

/** True when this browser can run the chapters' models at all. */
export async function detectWebGPU(): Promise<boolean> {
	if (typeof navigator === 'undefined' || !navigator.gpu) return false;
	try {
		const adapter = await Promise.race([
			navigator.gpu.requestAdapter(),
			new Promise<null>((r) => setTimeout(() => r(null), 8000))
		]);
		return adapter !== null;
	} catch {
		return false;
	}
}

// The diffusion worker: owns the GPU device, the weights and the corpus.
//
// Training and sampling both belong off the main thread — a denoising step is
// a full forward pass over three guidance branches, and doing that between
// animation frames would drop every one of them.

/// <reference lib="webworker" />

import { init, defaultDevice, tree, blockUntilReady } from '@jax-js/jax';
import { EMOJI_SHAPE, initParams, type DiffusionConfig } from './model';
import { loadEmoji, type EmojiCorpus } from './corpus';
import {
	allocBatch,
	makeBatch,
	makeOptimizer,
	makeSampler,
	mulberry32,
	NOTHING,
	type Condition,
	type Optimizer,
	type Sampler,
	type TrainBatch
} from './runtime';
import { unpack } from './checkpoint';
import type { TrainStepMetrics } from './engine';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Arr = any;

interface RpcRequest {
	id: number;
	op: string;
	[key: string]: unknown;
}

let cfg: DiffusionConfig | null = null;
let corpus: EmojiCorpus | null = null;
let params: Arr = null;
/** The shipped checkpoint, kept alongside so plates can compare the two. */
let shipped: Arr = null;
let opt: Optimizer | null = null;
let batchBuf: TrainBatch | null = null;
let spareBuf: TrainBatch | null = null;
let rand: () => number = mulberry32(1);
let stepCounter = 0;
let stopRequested = false;
let lr = 3e-4;
let batchSize = 32;
/** Samplers are keyed by picture count: jit compiles per shape. */
const samplers = new Map<number, Sampler>();

function post(msg: unknown, transfer: Transferable[] = []) {
	(self as unknown as Worker).postMessage(msg, transfer);
}

function samplerFor(count: number): Sampler {
	let s = samplers.get(count);
	if (!s) {
		s = makeSampler(cfg!, count);
		samplers.set(count, s);
	}
	return s;
}

async function handleInit(req: RpcRequest) {
	const devices = await init();
	if (!devices.includes('webgpu')) throw new Error('WebGPU unavailable in worker');
	defaultDevice('webgpu');

	corpus = await loadEmoji((req.base as string) ?? '');
	cfg = {
		...EMOJI_SHAPE,
		tags: corpus.meta.tags.length,
		styles: corpus.meta.sets.length,
		objective: (req.objective as DiffusionConfig['objective']) ?? 'flow'
	};
	batchSize = (req.batch as number) ?? 32;
	lr = (req.lr as number) ?? 3e-4;
	rand = mulberry32((req.seed as number) ?? 7);

	params = initParams(cfg, (req.seed as number) ?? 20260905);
	await blockUntilReady(params);
	opt = makeOptimizer(cfg, batchSize, params);
	batchBuf = allocBatch(cfg, batchSize);
	spareBuf = allocBatch(cfg, batchSize);
	makeBatch(corpus, cfg, batchSize, rand, batchBuf);
	stepCounter = 0;

	if (req.checkpoint) {
		const loaded = unpack(req.checkpoint as ArrayBuffer, initParams(cfg, 0));
		shipped = loaded.params;
		await blockUntilReady(shipped);
	}

	return {
		count: corpus.count,
		styles: corpus.styles,
		tags: corpus.meta.tags,
		sets: corpus.meta.sets,
		emoji: corpus.meta.emoji.map((e) => ({ cp: e.cp, name: e.name })),
		hasShipped: shipped !== null
	};
}

/**
 * Train for `steps`, streaming one metric per step.
 *
 * The next batch is built while the current one is still on the GPU, which is
 * not a micro-optimization: corrupting sixty-four pictures is a few hundred
 * thousand transcendental functions, and unoverlapped it was a third of the
 * wall clock.
 */
async function handleTrain(req: RpcRequest) {
	if (!cfg || !opt || !corpus) throw new Error('train before init');
	const steps = (req.steps as number) ?? 50;
	stopRequested = false;
	let done = 0;
	for (let i = 0; i < steps; i++) {
		if (stopRequested) break;
		const t0 = performance.now();
		const [lossArr, next] = opt.step(params, batchBuf!, lr);
		params = next;
		makeBatch(corpus, cfg, batchSize, rand, spareBuf!);
		[batchBuf, spareBuf] = [spareBuf!, batchBuf!];
		const loss = lossArr.item();
		const stepMs = performance.now() - t0;
		stepCounter++;
		done++;
		const m: TrainStepMetrics = {
			step: stepCounter,
			loss,
			stepMs,
			imagesPerSec: Math.round((batchSize * 1000) / stepMs)
		};
		post({ id: req.id, event: 'metrics', m });
		// yield so a 'stop' message can get through
		if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0));
	}
	return { completed: done, step: stepCounter };
}

async function handleSample(req: RpcRequest) {
	if (!cfg) throw new Error('sample before init');
	const count = (req.count as number) ?? 8;
	const source = req.fromShipped ? shipped : params;
	if (!source) throw new Error('no weights to sample from');

	const frames: ArrayBuffer[] = [];
	const t0 = performance.now();
	const pixels = await samplerFor(count).run({
		params: tree.ref(source),
		steps: (req.steps as number) ?? 24,
		a: (req.a as Condition) ?? NOTHING,
		b: req.b as Condition | undefined,
		guidanceA: (req.guidanceA as number) ?? 1,
		guidanceB: req.guidanceB as number | undefined,
		seed: (req.seed as number) ?? 1,
		eta: req.eta as number | undefined,
		trace: req.trace as 'state' | 'endpoint' | undefined,
		onStep: req.trace
			? (frame) => {
					frames.push(frame.slice().buffer as ArrayBuffer);
				}
			: undefined
	});
	const out = pixels.slice();
	return {
		pixels: out.buffer,
		count,
		frames: req.trace ? frames : undefined,
		ms: performance.now() - t0,
		__transfer: [out.buffer as ArrayBuffer, ...frames]
	};
}

/** Reference tiles straight from the corpus, for the plates that show data. */
function handleTiles(req: RpcRequest) {
	if (!corpus || !cfg) throw new Error('tiles before init');
	const indices = req.indices as number[];
	const style = (req.style as number) ?? 0;
	const dim = cfg.channels * cfg.res * cfg.res;
	const out = new Float32Array(indices.length * dim);
	for (let k = 0; k < indices.length; k++) {
		const src = (style * corpus.count + indices[k]) * dim;
		for (let i = 0; i < dim; i++) out[k * dim + i] = corpus.images[src + i] / 127.5 - 1;
	}
	return { pixels: out.buffer, count: indices.length, __transfer: [out.buffer as ArrayBuffer] };
}

function handleReset(req: RpcRequest) {
	if (!cfg || !opt) throw new Error('reset before init');
	tree.dispose(params);
	opt.dispose();
	params = initParams(cfg, (req.seed as number) ?? 20260905);
	opt = makeOptimizer(cfg, batchSize, params);
	stepCounter = 0;
	return { step: 0 };
}

const handlers: Record<string, (req: RpcRequest) => unknown | Promise<unknown>> = {
	init: handleInit,
	train: handleTrain,
	stop: () => {
		stopRequested = true;
		return {};
	},
	sample: handleSample,
	tiles: handleTiles,
	reset: handleReset,
	setlr: (req) => {
		lr = req.lr as number;
		return { lr };
	},
	dispose: () => {
		if (params) tree.dispose(params);
		if (shipped) tree.dispose(shipped);
		opt?.dispose();
		params = null;
		shipped = null;
		opt = null;
		return {};
	}
};

self.onmessage = async (e: MessageEvent) => {
	const req = e.data as RpcRequest;
	const handler = handlers[req.op];
	if (!handler) {
		post({ id: req.id, ok: false, error: `unknown op: ${req.op}` });
		return;
	}
	try {
		const result = (await handler(req)) as Record<string, unknown> | undefined;
		const transfer = (result?.__transfer as Transferable[]) ?? [];
		if (result) delete result.__transfer;
		post({ id: req.id, ok: true, result }, transfer);
	} catch (err) {
		post({ id: req.id, ok: false, error: err instanceof Error ? err.message : String(err) });
	}
};

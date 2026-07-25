// The MLP training worker: owns jax-js, the params, optimizer state and the
// dataset. Driven over a tiny promise RPC (see mlp-engine.ts). Idioms follow
// LLMVibes' battle-tested worker: everything jitted, batch tensors passed as
// .ref then disposed, params tree.ref'd at every jit boundary.

import { init, defaultDevice, numpy as np, nn, jit, valueAndGrad, tree, grad } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';
import type { Activation, MlpConfig } from './engine';

/* eslint-disable @typescript-eslint/no-explicit-any */

interface RpcRequest {
	id: number;
	op: string;
	[key: string]: unknown;
}

const post = (msg: unknown, transfer?: Transferable[]) =>
	(self as unknown as Worker).postMessage(msg, { transfer: transfer ?? [] });

// ── worker state ─────────────────────────────────────────────────────────────
let cfg: MlpConfig | null = null;
let params: any = null;
let optState: any = null;
let solver: ReturnType<typeof adam> | null = null;
let dataX: Float32Array | null = null;
let dataY: Float32Array | Int32Array | null = null;
let nRows = 0;
let valStart = 0;
let stepCounter = 0;
let stopRequested = false;
let device = 'none';

let jitStep: any = null;
let jitLossOnly: any = null;
// jit caches keyed by padded chunk size (jit compiles per shape)
const jitForwardBy = new Map<number, any>();
const jitActsBy = new Map<number, any>();
const jitFromBy = new Map<string, any>();
const jitInputGradBy = new Map<number, any>();

function mulberry32(seed: number) {
	return function () {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
let rng = mulberry32(1234);

// ── model ────────────────────────────────────────────────────────────────────

const ACT: Record<Activation, (x: any) => any> = {
	tanh: (x) => np.tanh(x),
	relu: (x) => nn.relu(x)
};

function initParams(c: MlpConfig, seed: number): any {
	const L = c.layers.length - 1;
	const w: any[] = [];
	const b: any[] = [];
	let s = seed >>> 0 || 1;
	const rand = mulberry32(s * 2654435761);
	for (let k = 0; k < L; k++) {
		const fin = c.layers[k];
		const fout = c.layers[k + 1];
		// Glorot for tanh, He for relu; final layer damped so the net starts humble.
		const limit = c.activation === 'relu' ? Math.sqrt(6 / fin) : Math.sqrt(6 / (fin + fout));
		const scale = k === L - 1 ? 0.4 : 1;
		const buf = new Float32Array(fin * fout);
		for (let i = 0; i < buf.length; i++) buf[i] = (rand() * 2 - 1) * limit * scale;
		w.push(np.array(buf).reshape([fin, fout]));
		b.push(np.zeros([fout]));
		s ^= 0x9e3779b9;
	}
	return { w, b };
}

/** Forward pass. Consumes x; every params leaf consumed exactly once. */
function forward(p: any, c: MlpConfig, x: any): any {
	const L = p.w.length;
	let h = x;
	for (let k = 0; k < L; k++) {
		h = np.dot(h, p.w[k]).add(p.b[k]);
		if (k < L - 1) h = ACT[c.activation](h);
	}
	return h;
}

/** Forward that also returns every post-activation hidden layer (+ output).
 * Intermediate layers are pushed as .ref (the working value flows onward);
 * the last layer is pushed directly — nothing is disposed inside the trace. */
function forwardActs(p: any, c: MlpConfig, x: any): any[] {
	const L = p.w.length;
	const outs: any[] = [];
	let h = x;
	for (let k = 0; k < L; k++) {
		h = np.dot(h, p.w[k]).add(p.b[k]);
		if (k < L - 1) h = ACT[c.activation](h);
		outs.push(k === L - 1 ? h : h.ref);
	}
	return outs;
}

/** Run the network from layer `from` onward, given that layer's activations. */
function forwardFrom(p: any, c: MlpConfig, from: number, h0: any): any {
	const L = p.w.length;
	let h = h0;
	for (let k = from; k < L; k++) {
		h = np.dot(h, p.w[k]).add(p.b[k]);
		if (k < L - 1) h = ACT[c.activation](h);
	}
	return h;
}

/** Training loss. Consumes x and y (y is targets [B,out] or one-hot [B,C]). */
function lossFn(p: any, c: MlpConfig, x: any, y: any): any {
	const pred = forward(p, c, x);
	if (c.loss === 'mse') {
		return np.mean(np.square(pred.sub(y)));
	}
	const logp = nn.logSoftmax(pred, -1);
	return np.mean(np.sum(logp.mul(y), -1).neg());
}

function disposeTree(t: any): void {
	for (const l of tree.leaves(t)) l.dispose();
}

function flattenParams(p: any): Float32Array {
	const leaves = tree.leaves(tree.ref(p)) as any[];
	const total = leaves.reduce((s: number, l: any) => s + l.size, 0);
	const out = new Float32Array(total);
	let offset = 0;
	for (const leaf of leaves) {
		out.set(leaf.dataSync(), offset);
		offset += leaf.size;
	}
	return out;
}

function loadParamsFlat(c: MlpConfig, flat: Float32Array): any {
	const template = initParams(c, 0);
	let offset = 0;
	return tree.map((leaf: any) => {
		const chunk = flat.slice(offset, offset + leaf.size);
		offset += leaf.size;
		const next = np.array(chunk).reshape(leaf.shape);
		leaf.dispose();
		return next;
	}, template);
}

// ── batches ──────────────────────────────────────────────────────────────────

const outDim = () => cfg!.layers[cfg!.layers.length - 1];
const inDim = () => cfg!.layers[0];

/** Assemble a [B, din] x tensor and its target tensor from sampled rows. */
function makeBatch(B: number, lo: number, hi: number, rand: () => number) {
	const c = cfg!;
	const din = inDim();
	const dout = outDim();
	const xBuf = new Float32Array(B * din);
	const span = Math.max(1, hi - lo);
	const idx: number[] = [];
	for (let b = 0; b < B; b++) idx.push(lo + Math.floor(rand() * span));
	for (let b = 0; b < B; b++) xBuf.set(dataX!.subarray(idx[b] * din, (idx[b] + 1) * din), b * din);
	const x = np.array(xBuf).reshape([B, din]);

	if (c.loss === 'xent') {
		const lBuf = new Int32Array(B);
		for (let b = 0; b < B; b++) lBuf[b] = (dataY as Int32Array)[idx[b]];
		const labels = np.array(lBuf, { dtype: np.int32 });
		return { x, y: nn.oneHot(labels, dout) };
	}
	// mse — autoencoders pass y === x by construction (dataY views the same rows)
	const yBuf = new Float32Array(B * dout);
	for (let b = 0; b < B; b++)
		yBuf.set((dataY as Float32Array).subarray(idx[b] * dout, (idx[b] + 1) * dout), b * dout);
	return { x, y: np.array(yBuf).reshape([B, dout]) };
}

/** One optimizer step. `sync` reads the loss (a GPU roundtrip); skipping it
 * on most steps lets tiny models train dispatch-bound instead of sync-bound —
 * the lazy graph resolves at the next synced step. */
function trainStep(sync: boolean): number {
	const B = cfg!.batchSize ?? 32;
	const { x, y } = makeBatch(B, 0, valStart, rng);
	const [lossVal, grads] = jitStep(tree.ref(params), x.ref, y.ref);
	const [updates, newOptState] = solver!.update(grads, optState, tree.ref(params));
	params = applyUpdates(params, updates);
	optState = newOptState;
	x.dispose();
	y.dispose();
	if (!sync) {
		lossVal.dispose();
		return NaN;
	}
	return lossVal.item();
}

// ── padded fixed-shape helpers (one jit signature per chunk size) ────────────

function getJitForward(chunk: number) {
	let f = jitForwardBy.get(chunk);
	if (!f) {
		const c = cfg!;
		f = jit((p: any, x: any) => forward(p, c, x));
		jitForwardBy.set(chunk, f);
	}
	return f;
}

function getJitActs(chunk: number) {
	let f = jitActsBy.get(chunk);
	if (!f) {
		const c = cfg!;
		f = jit((p: any, x: any) => forwardActs(p, c, x));
		jitActsBy.set(chunk, f);
	}
	return f;
}

function getJitFrom(from: number, chunk: number) {
	const key = `${from}:${chunk}`;
	let f = jitFromBy.get(key);
	if (!f) {
		const c = cfg!;
		f = jit((p: any, h: any) => forwardFrom(p, c, from, h));
		jitFromBy.set(key, f);
	}
	return f;
}

function getJitInputGrad(chunk: number) {
	let f = jitInputGradBy.get(chunk);
	if (!f) {
		const c = cfg!;
		// d/dx of the target-class score Σ logits·target — per-pixel evidence.
		f = jit((x: any, p: any, t: any) =>
			grad((xx: any, pp: any, tt: any) => np.sum(forward(pp, c, xx).mul(tt)))(x, p, t)
		);
		jitInputGradBy.set(chunk, f);
	}
	return f;
}

/** Predict on n rows (arbitrary n) by padding to fixed-size chunks. */
function predictAll(x: Float32Array, n: number, chunk: number): Float32Array {
	const din = inDim();
	const dout = outDim();
	const out = new Float32Array(n * dout);
	const f = getJitForward(chunk);
	for (let start = 0; start < n; start += chunk) {
		const rows = Math.min(chunk, n - start);
		const buf = new Float32Array(chunk * din);
		buf.set(x.subarray(start * din, (start + rows) * din));
		const xa = np.array(buf).reshape([chunk, din]);
		const pred = f(tree.ref(params), xa);
		const data = pred.dataSync() as Float32Array;
		out.set(data.subarray(0, rows * dout), start * dout);
	}
	return out;
}

// ── op handlers ──────────────────────────────────────────────────────────────

async function handleInit(req: RpcRequest) {
	const devices = await init();
	device = devices.includes('webgpu') ? 'webgpu' : devices.includes('wasm') ? 'wasm' : 'cpu';
	defaultDevice(device as 'webgpu' | 'wasm' | 'cpu');

	cfg = req.config as MlpConfig;
	dataX = new Float32Array(req.x as ArrayBuffer);
	dataY =
		cfg.loss === 'xent'
			? new Int32Array(req.y as ArrayBuffer)
			: new Float32Array(req.y as ArrayBuffer);
	nRows = req.n as number;
	rng = mulberry32(cfg.seed ?? 1234);
	stepCounter = 0;

	if (params) disposeTree(params);
	if (optState) disposeTree(optState);
	jitForwardBy.clear();
	jitActsBy.clear();
	jitFromBy.clear();
	jitInputGradBy.clear();

	params = req.checkpoint
		? loadParamsFlat(cfg, new Float32Array(req.checkpoint as ArrayBuffer))
		: initParams(cfg, cfg.seed ?? 42);
	solver = adam(cfg.lr ?? 3e-3, { b1: 0.9, b2: 0.99 });
	optState = solver.init(tree.ref(params));

	const c = cfg;
	jitStep = jit((p: any, x: any, y: any) => valueAndGrad((pp: any) => lossFn(pp, c, x, y))(p));
	jitLossOnly = jit((p: any, x: any, y: any) => lossFn(p, c, x, y));

	const vf = cfg.valFraction ?? 0.1;
	valStart = Math.max(1, Math.floor(nRows * (1 - vf)));

	const nParams = (tree.leaves(tree.ref(params)) as any[]).reduce((s, l) => {
		const size = l.size;
		l.dispose();
		return s + size;
	}, 0);
	return { device, paramCount: nParams };
}

/** Replace the dataset in place (same dims) — params and jits survive. */
function handleSetData(req: RpcRequest) {
	dataX = new Float32Array(req.x as ArrayBuffer);
	dataY =
		cfg!.loss === 'xent'
			? new Int32Array(req.y as ArrayBuffer)
			: new Float32Array(req.y as ArrayBuffer);
	nRows = req.n as number;
	const vf = cfg!.valFraction ?? 0.1;
	valStart = Math.max(1, Math.floor(nRows * (1 - vf)));
	return { n: nRows };
}

/** Re-roll the weights (fresh seed) without touching data or jit caches. */
function handleReset(req: RpcRequest) {
	if (params) disposeTree(params);
	if (optState) disposeTree(optState);
	params = initParams(cfg!, (req.seed as number) ?? Math.floor(rng() * 1e9));
	optState = solver!.init(tree.ref(params));
	stepCounter = 0;
	return {};
}

async function handleTrain(req: RpcRequest) {
	const steps = (req.steps as number) ?? 50;
	const syncEvery = Math.max(1, (req.syncEvery as number) ?? 1);
	stopRequested = false;
	let done = 0;
	let t0 = performance.now();
	let sinceSync = 0;
	for (let i = 0; i < steps; i++) {
		if (stopRequested) break;
		const sync = (i + 1) % syncEvery === 0 || i === steps - 1;
		const loss = trainStep(sync);
		stepCounter++;
		done++;
		sinceSync++;
		if (sync) {
			const stepMs = (performance.now() - t0) / sinceSync;
			post({ id: req.id, event: 'metrics', m: { step: stepCounter, loss, stepMs } });
			t0 = performance.now();
			sinceSync = 0;
			// Yield so 'stop' and other messages get through mid-run.
			await new Promise((r) => setTimeout(r, 0));
		}
	}
	return { completed: done, step: stepCounter };
}

function handlePredict(req: RpcRequest) {
	const n = req.n as number;
	const x = new Float32Array(req.x as ArrayBuffer);
	const chunk = (req.chunk as number) ?? 256;
	const out = predictAll(x, n, chunk);
	return { y: out.buffer, __transfer: [out.buffer] };
}

/** Post-activation values of every layer for the given rows. */
function handleActivations(req: RpcRequest) {
	const n = req.n as number;
	const x = new Float32Array(req.x as ArrayBuffer);
	const chunk = (req.chunk as number) ?? 1024;
	const c = cfg!;
	const din = inDim();
	const L = c.layers.length - 1;
	const layerBufs: Float32Array[] = [];
	for (let k = 0; k < L; k++) layerBufs.push(new Float32Array(n * c.layers[k + 1]));
	const f = getJitActs(chunk);
	for (let start = 0; start < n; start += chunk) {
		const rows = Math.min(chunk, n - start);
		const buf = new Float32Array(chunk * din);
		buf.set(x.subarray(start * din, (start + rows) * din));
		const xa = np.array(buf).reshape([chunk, din]);
		const acts = f(tree.ref(params), xa) as any[];
		for (let k = 0; k < L; k++) {
			const w = c.layers[k + 1];
			// dataSync realizes and consumes the handle — no dispose afterwards
			const data = acts[k].dataSync() as Float32Array;
			layerBufs[k].set(data.subarray(0, rows * w), start * w);
		}
	}
	const buffers = layerBufs.map((b) => b.buffer as ArrayBuffer);
	return { layers: buffers, widths: c.layers.slice(1), __transfer: buffers };
}

/** Run the tail of the network from layer `from`, given activations h. */
function handleForwardFrom(req: RpcRequest) {
	const from = req.from as number;
	const n = req.n as number;
	const h = new Float32Array(req.h as ArrayBuffer);
	const c = cfg!;
	const hDim = c.layers[from];
	const dout = outDim();
	const chunk = (req.chunk as number) ?? 256;
	const out = new Float32Array(n * dout);
	const f = getJitFrom(from, chunk);
	for (let start = 0; start < n; start += chunk) {
		const rows = Math.min(chunk, n - start);
		const buf = new Float32Array(chunk * hDim);
		buf.set(h.subarray(start * hDim, (start + rows) * hDim));
		const ha = np.array(buf).reshape([chunk, hDim]);
		const pred = f(tree.ref(params), ha);
		const data = pred.dataSync() as Float32Array;
		out.set(data.subarray(0, rows * dout), start * dout);
	}
	return { y: out.buffer, __transfer: [out.buffer] };
}

/** d(score of class t)/d(input pixels) for a single row — the saliency map. */
function handleInputGrad(req: RpcRequest) {
	const x = new Float32Array(req.x as ArrayBuffer);
	const target = req.target as number;
	const din = inDim();
	const dout = outDim();
	const chunk = 16;
	const xBuf = new Float32Array(chunk * din);
	xBuf.set(x.subarray(0, din));
	const tBuf = new Float32Array(chunk * dout);
	tBuf[target] = 1; // only row 0 contributes
	const f = getJitInputGrad(chunk);
	const g = f(
		np.array(xBuf).reshape([chunk, din]),
		tree.ref(params),
		np.array(tBuf).reshape([chunk, dout])
	);
	const data = (g.dataSync() as Float32Array).slice(0, din);
	return { g: data.buffer, __transfer: [data.buffer] };
}

/** Deterministic held-out evaluation: loss (+ accuracy when classifying). */
function handleEval() {
	const c = cfg!;
	const nVal = Math.max(1, Math.min(nRows - valStart, 512));
	const din = inDim();
	const dout = outDim();
	const xv = dataX!.subarray(valStart * din, (valStart + nVal) * din);
	const pred = predictAll(new Float32Array(xv), nVal, 256);
	if (c.loss === 'mse') {
		const yv = (dataY as Float32Array).subarray(valStart * dout, (valStart + nVal) * dout);
		let s = 0;
		for (let i = 0; i < nVal * dout; i++) {
			const d = pred[i] - yv[i];
			s += d * d;
		}
		return { loss: s / (nVal * dout) };
	}
	// xent: stable log-softmax on CPU + argmax accuracy
	let total = 0;
	let correct = 0;
	for (let r = 0; r < nVal; r++) {
		const row = pred.subarray(r * dout, (r + 1) * dout);
		let mx = -Infinity;
		let arg = 0;
		for (let j = 0; j < dout; j++)
			if (row[j] > mx) {
				mx = row[j];
				arg = j;
			}
		let lse = 0;
		for (let j = 0; j < dout; j++) lse += Math.exp(row[j] - mx);
		lse = mx + Math.log(lse);
		const label = (dataY as Int32Array)[valStart + r];
		total += lse - row[label];
		if (arg === label) correct++;
	}
	return { loss: total / nVal, accuracy: correct / nVal };
}

/** Mean training-set loss on a fixed batch (comparable across calls). */
function handleTrainLoss() {
	const r = mulberry32(5555);
	const { x, y } = makeBatch(Math.min(256, valStart), 0, valStart, r);
	const lossVal = jitLossOnly(tree.ref(params), x.ref, y.ref);
	x.dispose();
	y.dispose();
	return { loss: lossVal.item() };
}

/** Per-layer weight matrices (for weight-image visualizations). */
function handleWeights() {
	const c = cfg!;
	const L = c.layers.length - 1;
	const buffers: ArrayBuffer[] = [];
	for (let k = 0; k < L; k++) {
		const wl = (params.w[k] as any).ref.dataSync() as Float32Array;
		const bl = (params.b[k] as any).ref.dataSync() as Float32Array;
		buffers.push(wl.buffer as ArrayBuffer, bl.buffer as ArrayBuffer);
	}
	return { buffers, layers: c.layers.slice(), __transfer: buffers };
}

async function handleExport() {
	const flat = flattenParams(params);
	return { checkpoint: flat.buffer, __transfer: [flat.buffer] };
}

function handleSetLr(req: RpcRequest) {
	const lr = req.lr as number;
	if (!(lr > 0)) throw new Error(`bad lr: ${lr}`);
	if (optState) disposeTree(optState);
	solver = adam(lr, { b1: 0.9, b2: 0.99 });
	optState = solver.init(tree.ref(params));
	return { lr };
}

// ── dispatch ─────────────────────────────────────────────────────────────────
const handlers: Record<string, (req: RpcRequest) => unknown | Promise<unknown>> = {
	init: handleInit,
	setdata: handleSetData,
	reset: handleReset,
	train: handleTrain,
	stop: () => {
		stopRequested = true;
		return {};
	},
	predict: handlePredict,
	activations: handleActivations,
	forwardfrom: handleForwardFrom,
	inputgrad: handleInputGrad,
	eval: handleEval,
	trainloss: handleTrainLoss,
	weights: handleWeights,
	export: handleExport,
	setlr: handleSetLr,
	dispose: () => {
		if (params) disposeTree(params);
		if (optState) disposeTree(optState);
		params = null;
		optState = null;
		return {};
	}
};

self.onmessage = async (e: MessageEvent<RpcRequest>) => {
	const req = e.data;
	try {
		const handler = handlers[req.op];
		if (!handler) throw new Error(`unknown op: ${req.op}`);
		const result = (await handler(req)) as Record<string, unknown> & {
			__transfer?: Transferable[];
		};
		const transfer = result?.__transfer;
		if (transfer) delete result.__transfer;
		post({ id: req.id, ok: true, result }, transfer);
	} catch (err) {
		post({
			id: req.id,
			ok: false,
			error: err instanceof Error ? `${err.name}: ${err.message}` : String(err)
		});
	}
};

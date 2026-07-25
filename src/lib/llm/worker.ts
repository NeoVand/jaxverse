// The training worker: owns jax-js, the model params, the optimizer state and
// the token data. The main thread drives it via a tiny RPC protocol (see
// worker-engine.ts). Training was measured FASTER in a Worker than on the main
// thread (spikes/m0/RESULTS.md), with the UI staying at 60fps.

import {
	init,
	defaultDevice,
	numpy as np,
	nn,
	jit,
	valueAndGrad,
	tree,
	blockUntilReady
} from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';
import type { ModelConfig, PerTokenInfo, TrainStepMetrics } from './engine';
import {
	initParams,
	lossFn,
	forwardLogprobs,
	forwardWithAttention,
	forwardResiduals,
	paramCount,
	flattenParams,
	loadParams,
	disposeTree
} from './model';

/* eslint-disable @typescript-eslint/no-explicit-any */

interface RpcRequest {
	id: number;
	op: string;
	[key: string]: unknown;
}

const post = (msg: unknown, transfer?: Transferable[]) =>
	(self as unknown as Worker).postMessage(msg, { transfer: transfer ?? [] });

// ── worker state ─────────────────────────────────────────────────────────────
let cfg: ModelConfig | null = null;
let params: any = null;
let optState: any = null;
let solver: ReturnType<typeof adam> | null = null;
let tokenData: Uint16Array | null = null;
let jitStep: any = null;
let jitForward: any = null;
let jitForwardBatch: any = null; // [G, S] group rollouts (RLVR sampling)
let jitRl: any = null; // advantage-weighted policy-gradient step
let jitLoss: any = null;
let jitGradNorms: any = null;
let jitAttn: any = null;
let jitResidual: any = null;
let causalMask: any = null;
let valStart = 0; // train windows come from [0, valStart); val from [valStart, end)
let stepCounter = 0;
let stopRequested = false;
let device = 'none';

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

const BATCH = 8;

/** Sample BATCH windows whose start index lies in [lo, hi). */
function makeBatchOH(
	c: ModelConfig,
	data: Uint16Array,
	rand: () => number,
	lo: number,
	hi: number
) {
	const S = c.blockSize;
	const nTok = BATCH * S;
	const inputBuf = new Int32Array(nTok);
	const targetBuf = new Int32Array(nTok);
	const span = Math.max(1, Math.min(hi, data.length - S - 1) - lo);
	for (let b = 0; b < BATCH; b++) {
		const start = lo + Math.floor(rand() * span);
		for (let i = 0; i < S; i++) {
			inputBuf[b * S + i] = data[start + i];
			targetBuf[b * S + i] = data[start + i + 1];
		}
	}
	const inputIds = np.array(inputBuf, { dtype: np.int32 }).reshape([BATCH, S]);
	const posIds = np.tile(np.arange(S).astype(np.int32), [BATCH, 1]);
	const targetIds = np.array(targetBuf, { dtype: np.int32 }).reshape([BATCH, S]);
	return {
		tokenOH: nn.oneHot(inputIds, c.vocab),
		posOH: nn.oneHot(posIds, c.blockSize),
		targetOH: nn.oneHot(targetIds, c.vocab)
	};
}

/** One optimizer step; returns the loss (synced — per-step sync is currently
 * FASTER than pipelining in jax-js, see upstream issue #151). */
function trainStep(): number {
	const c = cfg!;
	const { tokenOH, posOH, targetOH } = makeBatchOH(c, tokenData!, rng, 0, valStart);
	const [lossVal, grads] = jitStep(tree.ref(params), tokenOH.ref, posOH.ref, targetOH.ref);
	const [updates, newOptState] = solver!.update(grads, optState, tree.ref(params));
	params = applyUpdates(params, updates);
	optState = newOptState;
	tokenOH.dispose();
	posOH.dispose();
	targetOH.dispose();
	return lossVal.item();
}

/** Fixed-shape single-sequence forward: tokens right-padded to blockSize.
 * Causal attention makes padding after the last real position irrelevant, so
 * ONE jit signature serves every prompt length (no per-length recompiles). */
function forwardSeq(tokens: number[]): Float32Array {
	const c = cfg!;
	const S = c.blockSize;
	const buf = new Int32Array(S);
	for (let i = 0; i < Math.min(tokens.length, S); i++) buf[i] = tokens[i];
	const inputIds = np.array(buf, { dtype: np.int32 }).reshape([1, S]);
	const posIds = np.arange(S).astype(np.int32).reshape([1, S]);
	const tokenOH = nn.oneHot(inputIds, c.vocab);
	const posOH = nn.oneHot(posIds, c.blockSize);
	const logprobs = jitForward(tree.ref(params), tokenOH, posOH); // [S, vocab]
	return logprobs.dataSync() as Float32Array;
}

function sampleFromRow(row: Float32Array, temperature: number, topK: number): number {
	const V = row.length;
	// logprobs → scaled logits → topK → renormalized categorical draw
	const idx = Array.from({ length: V }, (_, i) => i);
	if (topK > 0 && topK < V) {
		idx.sort((a, b) => row[b] - row[a]);
		idx.length = topK;
	}
	const t = Math.max(temperature, 1e-4);
	let maxv = -Infinity;
	for (const i of idx) maxv = Math.max(maxv, row[i] / t);
	let sum = 0;
	const ps = idx.map((i) => {
		const p = Math.exp(row[i] / t - maxv);
		sum += p;
		return p;
	});
	let r = rng() * sum;
	for (let j = 0; j < idx.length; j++) {
		r -= ps[j];
		if (r <= 0) return idx[j];
	}
	return idx[idx.length - 1];
}

// ── op handlers ──────────────────────────────────────────────────────────────
async function handleInit(req: RpcRequest) {
	const devices = await init();
	if (!devices.includes('webgpu')) throw new Error('WebGPU unavailable in worker');
	defaultDevice('webgpu');
	device = 'webgpu';
	cfg = req.config as ModelConfig;
	tokenData = new Uint16Array(req.tokenData as ArrayBuffer);
	rng = mulberry32((req.seed as number) ?? 1234);
	stepCounter = 0;
	if (params) disposeTree(params);
	if (optState) disposeTree(optState);
	params = req.checkpoint
		? loadParams(cfg, new Float32Array(req.checkpoint as ArrayBuffer))
		: initParams(cfg, (req.seed as number) ?? 42);
	await blockUntilReady(params);
	solver = adam((req.lr as number) ?? 3e-4, { b1: 0.9, b2: 0.99 });
	optState = solver.init(tree.ref(params));
	const c = cfg;
	jitStep = jit((p: any, a: any, b: any, t: any) =>
		valueAndGrad((pp: any) => lossFn(pp, c, a, b, t))(p)
	);
	jitForward = jit((p: any, a: any, b: any) => forwardLogprobs(p, c, c.blockSize, a, b));
	jitLoss = jit((p: any, a: any, b: any, t: any) => lossFn(p, c, a, b, t));
	jitGradNorms = jit((p: any, a: any, b: any, t: any) => {
		const [lossVal, grads] = valueAndGrad((pp: any) => lossFn(pp, c, a, b, t))(p);
		return [lossVal, tree.map((g: any) => np.sum(np.square(g)), grads)];
	});
	// causal mask for the attention-capture forward (0 on/below diag, -1e9 above)
	if (causalMask) causalMask.dispose();
	const mbuf = new Float32Array(c.blockSize * c.blockSize);
	for (let i = 0; i < c.blockSize; i++) {
		for (let j = i + 1; j < c.blockSize; j++) mbuf[i * c.blockSize + j] = -1e9;
	}
	causalMask = np.array(mbuf).reshape([c.blockSize, c.blockSize]);
	jitAttn = jit((p: any, a: any, b: any, m: any) => forwardWithAttention(p, c, a, b, m));
	jitResidual = jit((p: any, a: any, b: any) => forwardResiduals(p, c, a, b));
	// held-out tail: training samples from [0, valStart), validation from the rest
	valStart = Math.floor(tokenData.length * 0.95);
	const minVal = 4 * (c.blockSize + 1);
	if (tokenData.length - valStart < minVal) {
		valStart = Math.max(1, tokenData.length - minVal);
	}
	return { device, paramCount: paramCount(params), tokens: tokenData.length };
}

async function handleTrain(req: RpcRequest) {
	const steps = (req.steps as number) ?? 50;
	stopRequested = false;
	let done = 0;
	for (let i = 0; i < steps; i++) {
		if (stopRequested) break;
		const t0 = performance.now();
		const loss = trainStep();
		const stepMs = performance.now() - t0;
		stepCounter++;
		done++;
		const m: TrainStepMetrics = {
			step: stepCounter,
			loss,
			stepMs,
			tokensPerSec: Math.round((BATCH * cfg!.blockSize * 1000) / stepMs)
		};
		post({ id: req.id, event: 'metrics', m });
		// Yield to the worker's own event loop so 'stop' messages get through.
		if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0));
	}
	return { completed: done, step: stepCounter };
}

/** Full log-prob distribution over the next token given a context — powers
 * widgets that need to see (and mask) the whole distribution, like the chess
 * board's legal-move masking. ~vocab×4 bytes, transferred. */
function handleNextDist(req: RpcRequest) {
	const c = cfg!;
	const tokens = ((req.tokens as number[]) ?? []).slice(-c.blockSize);
	const lp = forwardSeq(tokens);
	const pos = Math.min(tokens.length, c.blockSize) - 1;
	const row = lp.slice(pos * c.vocab, (pos + 1) * c.vocab);
	return { row: row.buffer, __transfer: [row.buffer] };
}

function handleSample(req: RpcRequest) {
	const c = cfg!;
	const prompt = (req.promptTokens as number[]) ?? [];
	const temperature = (req.temperature as number) ?? 0.8;
	const topK = (req.topK as number) ?? 40;
	const maxTokens = Math.min((req.maxTokens as number) ?? 120, c.blockSize - 1);
	const stopAt = req.stopToken as number | undefined;
	let tokens = prompt.slice(-Math.floor(c.blockSize / 2));
	const generated: number[] = [];
	for (let i = 0; i < maxTokens; i++) {
		const lp = forwardSeq(tokens);
		const pos = Math.min(tokens.length, c.blockSize) - 1;
		const row = lp.subarray(pos * c.vocab, (pos + 1) * c.vocab) as Float32Array;
		const next = sampleFromRow(row, temperature, topK);
		if (stopAt !== undefined && next === stopAt) break;
		generated.push(next);
		tokens.push(next);
		if (tokens.length >= c.blockSize) tokens = tokens.slice(-Math.floor(c.blockSize / 2));
	}
	return { tokens: generated };
}

function handleInspect(req: RpcRequest) {
	const c = cfg!;
	const tokens = (req.tokens as number[]).slice(0, c.blockSize);
	const lp = forwardSeq(tokens);
	const out: PerTokenInfo[] = [];
	for (let i = 0; i < tokens.length; i++) {
		const info: PerTokenInfo = { id: tokens[i], text: '' };
		if (i > 0) {
			// loss of THIS token under the model's prediction from position i-1
			const row = lp.subarray((i - 1) * c.vocab, i * c.vocab);
			info.loss = -row[tokens[i]];
			let entropy = 0;
			const pairs: Array<[number, number]> = [];
			for (let v = 0; v < c.vocab; v++) {
				const p = Math.exp(row[v]);
				if (p > 1e-9) entropy -= p * row[v];
				pairs.push([v, p]);
			}
			info.entropy = entropy;
			pairs.sort((a, b) => b[1] - a[1]);
			info.topk = pairs.slice(0, 5);
		}
		out.push(info);
	}
	return { perToken: out };
}

async function handleExport() {
	const flat = flattenParams(params);
	return { checkpoint: flat.buffer, __transfer: [flat.buffer] };
}

/** Swap the training corpus in place — params, optimizer state and jits all
 * survive. Continuing training on a curated corpus IS supervised fine-tuning;
 * this op exists so the SFT stage is honestly the same machine, better data. */
function handleSetTokens(req: RpcRequest) {
	tokenData = new Uint16Array(req.tokenData as ArrayBuffer);
	valStart = Math.floor(tokenData.length * 0.95);
	const minVal = 4 * (cfg!.blockSize + 1);
	if (tokenData.length - valStart < minVal) {
		valStart = Math.max(1, tokenData.length - minVal);
	}
	return { tokens: tokenData.length };
}

/** Sample G continuations of one shared prompt in lockstep — one [G,S]
 * forward per generated token instead of G separate loops. Feeds RLVR. */
function handleSampleGroup(req: RpcRequest) {
	const c = cfg!;
	const S = c.blockSize;
	const G = (req.g as number) ?? 8;
	const prompt = ((req.promptTokens as number[]) ?? [0]).slice(-Math.floor(S / 2));
	const maxNew = Math.min((req.maxNew as number) ?? 16, S - prompt.length - 1);
	const temperature = (req.temperature as number) ?? 1.0;
	const topK = (req.topK as number) ?? 40;
	const stopAt = req.stopToken as number | undefined;

	const seqs: number[][] = Array.from({ length: G }, () => prompt.slice());
	const done: boolean[] = Array(G).fill(false);

	if (!jitForwardBatch) {
		jitForwardBatch = jit((p: any, a: any, b: any) => forwardLogprobs(p, c, S, a, b));
	}

	for (let t = 0; t < maxNew; t++) {
		const L = prompt.length + t; // lockstep: every live row has length L
		const buf = new Int32Array(G * S);
		for (let g = 0; g < G; g++) {
			const s = seqs[g];
			for (let i = 0; i < Math.min(s.length, S); i++) buf[g * S + i] = s[i];
		}
		const inputIds = np.array(buf, { dtype: np.int32 }).reshape([G, S]);
		const posIds = np.tile(np.arange(S).astype(np.int32), [G, 1]);
		const tokenOH = nn.oneHot(inputIds, c.vocab);
		const posOH = nn.oneHot(posIds, c.blockSize);
		const lp = jitForwardBatch(tree.ref(params), tokenOH, posOH); // [G·S, V]
		const data = lp.dataSync() as Float32Array;
		let allDone = true;
		for (let g = 0; g < G; g++) {
			if (done[g]) continue;
			const row = data.subarray((g * S + L - 1) * c.vocab, (g * S + L) * c.vocab);
			const next = sampleFromRow(row as Float32Array, temperature, topK);
			if (stopAt !== undefined && next === stopAt) {
				done[g] = true;
				continue;
			}
			seqs[g].push(next);
			allDone = false;
		}
		if (allDone) break;
	}
	return { seqs: seqs.map((s) => s.slice(prompt.length)) };
}

/** One REINFORCE step from verifier rewards (the RLVR update, for real).
 * seqs: Int32 [G, S+1] rows (-1 padding); advs: per-sequence advantages
 * (standardize client-side — GRPO's group-relative trick); starts[g]: index
 * of the first GENERATED token — prompt positions get no gradient. Loss is
 * −Σ Â·log π(token) over generated positions, normalized by their count. */
function handleRlStep(req: RpcRequest) {
	const c = cfg!;
	const S = c.blockSize;
	const rows = new Int32Array(req.seqs as ArrayBuffer);
	const advs = req.advs as number[];
	const starts = req.starts as number[];
	const G = advs.length;
	if (rows.length !== G * (S + 1)) throw new Error('rlstep: bad seqs shape');

	const inputBuf = new Int32Array(G * S);
	const targetBuf = new Int32Array(G * S);
	const wBuf = new Float32Array(G * S);
	let total = 0;
	for (let g = 0; g < G; g++) {
		for (let t = 0; t < S; t++) {
			const a = rows[g * (S + 1) + t];
			const b = rows[g * (S + 1) + t + 1];
			inputBuf[g * S + t] = a < 0 ? 0 : a;
			targetBuf[g * S + t] = b < 0 ? 0 : b;
			// credit only generated tokens: position t predicts row t+1
			if (a >= 0 && b >= 0 && t + 1 >= starts[g]) {
				wBuf[g * S + t] = advs[g];
				total++;
			}
		}
	}
	if (total === 0) return { loss: 0, step: stepCounter, skipped: true };
	for (let i = 0; i < wBuf.length; i++) wBuf[i] /= total;

	const inputIds = np.array(inputBuf, { dtype: np.int32 }).reshape([G, S]);
	const posIds = np.tile(np.arange(S).astype(np.int32), [G, 1]);
	const targetIds = np.array(targetBuf, { dtype: np.int32 }).reshape([G, S]);
	const tokenOH = nn.oneHot(inputIds, c.vocab);
	const posOH = nn.oneHot(posIds, c.blockSize);
	const targetOH = nn.oneHot(targetIds, c.vocab);
	const w = np.array(wBuf).reshape([G * S, 1]);

	if (!jitRl) {
		jitRl = jit((p: any, tok: any, pos: any, tgt: any, ww: any) =>
			valueAndGrad((pp: any) => {
				const logp = forwardLogprobs(pp, c, S, tok, pos); // [G·S, V]
				return np.sum(logp.mul(tgt.reshape([-1, c.vocab]).mul(ww))).neg();
			})(p)
		);
	}
	const [lossVal, grads] = jitRl(tree.ref(params), tokenOH.ref, posOH.ref, targetOH.ref, w.ref);
	const [updates, newOptState] = solver!.update(grads, optState, tree.ref(params));
	params = applyUpdates(params, updates);
	optState = newOptState;
	tokenOH.dispose();
	posOH.dispose();
	targetOH.dispose();
	w.dispose();
	stepCounter++;
	return { loss: lossVal.item(), step: stepCounter };
}

/** Replace the resident weights with a checkpoint, in place — the config, jits
 * and token data are preserved, so this is far cheaper than a full re-init (no
 * tokenData re-transfer). Powers the time-machine scrubber: load a waypoint's
 * dequantized weights, then sample. Requires a prior init with the matching
 * config. Adam moments are reset (a loaded snapshot carries no optimizer
 * history); the time machine only samples, so that cost is inert there. */
async function handleLoad(req: RpcRequest) {
	if (!cfg || !solver) throw new Error('load before init');
	const flat = new Float32Array(req.checkpoint as ArrayBuffer);
	if (params) disposeTree(params);
	params = loadParams(cfg, flat);
	await blockUntilReady(params);
	if (optState) disposeTree(optState);
	optState = solver.init(tree.ref(params));
	stepCounter = 0;
	return { paramCount: paramCount(params) };
}

/** Swap the optimizer's learning rate. Rebuilds Adam state (moments reset —
 * honest cost of changing γ mid-flight; the lab says so). */
function handleSetLr(req: RpcRequest) {
	const lr = req.lr as number;
	if (!(lr > 0)) throw new Error(`bad lr: ${lr}`);
	if (optState) disposeTree(optState);
	solver = adam(lr, { b1: 0.9, b2: 0.99 });
	optState = solver.init(tree.ref(params));
	return { lr };
}

/** Mean loss over a few FIXED held-out batches (deterministic seed, so calls
 * across training are comparable — the curve is real validation loss). */
function handleValLoss() {
	const c = cfg!;
	const vr = mulberry32(9999);
	const N = 4;
	let total = 0;
	for (let i = 0; i < N; i++) {
		const { tokenOH, posOH, targetOH } = makeBatchOH(
			c,
			tokenData!,
			vr,
			valStart,
			tokenData!.length
		);
		const lossVal = jitLoss(tree.ref(params), tokenOH.ref, posOH.ref, targetOH.ref);
		tokenOH.dispose();
		posOH.dispose();
		targetOH.dispose();
		total += lossVal.item();
	}
	return { valLoss: total / N };
}

/** Per-tensor gradient L2 norms on a FIXED diagnostic batch — same batch every
 * call, so norms are comparable across training time. */
function handleGradNorms() {
	const c = cfg!;
	const gr = mulberry32(7777);
	const { tokenOH, posOH, targetOH } = makeBatchOH(c, tokenData!, gr, 0, valStart);
	const [lossVal, sq] = jitGradNorms(tree.ref(params), tokenOH.ref, posOH.ref, targetOH.ref);
	tokenOH.dispose();
	posOH.dispose();
	targetOH.dispose();
	const norms: Array<{ name: string; norm: number }> = [];
	const take = (t: any, name: string) => norms.push({ name, norm: Math.sqrt(t.item()) });
	take(sq.wte, 'wte');
	take(sq.wpe, 'wpe');
	take(sq.lmHead, 'lmHead');
	(sq.layers as any[]).forEach((l, i) => {
		for (const k of ['wq', 'wk', 'wv', 'wo', 'mlpFc1', 'mlpFc2'] as const) take(l[k], `L${i}.${k}`);
	});
	return { loss: lossVal.item(), norms };
}

/** Real attention patterns for a prompt: per layer, an [H·S, S] float block of
 * softmax(QKᵀ/√d + M) rows (crop to seqLen client-side). Transferred. */
function handleAttention(req: RpcRequest) {
	const c = cfg!;
	const S = c.blockSize;
	const tokens = ((req.tokens as number[]) ?? []).slice(-S);
	const buf = new Int32Array(S);
	for (let i = 0; i < tokens.length; i++) buf[i] = tokens[i];
	const inputIds = np.array(buf, { dtype: np.int32 }).reshape([1, S]);
	const posIds = np.arange(S).astype(np.int32).reshape([1, S]);
	const tokenOH = nn.oneHot(inputIds, c.vocab);
	const posOH = nn.oneHot(posIds, c.blockSize);
	const [logprobs, attn] = jitAttn(tree.ref(params), tokenOH, posOH, causalMask.ref);
	logprobs.dispose();
	const layers: ArrayBuffer[] = [];
	for (const a of attn as any[]) {
		const d = a.dataSync() as Float32Array;
		layers.push(d.buffer as ArrayBuffer);
	}
	return {
		layers,
		nHead: c.nHead,
		blockSize: S,
		seqLen: tokens.length,
		__transfer: layers
	};
}

/** Residual stream after every block for a token sequence: nLayer buffers,
 * each [seqLen · nEmbd] (sliced to the real length). The surface linear probes
 * read to test whether the board is linearly decodable. Transferred. */
function handleResidual(req: RpcRequest) {
	const c = cfg!;
	const S = c.blockSize;
	const tokens = ((req.tokens as number[]) ?? []).slice(-S);
	const seqLen = tokens.length;
	const buf = new Int32Array(S);
	for (let i = 0; i < seqLen; i++) buf[i] = tokens[i];
	const inputIds = np.array(buf, { dtype: np.int32 }).reshape([1, S]);
	const posIds = np.arange(S).astype(np.int32).reshape([1, S]);
	const tokenOH = nn.oneHot(inputIds, c.vocab);
	const posOH = nn.oneHot(posIds, c.blockSize);
	const [logprobs, res] = jitResidual(tree.ref(params), tokenOH, posOH);
	logprobs.dispose();
	const layers: ArrayBuffer[] = [];
	for (const r of res as any[]) {
		const full = r.dataSync() as Float32Array; // [S · nEmbd]
		const sliced = full.slice(0, seqLen * c.nEmbd); // real positions only
		layers.push(sliced.buffer as ArrayBuffer);
	}
	return { layers, seqLen, nEmbd: c.nEmbd, nLayer: c.nLayer, __transfer: layers };
}

// ── dispatch ─────────────────────────────────────────────────────────────────
const handlers: Record<string, (req: RpcRequest) => unknown | Promise<unknown>> = {
	init: handleInit,
	train: handleTrain,
	stop: () => {
		stopRequested = true;
		return {};
	},
	sample: handleSample,
	samplegroup: handleSampleGroup,
	settokens: handleSetTokens,
	rlstep: handleRlStep,
	nextdist: handleNextDist,
	inspect: handleInspect,
	export: handleExport,
	load: handleLoad,
	valloss: handleValLoss,
	gradnorms: handleGradNorms,
	attention: handleAttention,
	residual: handleResidual,
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
	// 'stop' must preempt: handle it synchronously even mid-train.
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

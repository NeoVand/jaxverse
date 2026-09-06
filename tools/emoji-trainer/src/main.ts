// Offline trainer for the shipped emoji checkpoints.
//
// It runs in headless Chromium so that the weights the book ships were
// produced by the same jax-js, the same kernels and the same code as the
// weights a reader trains in their own tab. scripts/train-emoji.mjs drives it
// and pulls snapshots out through window.__snapshot().

import { blockUntilReady, init, defaultDevice, jit, numpy as np, tree } from '@jax-js/jax';
import {
	EMOJI_SHAPE,
	initParams,
	type DiffusionConfig,
	type Objective
} from '$lib/diffusion/model';
import { loadEmoji } from '$lib/diffusion/corpus';
import {
	allocBatch,
	makeBatch,
	makeOptimizer,
	type BatchOptions,
	makeSampler,
	mulberry32,
	type Corpus
} from '$lib/diffusion/runtime';
import { packFloat32, packInt8, unpack } from '$lib/diffusion/checkpoint';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Arr = any;

const out = document.getElementById('out') as HTMLPreElement;
const preview = document.getElementById('preview') as HTMLCanvasElement;
let head = '';
let tail: string[] = [];
const log = (s: string) => {
	tail.push(s);
	if (tail.length > 14) tail = tail.slice(-14);
	out.textContent = `${head}\n${tail.join('\n')}`;
};

const params = new URLSearchParams(location.search);
const OBJECTIVE = (params.get('objective') ?? 'flow') as Objective;
const BATCH = Number(params.get('batch') ?? 32);
const LR = Number(params.get('lr') ?? 3e-4);
const WARMUP = 300;
/** Horizon for the cosine decay. Stopping early just means it never finished
 *  decaying, which costs a little polish and nothing else. */
const HORIZON = Number(params.get('horizon') ?? 55000);
/** Weight-average decay. At 0.9995 the average remembers roughly the last two
 *  thousand steps, which is the right scale for runs of tens of thousands. */
const EMA_DECAY = Number(params.get('ema') ?? 0.9995);
/** Architecture and data recipe, so a run can be launched on whatever
 *  tools/recipe-ablation picked without editing this file. */
const PATCH = Number(params.get('patch') ?? EMOJI_SHAPE.patch);
const DIM = Number(params.get('dim') ?? EMOJI_SHAPE.dim);
const LAYERS = Number(params.get('layers') ?? EMOJI_SHAPE.layers);
const HEADS = Number(params.get('heads') ?? EMOJI_SHAPE.heads);
const JITTER = Number(params.get('jitter') ?? 3);
/**
 * Uniform, on the evidence rather than the fashion. Logit-normal sampling is
 * what Stable Diffusion 3 reports its largest gain from, and at this size and
 * budget it was measurably worse: it puts about 1.4% of samples below τ = 0.1
 * against uniform's 10%, and those near-clean levels are what sharpen the last
 * steps of the walk. See tools/recipe-ablation.
 */
const TSAMPLE = (params.get('tsample') ?? 'uniform') as BatchOptions['tSample'];
const CLIP = Number(params.get('clip') ?? 1);

declare global {
	interface Window {
		__snapshot?: (kind: 'f32' | 'i8') => string;
		__resume?: (b64: string) => void;
		__stop?: () => void;
		__state?: { step: number; loss: number; ready: boolean };
	}
}

function toBase64(buf: ArrayBuffer): string {
	const bytes = new Uint8Array(buf);
	let s = '';
	for (let i = 0; i < bytes.length; i += 0x8000) {
		s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
	}
	return btoa(s);
}

function fromBase64(b64: string): ArrayBuffer {
	const bin = atob(b64);
	const bytes = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
	return bytes.buffer;
}

/** Draw eight samples so a human watching the run can see it working. */
function paint(frame: Float32Array, cfg: DiffusionConfig, n: number) {
	const { res, channels } = cfg;
	const dim = channels * res * res;
	const ctx = preview.getContext('2d')!;
	const img = ctx.createImageData(res * n, res);
	for (let k = 0; k < n; k++) {
		for (let y = 0; y < res; y++) {
			for (let x = 0; x < res; x++) {
				const p = y * res + x;
				const s = k * dim + p;
				const d = (y * res * n + k * res + x) * 4;
				const a = Math.min(Math.max((frame[s + 3 * res * res] + 1) * 127.5, 0), 255);
				for (let c = 0; c < 3; c++) {
					const v = Math.min(Math.max((frame[s + c * res * res] + 1) * 127.5, 0), 255);
					// premultiplied over the mid grey the canvas is painted with
					img.data[d + c] = Math.round(v + 136 * (1 - a / 255));
				}
				img.data[d + 3] = 255;
			}
		}
	}
	ctx.putImageData(img, 0, 0);
	ctx.imageSmoothingEnabled = false;
	ctx.drawImage(preview, 0, 0, res * n, res, 0, 0, preview.width, preview.height);
}

async function main() {
	const devices = await init();
	if (!devices.includes('webgpu')) {
		log('error: no WebGPU');
		return;
	}
	defaultDevice('webgpu');

	log('loading corpus…');
	const emoji = await loadEmoji('');
	const cfg: DiffusionConfig = {
		...EMOJI_SHAPE,
		patch: PATCH,
		dim: DIM,
		layers: LAYERS,
		heads: HEADS,
		tags: emoji.meta.tags.length,
		styles: emoji.meta.sets.length,
		objective: OBJECTIVE
	};
	const corpus: Corpus = emoji;
	const batchOpts: BatchOptions = { tSample: TSAMPLE, jitter: JITTER };

	let weights: Arr = initParams(cfg, 20260905);
	let step = 0;
	const nParams = (tree.leaves(tree.ref(weights)) as Arr[]).reduce((s: number, l: Arr) => {
		const n = l.size;
		l.dispose();
		return s + n;
	}, 0);
	head =
		`objective ${OBJECTIVE} · ${(nParams / 1e6).toFixed(2)}M params · batch ${BATCH} · lr ${LR}\n` +
		`${emoji.count} emoji x ${emoji.styles} styles · ${cfg.tags} tags`;

	let opt = makeOptimizer(cfg, BATCH, weights, CLIP);

	// An exponential moving average of the weights, kept alongside the ones
	// being trained and used for every picture the model draws.
	//
	// This is standard in every diffusion implementation and it is not a
	// nicety: the weights bounce around their own trajectory from batch to
	// batch, and the average of the last few thousand positions is a
	// consistently better model than wherever the walk happens to be standing.
	// The raw weights are what gets saved for resuming; the average is what
	// gets shipped.
	let emaWeights: Arr = tree.map((l: Arr) => l.mul(1), tree.ref(weights));
	const emaStep = jit((e: Arr, p: Arr, k: Arr) => {
		const [el, def] = tree.flatten(e) as [Arr[], Arr];
		const pl = tree.leaves(p) as Arr[];
		const keep = k.ref.slice([0, 1]);
		const take = k.slice([1, 2]);
		const out: Arr[] = [];
		for (let i = 0; i < el.length; i++) {
			out.push(el[i].mul(keep.ref).add(pl[i].mul(take.ref)));
		}
		keep.dispose();
		take.dispose();
		return tree.unflatten(def, out);
	});
	// Two batch buffers, ping-ponged: the CPU builds the next one while the GPU
	// is still busy with this one. Corrupting a picture is a few hundred
	// thousand transcendentals per step, and unoverlapped it was most of the
	// wall clock.
	let batch = allocBatch(cfg, BATCH);
	let spare = allocBatch(cfg, BATCH);
	const rand = mulberry32(1234);
	const sampler = makeSampler(cfg, 8);
	makeBatch(corpus, cfg, BATCH, rand, batch, batchOpts);

	let stopped = false;
	window.__stop = () => {
		stopped = true;
	};
	// int8 ships and is drawn from, so it carries the average; float32 exists
	// only to resume a run, so it carries the weights the optimizer is on.
	window.__snapshot = (kind) =>
		toBase64(
			kind === 'i8' ? packInt8(emaWeights, step, OBJECTIVE) : packFloat32(weights, step, OBJECTIVE)
		);
	// Resume is deferred to a safe point in the loop rather than applied where
	// the driver happens to call it. Applied inline it lands during one of the
	// loop's awaits, frees the weights and the average out from under whatever
	// is waiting on them, and the run hangs on the spot.
	let pendingResume: string | null = null;
	window.__resume = (b64) => {
		pendingResume = b64;
	};
	window.__state = { step: 0, loss: NaN, ready: true };

	let lossEma = NaN;
	let sinceLog = performance.now();
	// Warmup counts from when THIS process started, not from the checkpoint's
	// step: a resumed run has restored weights but empty Adam moments, and a
	// full-rate step on top of empty moments is how a run dies in its first
	// second. Decay, by contrast, follows the absolute step.
	let sinceStart = 0;
	for (;;) {
		if (stopped) break;
		if (pendingResume !== null) {
			const loaded = unpack(fromBase64(pendingResume), weights);
			pendingResume = null;
			weights = loaded.params;
			step = loaded.steps;
			tree.dispose(emaWeights);
			emaWeights = tree.map((l: Arr) => l.mul(1), tree.ref(weights));
			opt.dispose();
			opt = makeOptimizer(cfg, BATCH, weights, CLIP);
			sinceStart = 0;
			log(`resumed at step ${step}`);
		}
		const warm = Math.min(1, ++sinceStart / WARMUP);
		const decay = 0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, step / HORIZON)));
		const lr = LR * warm * decay;
		const [lossArr, next] = opt.step(weights, batch, lr);
		weights = next;
		// step() has already copied the batch to the device, so the spare can be
		// filled while that work is in flight; the loss read below is the sync.
		makeBatch(corpus, cfg, BATCH, rand, spare, batchOpts);
		[batch, spare] = [spare, batch];
		const loss = lossArr.item();
		step++;

		// Ramp the decay in rather than starting at the target: an average that
		// remembers ten thousand steps is useless on the first hundred, when it
		// is still mostly remembering the random initialization.
		const d = Math.min(EMA_DECAY, (1 + sinceStart) / (10 + sinceStart));
		emaWeights = emaStep(emaWeights, tree.ref(weights), np.array(new Float32Array([d, 1 - d])));

		lossEma = Number.isNaN(lossEma) ? loss : lossEma * 0.98 + loss * 0.02;
		window.__state = { step, loss: lossEma, ready: true };

		if (step % 200 === 0) {
			const now = performance.now();
			const ms = (now - sinceLog) / 200;
			sinceLog = now;
			log(
				`step ${step} · loss ${lossEma.toFixed(4)} · ${ms.toFixed(0)} ms/step · lr ${lr.toFixed(6)}`
			);
		}
		if (step % 20 === 0) {
			// Force the average. Reading the loss each step realizes the weight
			// graph, but nothing ever reads the average, so its deferred
			// operations pile up one per step and the stack overflows a few
			// thousand steps in.
			await blockUntilReady(tree.ref(emaWeights));
			// Yield long enough that the driver's CDP calls get scheduled: too
			// rare or too short a yield leaves the main thread busy almost all
			// the time and the run looks hung from outside while it is fine.
			await new Promise((r) => setTimeout(r, 2));
		}
		if (step % 2000 === 0) {
			const frame = await sampler.run({
				// draw from the average, which is what ships
				params: tree.ref(emaWeights),
				steps: OBJECTIVE === 'flow' ? 24 : 40,
				a: { tags: [], style: null },
				guidanceA: 1,
				seed: 99
			});
			paint(frame, cfg, 8);
			await new Promise((r) => setTimeout(r, 0));
		}
	}
	opt.dispose();
	tree.dispose(emaWeights);
	log('stopped');
}

void main().catch((e) => {
	log(`error: ${e?.stack ?? e}`);
});

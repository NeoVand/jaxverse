// The training step and the samplers: everything that turns the network in
// model.ts into a thing that learns and a thing that draws.
//
// Both chapters and the offline trainer import this file, so a checkpoint
// trained overnight and a checkpoint trained in the reader's own tab came out
// of exactly the same arithmetic.

import { numpy as np, jit, valueAndGrad, tree } from '@jax-js/jax';
import {
	type DiffusionConfig,
	alphaBar,
	condWidth,
	forward,
	lossFn,
	TIME_FEATURES,
	writeTimeFeatures
} from './model';

// jax-js arrays are consumed on use and typed loosely at this seam.
/* eslint-disable @typescript-eslint/no-explicit-any */
type Arr = any;

/** Deterministic, seedable, and identical in the worker and the trainer. */
export function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** Box–Muller. The noise this whole subject is about has to be Gaussian. */
export function gaussianFill(buf: Float32Array, rand: () => number): void {
	for (let i = 0; i < buf.length; i += 2) {
		const u = Math.max(rand(), 1e-7);
		const r = Math.sqrt(-2 * Math.log(u));
		const th = 2 * Math.PI * rand();
		buf[i] = r * Math.cos(th);
		if (i + 1 < buf.length) buf[i + 1] = r * Math.sin(th);
	}
}

// ------------------------------------------------------------ conditions ---

export interface Condition {
	/** Which garment to draw, or null for "anything". */
	label: number | null;
	/**
	 * A second garment to lean towards, with `mix` saying how far.
	 *
	 * Training only ever showed the model a one-hot, so a half-and-half vector
	 * is a question nobody asked it during training. It answers anyway, and
	 * what it answers is the subject of one of the plates.
	 */
	other?: number | null;
	/** 0 keeps `label`, 1 arrives fully at `other`. */
	mix?: number;
}

export const NOTHING: Condition = { label: null };

/**
 * Write one row of the conditioning vector.
 *
 * The trailing flag says whether the label block means anything. Without it
 * an absent label and class zero would arrive as the same all-zero block, and
 * guidance works by asking the model the same question twice — once with the
 * label and once without — so the two have to be distinguishable.
 */
export function writeCondition(
	dst: Float32Array,
	row: number,
	c: DiffusionConfig,
	tau: number,
	cond: Condition
): void {
	const w = condWidth(c);
	const off = row * w;
	dst.fill(0, off, off + w);
	writeTimeFeatures(dst, off, tau);
	const labelBase = off + TIME_FEATURES;
	const inRange = (l: number | null | undefined): l is number =>
		l !== null && l !== undefined && l >= 0 && l < c.classes;
	const mix = cond.mix ?? 0;
	if (inRange(cond.label)) {
		dst[labelBase + cond.label] = inRange(cond.other) ? 1 - mix : 1;
		dst[labelBase + c.classes] = 1;
	}
	if (inRange(cond.other) && mix > 0) {
		dst[labelBase + cond.other] += mix;
		dst[labelBase + c.classes] = 1;
	}
}

// -------------------------------------------------------------- training ---

/** Explicitly backed by ArrayBuffer, not ArrayBufferLike: jax-js will not
 *  accept a view that might sit on a SharedArrayBuffer. */
export interface TrainBatch {
	x: Float32Array<ArrayBuffer>;
	cond: Float32Array<ArrayBuffer>;
	target: Float32Array<ArrayBuffer>;
}

export interface Corpus {
	/**
	 * Grayscale ink, one byte per pixel: picture i, pixel p lives at
	 * i · res² + p. Kept as bytes because that is the precision the dataset
	 * arrived in, and because the float version is four times the size.
	 */
	images: Uint8Array;
	/** Class index, one per picture. */
	labels: Uint8Array;
	count: number;
}

/** Bytes to the [-1, 1] the network trains on. */
export const toUnit = (b: number) => b / 127.5 - 1;

export interface BatchOptions {
	/**
	 * Where along the ruin to spend training effort.
	 *
	 * `uniform` treats every noise level as equally worth learning. It is the
	 * obvious choice and it is not the best one: the levels near either end are
	 * nearly free — at the clean end there is almost nothing to remove, at the
	 * noisy end almost nothing to recover — while the middle is where the
	 * picture is actually decided. `logit-normal` draws the level as a
	 * sigmoid of a standard normal, which puts most of the samples in that
	 * middle. It is the change Stable Diffusion 3 reported the largest gain from.
	 */
	tSample?: 'uniform' | 'logit-normal';
	/**
	 * Shift each picture by up to this many pixels in each direction.
	 *
	 * Garments are photographed tight to the frame, so a large shift crops a
	 * sleeve off rather than moving it. Small values only.
	 */
	jitter?: number;
	/** Mirror half the batch. A shoe faces either way, so this is free data. */
	flip?: boolean;
}

export function allocBatch(c: DiffusionConfig, batch: number): TrainBatch {
	const dim = c.channels * c.res * c.res;
	return {
		x: new Float32Array(batch * dim),
		cond: new Float32Array(batch * condWidth(c)),
		target: new Float32Array(batch * dim)
	};
}

/**
 * One training example is: take a picture, choose how far to destroy it,
 * destroy it that far, and write down what would undo the damage. The only
 * thing the two chapters disagree about is that last clause.
 *
 * A tenth of the rows drop the label, so one set of weights learns both the
 * conditional field and the unconditional one that guidance subtracts from it.
 */
export function makeBatch(
	corpus: Corpus,
	c: DiffusionConfig,
	batch: number,
	rand: () => number,
	out: TrainBatch,
	opts: BatchOptions = {}
): void {
	const { res } = c;
	const plane = res * res;
	const dim = c.channels * plane;
	const noise = new Float32Array(dim);
	const clean = new Float32Array(dim);
	const jitter = opts.jitter ?? 0;

	for (let b = 0; b < batch; b++) {
		const idx = Math.floor(rand() * corpus.count);
		const src = idx * dim;
		gaussianFill(noise, rand);

		// Read the picture out with its augmentation applied. Pixels shifted in
		// from outside the frame are black, which is the value the empty
		// background already has, so the shift introduces no edge.
		const dx = jitter ? Math.round((rand() * 2 - 1) * jitter) : 0;
		const dy = jitter ? Math.round((rand() * 2 - 1) * jitter) : 0;
		const mirror = opts.flip === true && rand() < 0.5;
		if (dx === 0 && dy === 0 && !mirror) {
			for (let i = 0; i < dim; i++) clean[i] = toUnit(corpus.images[src + i]);
		} else {
			clean.fill(-1);
			for (let y = 0; y < res; y++) {
				const sy = y - dy;
				if (sy < 0 || sy >= res) continue;
				for (let x = 0; x < res; x++) {
					const sx = mirror ? res - 1 - (x - dx) : x - dx;
					if (sx < 0 || sx >= res) continue;
					for (let ch = 0; ch < c.channels; ch++) {
						clean[ch * plane + y * res + x] = toUnit(
							corpus.images[src + ch * plane + sy * res + sx]
						);
					}
				}
			}
		}

		const tau = opts.tSample === 'logit-normal' ? 1 / (1 + Math.exp(-gaussian1(rand))) : rand();

		const o = b * dim;
		if (c.objective === 'flow') {
			for (let i = 0; i < dim; i++) {
				out.x[o + i] = (1 - tau) * clean[i] + tau * noise[i];
				out.target[o + i] = noise[i] - clean[i];
			}
		} else {
			const ab = alphaBar(tau);
			const sa = Math.sqrt(ab);
			const sn = Math.sqrt(1 - ab);
			for (let i = 0; i < dim; i++) {
				out.x[o + i] = sa * clean[i] + sn * noise[i];
				out.target[o + i] = noise[i];
			}
		}

		writeCondition(out.cond, b, c, tau, {
			label: rand() < 0.1 ? null : corpus.labels[idx]
		});
	}
}

/** One standard normal draw. */
function gaussian1(rand: () => number): number {
	const u = Math.max(rand(), 1e-7);
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

export interface Optimizer {
	/** Consumes `params`, returns the loss and the updated parameters. */
	step(params: Arr, b: TrainBatch, lr: number): [Arr, Arr];
	dispose(): void;
}

/**
 * Adam, hand-rolled so the update fits inside the jit boundary.
 *
 * optax's version reads its own step counter with .item(), which a tracer
 * cannot answer, so its update has to run outside jit — and outside jit it
 * dispatches several tiny kernels per parameter. With forty-odd parameters
 * that launch traffic, not the matmuls, is most of the step: 151 ms measured
 * split against 88 ms fused. The bias corrections and the learning rate come
 * in as one traced triple instead, so the shape signature never changes and
 * the kernel compiles once.
 */
export function makeOptimizer(
	c: DiffusionConfig,
	batch: number,
	params: Arr,
	/**
	 * Clip the gradient to this global norm before the Adam step.
	 *
	 * Standard in diffusion training and omitted from the first version of this
	 * model, which was fine at 64 tokens and diverged outright at 256: a single
	 * unlucky batch at a high noise level produces a gradient large enough to
	 * throw the weights somewhere they never recover from. Clipping costs one
	 * extra reduction per step and removes the failure mode.
	 */
	clipNorm = 1
): Optimizer {
	let m: Arr = tree.map((l: Arr) => np.zerosLike(l), tree.ref(params));
	let v: Arr = tree.map((l: Arr) => np.zerosLike(l), tree.ref(params));
	let t = 0;

	const fused = jit((p: Arr, mm: Arr, vv: Arr, k: Arr, x: Arr, cond: Arr, target: Arr) => {
		const [loss, grads] = valueAndGrad((pp: Arr) => lossFn(pp, c, batch, x, cond, target))(
			tree.ref(p)
		);
		const [leaves, def] = tree.flatten(p) as [Arr[], Arr];
		let gl = tree.leaves(grads) as Arr[];
		const ml = tree.leaves(mm) as Arr[];
		const vl = tree.leaves(vv) as Arr[];
		const c1 = k.ref.slice([0, 1]);
		const c2 = k.ref.slice([1, 2]);
		const lr = k.slice([2, 3]);

		if (clipNorm > 0) {
			// one global norm across every tensor, then a single shared rescale
			let sq: Arr = null;
			for (const g of gl) {
				const term = np.sum(np.square(g.ref));
				sq = sq === null ? term : sq.add(term);
			}
			const norm = np.sqrt(sq).add(1e-6);
			const cap = np.array(new Float32Array([clipNorm]));
			const scale = np.minimum(cap.div(norm), 1);
			const last = gl.length - 1;
			gl = gl.map((g, i) => g.mul(i === last ? scale : scale.ref));
		}

		const nextP: Arr[] = [];
		const nextM: Arr[] = [];
		const nextV: Arr[] = [];
		for (let i = 0; i < gl.length; i++) {
			const mi = ml[i].mul(0.9).add(gl[i].ref.mul(0.1));
			const vi = vl[i].mul(0.99).add(np.square(gl[i]).mul(0.01));
			const mhat = mi.ref.mul(c1.ref);
			const vhat = vi.ref.mul(c2.ref);
			nextP.push(leaves[i].sub(mhat.mul(lr.ref).div(np.sqrt(vhat).add(1e-8))));
			nextM.push(mi);
			nextV.push(vi);
		}
		c1.dispose();
		c2.dispose();
		lr.dispose();
		return [
			loss,
			tree.unflatten(def, nextP),
			tree.unflatten(def, nextM),
			tree.unflatten(def, nextV)
		];
	});

	return {
		step(p: Arr, b: TrainBatch, lr: number) {
			t++;
			const k = np.array(
				new Float32Array([1 / (1 - Math.pow(0.9, t)), 1 / (1 - Math.pow(0.99, t)), lr])
			);
			const x = np.array(b.x).reshape([batch, c.channels, c.res, c.res]);
			const cond = np.array(b.cond).reshape([batch, condWidth(c)]);
			const target = np.array(b.target).reshape([batch, c.channels, c.res, c.res]);
			const [loss, p2, m2, v2] = fused(p, m, v, k, x, cond, target);
			m = m2;
			v = v2;
			return [loss, p2];
		},
		dispose() {
			tree.dispose(m);
			tree.dispose(v);
		}
	};
}

// -------------------------------------------------------------- sampling ---

/**
 * Guidance always runs three branches — unconditional, prompt A, prompt B —
 * so the compiled shape never changes. A single-prompt request pays for one
 * branch it does not need, which is cheaper than a recompile.
 */
export const BRANCHES = 3;

export interface SampleRequest {
	/** Consumed: pass `tree.ref(weights)` if you still need them afterwards. */
	params: Arr;
	/** Denoising steps. Fewer is faster, and for `eps` also worse. */
	steps: number;
	a: Condition;
	b?: Condition;
	/** Guidance strengths. 1 is the raw conditional field; 0 is unconditional. */
	guidanceA: number;
	guidanceB?: number;
	/** Fixed noise, so the same seed keeps returning the same picture. */
	seed: number;
	/** 0 is deterministic (DDIM); 1 is ancestral DDPM. Ignored for `flow`. */
	eta?: number;
	/**
	 * Receives a frame after every step, for the trajectory plates.
	 * `state` is the picture as it actually is at that rung; `endpoint` is the
	 * finished picture the model's answer implies from there, which is the
	 * thing that reveals how curved the path is.
	 */
	trace?: 'state' | 'endpoint';
	onStep?: (frame: Float32Array, done: number, steps: number) => void;
}

export interface Sampler {
	/** Resolves to [count · channels · res · res] in roughly [-1, 1]. */
	run(req: SampleRequest): Promise<Float32Array>;
}

/**
 * Composable guidance, which is one line of arithmetic and the reason two
 * prompts can be answered with one picture.
 *
 * Each branch predicts a field over the same image. The unconditional branch
 * says what any picture would do from here; a conditional branch says what a
 * picture matching that prompt would do. The difference between them is what
 * the prompt is asking for — and differences are vectors, so they add.
 */
export function makeSampler(c: DiffusionConfig, count: number): Sampler {
	const dim = c.channels * c.res * c.res;
	const B = BRANCHES * count;
	const net = jit((p: Arr, x: Arr, cond: Arr) => forward(p, c, B, x, cond));

	const xBuf = new Float32Array(B * dim);
	const condBuf = new Float32Array(B * condWidth(c));
	const state = new Float32Array(count * dim);
	const field = new Float32Array(count * dim);
	const noise = new Float32Array(count * dim);
	const endpoint = new Float32Array(count * dim);

	return {
		async run(req: SampleRequest): Promise<Float32Array> {
			const rand = mulberry32(req.seed);
			gaussianFill(state, rand);
			const steps = Math.max(1, req.steps);
			const condB = req.b ?? NOTHING;
			const wB = req.b ? (req.guidanceB ?? 0) : 0;

			for (let s = 0; s < steps; s++) {
				const tau = 1 - s / steps; // 1 is pure noise, 0 is a finished picture
				const tauNext = 1 - (s + 1) / steps;

				for (let br = 0; br < BRANCHES; br++) {
					xBuf.set(state, br * count * dim);
				}
				for (let n = 0; n < count; n++) {
					writeCondition(condBuf, n, c, tau, NOTHING);
					writeCondition(condBuf, count + n, c, tau, req.a);
					writeCondition(condBuf, 2 * count + n, c, tau, condB);
				}

				const out = await net(
					tree.ref(req.params),
					np.array(xBuf).reshape([B, c.channels, c.res, c.res]),
					np.array(condBuf).reshape([B, condWidth(c)])
				).data();

				const aOff = count * dim;
				const bOff = 2 * count * dim;
				for (let i = 0; i < field.length; i++) {
					const u = out[i];
					field[i] = u + req.guidanceA * (out[aOff + i] - u) + wB * (out[bOff + i] - u);
				}

				if (c.objective === 'flow') {
					// Euler, along a path training made as straight as it could
					const dt = tau - tauNext;
					if (req.trace === 'endpoint') {
						// x_τ = (1−τ)x₀ + τε and v = ε − x₀, so x₀ = x_τ − τv.
						// Clamped exactly like the eps branch below: the trace is a
						// comparison between the two, and clamping one and not the
						// other would decide it before the models did.
						for (let i = 0; i < state.length; i++) {
							endpoint[i] = Math.min(Math.max(state[i] - tau * field[i], -1), 1);
						}
					}
					for (let i = 0; i < state.length; i++) state[i] -= dt * field[i];
				} else {
					const ab = alphaBar(tau);
					const abNext = tauNext <= 0 ? 1 : alphaBar(tauNext);
					const sa = Math.sqrt(ab);
					const sn = Math.sqrt(1 - ab);
					const eta = tauNext <= 0 ? 0 : (req.eta ?? 0);
					const sigma = eta * Math.sqrt(((1 - abNext) / (1 - ab)) * Math.max(1 - ab / abNext, 0));
					const keep = Math.sqrt(Math.max(1 - abNext - sigma * sigma, 0));
					if (sigma > 0) gaussianFill(noise, rand);
					for (let i = 0; i < state.length; i++) {
						// clamping the implied clean image is what stops a short
						// schedule from walking off into saturated garbage
						const x0 = Math.min(Math.max((state[i] - sn * field[i]) / sa, -1), 1);
						if (req.trace === 'endpoint') endpoint[i] = x0;
						state[i] = Math.sqrt(abNext) * x0 + keep * field[i];
						if (sigma > 0) state[i] += sigma * noise[i];
					}
				}
				req.onStep?.(req.trace === 'endpoint' ? endpoint : state, s + 1, steps);
			}
			tree.dispose(req.params);
			return state;
		}
	};
}

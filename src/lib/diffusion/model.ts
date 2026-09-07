// The garment denoiser, shared by the training worker and the offline trainer.
//
// A diffusion transformer: patchify a 28x28 grayscale picture into 7x7 tokens,
// run pre-norm attention blocks over them, un-patchify back to pixels. A conv
// U-Net was built and timed first and lost — 264 ms per step against the
// transformer's 88 — because a U-Net spends its whole budget on 3x3 kernels
// at full resolution, while everything here happens on 49 tokens, which is
// the shape jax-js is fastest at. It is also the architecture the current
// generation of image models actually uses.
//
// One network serves both chapters. `objective` decides what it is asked to
// predict — the noise that was added, or the velocity that removes it — and
// nothing else about the machine changes.

import { numpy as np, nn, lax, random, tree } from '@jax-js/jax';

// jax-js arrays are consumed on use and typed loosely at this seam; `Arr`
// marks the places where that is deliberate rather than sloppy.
/* eslint-disable @typescript-eslint/no-explicit-any */
type Arr = any;

export type Objective = 'eps' | 'flow';

export interface DiffusionConfig {
	/** Display name for the run — no behaviour. */
	name?: string;
	/** Side of the square image. */
	res: number;
	/** Grayscale ink, one channel. */
	channels: number;
	/** Token side; res/patch must be a whole number. */
	patch: number;
	dim: number;
	layers: number;
	heads: number;
	/** How many garment classes the label one-hot covers. */
	classes: number;
	objective: Objective;
}

export const FASHION_SHAPE = {
	name: 'fashion',
	res: 28,
	channels: 1,
	patch: 4,
	dim: 192,
	layers: 4,
	heads: 4,
	classes: 10
} as const;

/** Sinusoidal features for the noise level; 16 frequencies, sin and cos. */
export const TIME_FEATURES = 32;

export const tokenCount = (c: DiffusionConfig) => (c.res / c.patch) ** 2;
/** time ‖ label one-hot ‖ one flag saying whether the label is present */
export const condWidth = (c: DiffusionConfig) => TIME_FEATURES + c.classes + 1;

export function paramCount(c: DiffusionConfig): number {
	const { dim: d, layers, channels: ch, patch } = c;
	return (
		condWidth(c) * d +
		d * d +
		d * ch * patch * patch +
		tokenCount(c) * d +
		layers * (d * 4 * d + 4 * d * d + d * 4 * d + 4 * d * d) +
		d * 2 * d +
		ch * d * patch * patch
	);
}

// ------------------------------------------------------------- schedules ---

/**
 * Cosine schedule: the fraction of the signal still present at noise level τ.
 * Linear schedules spend most of their steps on images already destroyed;
 * the cosine spends them where the picture is still deciding what it is.
 */
export function alphaBar(tau: number): number {
	const s = 0.008;
	const f = (u: number) => Math.cos(((u + s) / (1 + s)) * (Math.PI / 2)) ** 2;
	return Math.min(Math.max(f(tau) / f(0), 1e-6), 1);
}

/**
 * The schedule diffusion started with: variance added at a rate rising
 * linearly from 1e-4 to 0.02 over a thousand steps. Written continuously,
 * that is the exponential of the integral of that rate. It reaches 4e-5 at
 * the top, and it gets there far too early — which is the plate's point.
 */
export function alphaBarLinear(tau: number): number {
	const b0 = 1e-4;
	const b1 = 0.02;
	const integral = b0 * tau + ((b1 - b0) * tau * tau) / 2;
	return Math.min(Math.max(Math.exp(-1000 * integral), 1e-6), 1);
}

/** Write the sinusoidal features for one noise level into `dst` at `off`. */
export function writeTimeFeatures(dst: Float32Array, off: number, tau: number): void {
	for (let k = 0; k < TIME_FEATURES / 2; k++) {
		const freq = Math.exp((k / (TIME_FEATURES / 2 - 1)) * Math.log(1000));
		dst[off + 2 * k] = Math.sin(tau * freq);
		dst[off + 2 * k + 1] = Math.cos(tau * freq);
	}
}

// ---------------------------------------------------------------- params ---

export function initParams(c: DiffusionConfig, seed: number): Arr {
	const n = 8 + c.layers * 8;
	const keys = random.split(random.key(seed), n);
	let ki = 0;
	const nk = () => {
		ki++;
		return ki < n ? keys.ref.slice(ki - 1) : keys.slice(ki - 1);
	};
	const gauss = (shape: number[], scale: number) => random.normal(nk(), shape).mul(scale);
	const s = Math.sqrt(3 / c.dim);
	const uni = (shape: number[], k: number) =>
		random.uniform(nk(), shape, { minval: -k * s, maxval: k * s });
	const { dim: d, patch, channels: ch } = c;

	const params: Arr = {
		condIn: gauss([condWidth(c), d], 0.05),
		condOut: gauss([d, d], 0.05),
		patchify: gauss([d, ch, patch, patch], 0.05),
		pos: gauss([tokenCount(c), d], 0.02),
		blocks: [] as Arr[],
		headFilm: gauss([d, 2 * d], 0.02),
		// Small, not zero: a zero output projection blocks every gradient into
		// the block interior at step 0, which the language worker learned the
		// hard way and wrote down.
		unpatch: gauss([ch, d, patch, patch], 0.01)
	};
	for (let i = 0; i < c.layers; i++) {
		params.blocks.push({
			film: gauss([d, 4 * d], 0.02),
			wq: uni([d, d], 1),
			wk: uni([d, d], 1),
			wv: uni([d, d], 1),
			wo: uni([d, d], 0.2),
			fc1: uni([d, 4 * d], 0.4),
			fc2: uni([4 * d, d], 0.2)
		});
	}
	return params;
}

export function disposeTree(t: Arr): void {
	tree.dispose(t);
}

export function countLive(params: Arr): number {
	const leaves = tree.leaves(tree.ref(params)) as Arr[];
	let total = 0;
	for (const l of leaves) {
		total += l.size;
		l.dispose();
	}
	return total;
}

// --------------------------------------------------------------- forward ---

function rmsnorm(x: Arr) {
	const ms = np.mean(np.square(x.ref), -1, { keepdims: true });
	return x.div(np.sqrt(ms.add(1e-5)));
}

/**
 * The whole network. `x` is [B, C, R, R]; `cond` is [B, condWidth]; the result
 * has x's shape and means either the noise or the velocity, per `objective`.
 *
 * Every projection is a 2-D matmul on the flattened [B·tokens, dim] stream:
 * jax-js's backward pass for a batched dot materializes an intermediate the
 * size of the product, so keeping the rank at two is the difference between
 * training and running out of memory.
 */
export function forward(params: Arr, c: DiffusionConfig, B: number, x: Arr, cond: Arr): Arr {
	const { dim: d, heads, patch, res } = c;
	const tok = tokenCount(c);
	const headDim = d / heads;
	const side = res / patch;

	/** adaLN: the condition supplies a scale and shift per feature. */
	const modulate = (v: Arr, sb: Arr, off: number) => {
		const scale = sb.ref.slice([], [off, off + d]).reshape([B, 1, d]);
		const shift = sb.slice([], [off + d, off + 2 * d]).reshape([B, 1, d]);
		return v
			.reshape([B, tok, d])
			.mul(scale.add(1))
			.add(shift)
			.reshape([B * tok, d]);
	};

	const cvec = nn.silu(np.dot(nn.silu(np.dot(cond, params.condIn)), params.condOut));

	// A stride-`patch` convolution is exactly "cut into tiles, embed each tile".
	let h = lax.conv(x, params.patchify, [patch, patch], 'VALID');
	h = np.transpose(h.reshape([B, d, tok]), [0, 2, 1]).reshape([B * tok, d]);
	h = h.add(np.tile(params.pos, [B, 1]));

	for (const layer of params.blocks) {
		const sb = np.dot(cvec.ref, layer.film);

		const res1 = h.ref;
		const a = modulate(rmsnorm(h), sb.ref, 0);
		const q = np.dot(a.ref, layer.wq).reshape([B, tok, heads, headDim]);
		const k = np.dot(a.ref, layer.wk).reshape([B, tok, heads, headDim]);
		const v = np.dot(a, layer.wv).reshape([B, tok, heads, headDim]);
		// No causal mask: a patch may look at every other patch, which is the
		// one structural difference from the language model in chapter 5.
		const o = nn.dotProductAttention(q, k, v);
		h = np.dot(o.reshape([B * tok, d]), layer.wo).add(res1);

		const res2 = h.ref;
		let f = modulate(rmsnorm(h), sb, 2 * d);
		f = nn.gelu(np.dot(f, layer.fc1));
		h = np.dot(f, layer.fc2).add(res2);
	}

	h = modulate(rmsnorm(h), np.dot(cvec, params.headFilm), 0);
	h = np.transpose(h.reshape([B, tok, d]), [0, 2, 1]).reshape([B, d, side, side]);
	return lax.convTranspose(h, params.unpatch, [patch, patch], 'VALID');
}

/** Mean squared error against whatever the objective says the target is. */
export function lossFn(
	params: Arr,
	c: DiffusionConfig,
	B: number,
	x: Arr,
	cond: Arr,
	target: Arr
): Arr {
	return np.mean(np.square(forward(params, c, B, x, cond).sub(target)));
}

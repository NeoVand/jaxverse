/**
 * Experimental temporal-transformer variant. Not shipped, not a LeWM reproduction.
 * Keeps the current MLP encoder, two-frame/action alignment, residual prediction,
 * and exact joint objective. Only the dynamics predictor changes.
 */
import { numpy as np, nn, tree } from '@jax-js/jax';

/* eslint-disable @typescript-eslint/no-explicit-any */
export type Tensor = any;

export interface WorldConfig {
	resolution: number;
	hidden: number;
	latent: number;
	predictorHidden: number;
	projections: number;
	regularization: number;
}

export const WORLD_CONFIG: WorldConfig = {
	resolution: 32,
	hidden: 128,
	latent: 8,
	predictorHidden: 128,
	projections: 32,
	regularization: 0.01
};

export function seededRandom(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

export function normalRandom(rand: () => number): number {
	return Math.sqrt(-2 * Math.log(Math.max(1e-9, rand()))) * Math.cos(2 * Math.PI * rand());
}

export const TRANSFORMER_SHAPE = {
	width: 32,
	layers: 2,
	heads: 2,
	feedForward: 64
} as const;

export function parameterCount(c: WorldConfig): number {
	const { width: w, layers, feedForward: f } = TRANSFORMER_SHAPE;
	const d = c.latent;
	const encoder = (c.resolution ** 2 + 1) * c.hidden + (c.hidden + 1) * d;
	const input = (d + 1) * w;
	const actions = 3 * w;
	const positions = 2 * w;
	const block = 4 * (w + 1) * w + (w + 1) * f + (f + 1) * w + (w + 1) * 6 * w + 4 * w;
	const finalNorm = 2 * w;
	const output = (w + 1) * d;
	return encoder + input + actions + positions + layers * block + finalNorm + output;
}

export function initWorldParams(c: WorldConfig, seed: number): Tensor {
	const rand = seededRandom(seed);
	const layer = (input: number, output: number, scale = 1) => {
		const data = new Float32Array(input * output);
		for (let i = 0; i < data.length; i++)
			data[i] = normalRandom(rand) * Math.sqrt(2 / input) * scale;
		return {
			w: np.array(data).reshape([input, output]),
			b: np.zeros([output])
		};
	};
	// Initialize the encoder first, in exactly the current MLP model's order.
	const encoder = [layer(c.resolution ** 2, c.hidden), layer(c.hidden, c.latent, 0.7)];
	const { width: w, layers, feedForward: f } = TRANSFORMER_SHAPE;
	const norm = () => ({ scale: np.ones([w]), shift: np.zeros([w]) });
	const input = layer(c.latent, w);
	const action = layer(2, w);
	const position = np
		.array(Float32Array.from({ length: 2 * w }, () => normalRandom(rand) * 0.02))
		.reshape([2, w]);
	const blocks = Array.from({ length: layers }, () => ({
		norm1: norm(),
		norm2: norm(),
		q: layer(w, w),
		k: layer(w, w),
		v: layer(w, w),
		out: layer(w, w, 0.2),
		fc1: layer(w, f),
		fc2: layer(f, w, 0.2),
		// As in official LeWM, zero AdaLN modulation/gates begin each block as identity.
		ada: { w: np.zeros([w, 6 * w]), b: np.zeros([6 * w]) }
	}));
	return {
		encoder,
		predictor: {
			input,
			action,
			position,
			blocks,
			finalNorm: norm(),
			output: layer(w, c.latent, 0.15)
		}
	};
}

/** Consumes encoder and pixels [B, R²]. The sensor uses white=1, dark=0. */
export function encode(encoder: Tensor, pixels: Tensor): Tensor {
	// Fixed pixel scaling is not privileged state preprocessing. It makes the
	// small dark mechanism occupy a useful numerical range on white paper.
	let h = pixels.neg().add(1).mul(4);
	h = nn.gelu(np.dot(h, encoder[0].w).add(encoder[0].b));
	return np.dot(h, encoder[1].w).add(encoder[1].b);
}

/** Standard per-token LayerNorm. Its output is internal, never the SIGReg embedding. */
function layerNorm(x: Tensor, params: Tensor): Tensor {
	const centered = x.ref.sub(np.mean(x, -1, { keepdims: true }));
	const variance = np.mean(np.square(centered.ref), -1, { keepdims: true });
	return centered
		.div(np.sqrt(variance.add(1e-6)))
		.mul(params.scale)
		.add(params.shift);
}

/**
 * Two tokens [z[t-1], z[t]], each conditioned on its outgoing action.
 * a[t-1] caused the observed transition, a[t] proposes the next torque.
 * Temporal causal attention; output at t predicts a residual added to z[t].
 * predictorHidden is ignored by this isolated fixed-width experimental variant.
 */
export function predict(
	predictor: Tensor,
	previous: Tensor,
	current: Tensor,
	actions: Tensor
): Tensor {
	const { width: w, heads } = TRANSFORMER_SHAPE;
	const batch = current.shape[0];
	const latent = current.shape[1];
	const tokens = np.stack([previous, current.ref], 1).reshape([batch * 2, latent]);
	let h = np.dot(tokens, predictor.input.w).add(predictor.input.b);
	h = h.add(np.tile(predictor.position, [batch, 1]));
	const conditions = np
		.dot(actions.reshape([batch * 2, 2]), predictor.action.w)
		.add(predictor.action.b);
	for (const block of predictor.blocks) {
		const modulation = np.dot(nn.silu(conditions.ref), block.ada.w).add(block.ada.b);
		const parts = Array.from({ length: 6 }, (_, i) =>
			(i === 5 ? modulation : modulation.ref).slice([], [i * w, (i + 1) * w])
		);
		const [shift1, scale1, gate1, shift2, scale2, gate2] = parts;
		const residual1 = h.ref;
		const a = layerNorm(h, block.norm1).mul(scale1.add(1)).add(shift1);
		const q = np
			.dot(a.ref, block.q.w)
			.add(block.q.b)
			.reshape([batch, 2, heads, w / heads]);
		const k = np
			.dot(a.ref, block.k.w)
			.add(block.k.b)
			.reshape([batch, 2, heads, w / heads]);
		const v = np
			.dot(a, block.v.w)
			.add(block.v.b)
			.reshape([batch, 2, heads, w / heads]);
		const attention = nn.dotProductAttention(q, k, v, { isCausal: true }).reshape([batch * 2, w]);
		h = np.dot(attention, block.out.w).add(block.out.b).mul(gate1).add(residual1);
		const residual2 = h.ref;
		const f = layerNorm(h, block.norm2).mul(scale2.add(1)).add(shift2);
		const ff = nn.gelu(np.dot(f, block.fc1.w).add(block.fc1.b));
		h = np.dot(ff, block.fc2.w).add(block.fc2.b).mul(gate2).add(residual2);
	}
	conditions.dispose();
	const last = layerNorm(h, predictor.finalNorm)
		.reshape([batch, 2, w])
		.slice([], [1, 2])
		.reshape([batch, w]);
	// This final affine projection leaves the eight-dimensional output unconstrained.
	return np.dot(last, predictor.output.w).add(predictor.output.b).add(current);
}

/** Fresh unit projection columns, shared across sequence positions in a batch. */
export function projectionDirections(
	c: WorldConfig,
	rand: () => number
): Float32Array<ArrayBuffer> {
	const a = new Float32Array(c.latent * c.projections);
	for (let m = 0; m < c.projections; m++) {
		let norm = 0;
		for (let d = 0; d < c.latent; d++) {
			const v = normalRandom(rand);
			a[d * c.projections + m] = v;
			norm += v * v;
		}
		for (let d = 0; d < c.latent; d++) a[d * c.projections + m] /= Math.sqrt(norm);
	}
	return a;
}

/**
 * Exact quadrature/reduction recipe from LeWM module.py (17 knots, [0,3]).
 * z [T,B,D], directions [D,M]. No centering, whitening, or norm constraint.
 * The expectation is across B independently at each T, then average T and M.
 */
export function sigreg(z: Tensor, directions: Tensor): Tensor {
	const [times, batch, dim] = z.shape;
	const projections = directions.shape[1];
	const knots = 17;
	const ts = new Float32Array(knots);
	const phi = new Float32Array(knots);
	const weights = new Float32Array(knots);
	for (let k = 0; k < knots; k++) {
		ts[k] = (3 * k) / (knots - 1);
		phi[k] = Math.exp((-ts[k] * ts[k]) / 2);
		weights[k] = (k === 0 || k === knots - 1 ? 1 : 2) * (3 / (knots - 1)) * phi[k];
	}
	const x = np
		.dot(z.reshape([times * batch, dim]), directions)
		.reshape([times, batch, projections, 1])
		.mul(np.array(ts));
	const real = np.mean(np.cos(x.ref), 1).sub(np.array(phi));
	const imag = np.mean(np.sin(x), 1);
	const err = np.square(real).add(np.square(imag));
	return np.mean(np.sum(err.mul(np.array(weights)), -1)).mul(batch);
}

/** Consumes inputs. Both source and target embeddings receive gradients. */
export function objective(
	params: Tensor,
	c: WorldConfig,
	pixels: Tensor,
	actions: Tensor,
	directions: Tensor
): [Tensor, Tensor] {
	const batch = actions.shape[0];
	const z = encode(params.encoder, pixels.reshape([3 * batch, c.resolution ** 2])).reshape([
		3,
		batch,
		c.latent
	]);
	const pred = predict(params.predictor, z.ref.slice(0), z.ref.slice(1), actions);
	const prediction = np.mean(np.square(pred.sub(z.ref.slice(2))));
	const regularizer = sigreg(z, directions);
	const total = prediction.ref.add(regularizer.ref.mul(c.regularization));
	return [total, np.stack([prediction, regularizer])];
}

export function disposeWorldParams(params: Tensor): void {
	tree.dispose(params);
}

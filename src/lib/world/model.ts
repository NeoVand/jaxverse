/** A small joint-embedding world model. No simulator or state labels enter this module. */
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

export function parameterCount(c: WorldConfig): number {
	const input = c.resolution ** 2;
	const h = c.hidden;
	const p = c.predictorHidden;
	const d = c.latent;
	return (input + 1) * h + (h + 1) * d + (2 * d + 5) * p + (p + 1) * p + (p + 1) * d;
}

export function initWorldParams(c: WorldConfig, seed: number): Tensor {
	const rand = seededRandom(seed);
	const layer = (input: number, output: number, scale = 1) => {
		const data = new Float32Array(input * output);
		for (let i = 0; i < data.length; i++)
			data[i] = normalRandom(rand) * Math.sqrt(2 / input) * scale;
		return { w: np.array(data).reshape([input, output]), b: np.zeros([output]) };
	};
	return {
		encoder: [layer(c.resolution ** 2, c.hidden), layer(c.hidden, c.latent, 0.7)],
		predictor: [
			layer(2 * c.latent + 4, c.predictorHidden),
			layer(c.predictorHidden, c.predictorHidden),
			layer(c.predictorHidden, c.latent, 0.15)
		]
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

/**
 * Consumes predictor and arrays. Inputs are z[t-1], z[t], a[t-1], a[t].
 * a[t-1] caused the observed transition; a[t] is the proposed next torque.
 * A residual parameterization starts near persistence; its correction is learned.
 */
export function predict(
	predictor: Tensor,
	previous: Tensor,
	current: Tensor,
	actions: Tensor
): Tensor {
	let h = np.concatenate([previous, current.ref, actions], 1);
	h = nn.gelu(np.dot(h, predictor[0].w).add(predictor[0].b));
	h = nn.gelu(np.dot(h, predictor[1].w).add(predictor[1].b));
	return np.dot(h, predictor[2].w).add(predictor[2].b).add(current);
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

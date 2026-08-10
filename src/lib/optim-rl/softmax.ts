// Softmax utilities shared by every tabular policy in the reward chapter.

import type { Rand } from './rng';

/** Numerically stable softmax. Pass `out` in hot loops to avoid allocation. */
export function softmax(logits: ArrayLike<number>, out?: Float64Array): Float64Array {
	const n = logits.length;
	const p = out ?? new Float64Array(n);
	let mx = -Infinity;
	for (let i = 0; i < n; i++) if (logits[i] > mx) mx = logits[i];
	let z = 0;
	for (let i = 0; i < n; i++) {
		p[i] = Math.exp(logits[i] - mx);
		z += p[i];
	}
	for (let i = 0; i < n; i++) p[i] /= z;
	return p;
}

/** Sample an index from a probability vector (assumed to sum to 1). */
export function sampleFrom(probs: ArrayLike<number>, rand: Rand): number {
	let u = rand();
	for (let i = 0; i < probs.length; i++) {
		u -= probs[i];
		if (u <= 0) return i;
	}
	return probs.length - 1; // float residue lands on the last arm
}

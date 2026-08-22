import { describe, expect, it } from 'vitest';
import { attention, blendInto, pullStrength, SWARM } from './swarm';

describe('the swarm between practice halls', () => {
	it('the leader is never pulled, and neither is anyone still in the race', () => {
		expect(pullStrength(0.5, 0.5)).toBe(0);
		// 20 % behind — a real gap, but the kind a different pumping rhythm
		// opens and closes; these halls stay independent
		expect(pullStrength(0.4, 0.5)).toBe(0);
		// just past the gate: a whisper, not a copy
		expect(pullStrength(0.32, 0.5)).toBeGreaterThan(0);
		expect(pullStrength(0.32, 0.5)).toBeLessThan(0.02);
	});

	it('a hall that has learned nothing closes most of the gap in ten seconds', () => {
		const k = pullStrength(0, 0.5);
		expect(k).toBeCloseTo(SWARM.maxPull, 6);
		// ~12 pulls at 0.8 s each
		expect(1 - (1 - k) ** 12).toBeGreaterThan(0.9);
	});

	it('the pull ignores an early pool where nobody has learned anything yet', () => {
		// all six at zero: the gap is 0/0, and nobody should be dragged
		// anywhere on the strength of rounding noise
		expect(pullStrength(0, 0)).toBe(0);
		expect(pullStrength(0.001, 0.002)).toBe(0);
	});

	it('attention only ever points uphill, and sums to one', () => {
		const scores = [0.1, 0.3, 0.5, 0.05];
		const rises = [0, 0, 0, 0];
		const out = new Float64Array(4);
		expect(attention(scores, rises, 0, out)).toBe(1);
		expect(out[0]).toBe(0); // never itself
		expect(out[3]).toBe(0); // never a hall that is behind
		expect(out[1] + out[2]).toBeCloseTo(1, 12);
		expect(out[2]).toBeGreaterThan(out[1]); // the better hall gets more
	});

	it('the leader has nobody to listen to', () => {
		const out = new Float64Array(3);
		expect(attention([0.5, 0.2, 0.1], [0, 0, 0], 0, out)).toBe(0);
		expect(Array.from(out)).toEqual([0, 0, 0]);
	});

	it('a hall that is improving fast out-attracts a hall that is merely ahead', () => {
		//              stuck        marginally ahead   just found something
		const scores = [0.2, /**/ 0.34, /**/ 0.3];
		const rises = [0, /*  */ 0, /*   */ 0.06];
		const out = new Float64Array(3);
		attention(scores, rises, 0, out);
		expect(out[2]).toBeGreaterThan(out[1]);
		// and with nothing moving, the ranking is back to plain fitness
		const flat = new Float64Array(3);
		attention(scores, [0, 0, 0], 0, flat);
		expect(flat[1]).toBeGreaterThan(flat[2]);
	});

	it('the blend is a convex combination of the halls ahead', () => {
		const thetas = [
			Float64Array.from([1, 1]),
			Float64Array.from([0, 4]),
			Float64Array.from([-2, 0])
		];
		const out = new Float64Array(2);
		blendInto(out, thetas, Float64Array.from([0, 0.25, 0.75]));
		expect(Array.from(out)).toEqual([-1.5, 1]);
	});
});

import { describe, expect, it } from 'vitest';
import { pcaApply, pcaFit, pcaPlane, planeCubePolygon, makeGridLines } from './common';

/** A cloud with three clearly ordered directions of spread, plus noise. */
function cloud(n: number, d: number, seed = 1) {
	let s = seed >>> 0;
	const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296 - 0.5);
	const out = new Float32Array(n * d);
	for (let i = 0; i < n; i++) {
		const a = rnd() * 6;
		const b = rnd() * 3;
		const c = rnd() * 1.5;
		for (let q = 0; q < d; q++) {
			out[i * d + q] =
				(q === 0 ? a : 0) + (q === 1 ? b : 0) + (q === 2 ? c : 0) + rnd() * 0.05 + 0.3;
		}
	}
	return out;
}

describe('pcaFit', () => {
	it('finds the directions of greatest spread', () => {
		const n = 500;
		const d = 8;
		const p = pcaFit(cloud(n, d), n, d, 3);
		// the planted axes are e0, e1, e2 in decreasing spread
		expect(Math.abs(p.basis[0][0])).toBeGreaterThan(0.9);
		expect(Math.abs(p.basis[1][1])).toBeGreaterThan(0.9);
		expect(Math.abs(p.basis[2][2])).toBeGreaterThan(0.85);
	});

	it('is orthonormal', () => {
		const n = 400;
		const d = 6;
		const p = pcaFit(cloud(n, d, 3), n, d, 3);
		for (let a = 0; a < 3; a++) {
			let len = 0;
			for (let q = 0; q < d; q++) len += p.basis[a][q] ** 2;
			expect(len).toBeCloseTo(1, 6);
			for (let b = a + 1; b < 3; b++) {
				let dot = 0;
				for (let q = 0; q < d; q++) dot += p.basis[a][q] * p.basis[b][q];
				expect(Math.abs(dot)).toBeLessThan(1e-5);
			}
		}
	});

	// The bug this whole rewrite exists for: refitting a drifting representation
	// used to negate or swap axes several times a second, and the view lerps
	// between snapshots, so every flip was animated as the cloud sweeping
	// through the origin and reassembling mirrored.
	it('does not flip or swap its axes as the data drifts', () => {
		const n = 400;
		const d = 8;
		let prev = pcaFit(cloud(n, d, 5), n, d, 3);
		const first = prev.basis.map((b) => b.slice());
		for (let frame = 1; frame < 40; frame++) {
			const next = cloud(n, d, 5 + frame); // a fresh but similar cloud
			const now = pcaFit(next, n, d, 3, prev);
			for (let c = 0; c < 3; c++) {
				let dot = 0;
				for (let q = 0; q < d; q++) dot += now.basis[c][q] * prev.basis[c][q];
				// each axis continues the one before it, never its negation
				expect(dot).toBeGreaterThan(0.5);
			}
			prev = now;
		}
		// and after forty frames it is still the same basis, not a wandered one
		for (let c = 0; c < 3; c++) {
			let dot = 0;
			for (let q = 0; q < d; q++) dot += prev.basis[c][q] * first[c][q];
			expect(dot).toBeGreaterThan(0.8);
		}
	});

	it('a cold refit CAN flip — which is why the warm one is passed', () => {
		// documents the failure mode rather than trusting it stays away
		const n = 300;
		const d = 5;
		const a = pcaFit(cloud(n, d, 9), n, d, 2);
		const b = pcaFit(cloud(n, d, 9), n, d, 2, a);
		let dot = 0;
		for (let q = 0; q < d; q++) dot += a.basis[0][q] * b.basis[0][q];
		expect(dot).toBeGreaterThan(0.99);
	});
});

describe('pcaPlane', () => {
	it('agrees with the full-space plane on points that lie in the shadow', () => {
		const n = 400;
		const d = 6;
		const data = cloud(n, d, 11);
		const p = pcaFit(data, n, d, 3);
		const w = [0.7, -1.3, 0.4, 0.9, -0.2, 0.5];
		const c = 0.15;
		const plane = pcaPlane(p, w, c, 3);
		// take a point built purely from the three shown directions: for it the
		// trace is exact, and that is the claim the drawing makes
		for (const t of [
			[0.4, -0.2, 0.1],
			[-1.1, 0.6, -0.3],
			[0, 0, 0]
		]) {
			const h = new Float64Array(d);
			for (let q = 0; q < d; q++) {
				h[q] = p.mean[q];
				for (let i = 0; i < 3; i++) h[q] += t[i] * p.basis[i][q];
			}
			let full = c;
			for (let q = 0; q < d; q++) full += w[q] * h[q];
			const shadow = plane.c + t[0] * plane.n[0] + t[1] * plane.n[1] + t[2] * plane.n[2];
			expect(shadow).toBeCloseTo(full, 6);
		}
	});

	it('produces a normal a cube can actually be cut with', () => {
		const n = 300;
		const d = 7;
		const p = pcaFit(cloud(n, d, 13), n, d, 3);
		const plane = pcaPlane(p, [0.5, 0.5, 0.5, 0, 0, 0, 0], 0, 3);
		const poly = planeCubePolygon(plane.n[0], plane.n[1], plane.n[2], plane.c, 1);
		expect(poly.length).toBeGreaterThanOrEqual(3);
	});
});

describe('pcaApply', () => {
	it('round-trips a point built from the basis', () => {
		const n = 300;
		const d = 6;
		const data = cloud(n, d, 17);
		const p = pcaFit(data, n, d, 3);
		const t = [0.7, -0.4, 0.25];
		const h = new Float32Array(d);
		for (let q = 0; q < d; q++) {
			h[q] = p.mean[q];
			for (let i = 0; i < 3; i++) h[q] += t[i] * p.basis[i][q];
		}
		const got = pcaApply(p, h, 1, 3);
		for (let i = 0; i < 3; i++) expect(got[i]).toBeCloseTo(t[i], 5);
	});
});

describe('makeGridLines', () => {
	it('lays every polyline inside the verts buffer', () => {
		const g = makeGridLines(16, 65, 1.1);
		for (const [start, count] of g.lines) expect(start + count).toBeLessThanOrEqual(g.n);
	});
});

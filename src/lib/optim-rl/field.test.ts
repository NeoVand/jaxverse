import { describe, expect, it } from 'vitest';
import {
	CATCH_WEIGHT,
	RIG_REACH,
	camera,
	fieldBox,
	fitness,
	fieldBox as makeBox,
	isoLine,
	niceStep,
	isoWindow,
	separate
} from './field';

const BOX = { left: 40, right: 600, top: 30, bottom: 300 };
const VIEW = [0, 1, 0, 1];
/** A window big enough that these tests are about the fan, not the frame. */
const WIDE: [number, number] = [-1e4, 1e4];

/** Recover the fitness of a screen point, so the invariant can be checked
 * on what is actually drawn rather than on what was intended. */
function fitnessOf(cam: ReturnType<typeof camera>, view: number[], sx: number, sy: number) {
	const x = view[0] + ((sx - BOX.left) / (BOX.right - BOX.left)) * (view[1] - view[0]);
	const y = view[2] + ((BOX.bottom - sy) / (BOX.bottom - BOX.top)) * (view[3] - view[2]);
	return fitness(x, y);
}

describe('separate', () => {
	it('preserves every hall’s fitness exactly — the whole point', () => {
		const cam = camera(VIEW, BOX);
		// A converged pool: six halls within a percent of each other, which is
		// where the old repulsion started inventing an order.
		const data = [
			[0.98, 0.55],
			[0.97, 0.56],
			[0.98, 0.54],
			[0.99, 0.55],
			[0.97, 0.55],
			[0.98, 0.56]
		];
		const n = data.length;
		const px = Float64Array.from(data, (d) => cam.x(d[0]));
		const py = Float64Array.from(data, (d) => cam.y(d[1]));
		const before = data.map((d) => fitness(d[0], d[1]));
		separate(px, py, n, 60, cam.iso, WIDE);
		for (let i = 0; i < n; i++)
			expect(fitnessOf(cam, VIEW, px[i], py[i])).toBeCloseTo(before[i], 9);
	});

	it('actually separates them', () => {
		const cam = camera(VIEW, BOX);
		const n = 6;
		const px = Float64Array.from({ length: n }, () => cam.x(0.98));
		const py = Float64Array.from({ length: n }, () => cam.y(0.55));
		const used = separate(px, py, n, 40, cam.iso, WIDE);
		expect(used).toBeCloseTo(40, 6);
		for (let a = 0; a < n; a++)
			for (let b = a + 1; b < n; b++)
				expect(Math.hypot(px[a] - px[b], py[a] - py[b])).toBeGreaterThan(39);
	});

	it('never orders two halls against their fitness', () => {
		const cam = camera(VIEW, BOX);
		// deliberately adversarial: the better hall starts on the right of the
		// canvas but is crowded by three worse ones
		const data = [
			[0.6, 0.9],
			[0.61, 0.2],
			[0.6, 0.21],
			[0.59, 0.2]
		];
		const n = data.length;
		const px = Float64Array.from(data, (d) => cam.x(d[0]));
		const py = Float64Array.from(data, (d) => cam.y(d[1]));
		separate(px, py, n, 55, cam.iso, WIDE);
		const drawn = Array.from({ length: n }, (_, i) => fitnessOf(cam, VIEW, px[i], py[i]));
		const want = data.map((d) => fitness(d[0], d[1]));
		const byDrawn = [...drawn.keys()].sort((a, b) => drawn[a] - drawn[b]);
		const byWant = [...want.keys()].sort((a, b) => want[a] - want[b]);
		expect(byDrawn).toEqual(byWant);
	});

	it('shrinks the fan rather than overflowing the room it has', () => {
		const cam = camera(VIEW, BOX);
		const n = 20;
		const px = Float64Array.from({ length: n }, () => cam.x(0.5));
		const py = Float64Array.from({ length: n }, () => cam.y(0.5));
		const t0 = cam.x(0.5) * cam.iso[0] + cam.y(0.5) * cam.iso[1];
		const used = separate(px, py, n, 60, cam.iso, [t0 - 95, t0 + 95]);
		expect(used).toBeCloseTo(10, 6);
		let lo = Infinity;
		let hi = -Infinity;
		for (let i = 0; i < n; i++) {
			const t = px[i] * cam.iso[0] + py[i] * cam.iso[1];
			lo = Math.min(lo, t);
			hi = Math.max(hi, t);
		}
		expect(hi - lo).toBeLessThanOrEqual(190 + 1e-6);
	});

	it('leaves a single hall exactly where it was', () => {
		const cam = camera(VIEW, BOX);
		const px = Float64Array.from([cam.x(0.4)]);
		const py = Float64Array.from([cam.y(0.7)]);
		separate(px, py, 1, 40, cam.iso, WIDE);
		expect(px[0]).toBeCloseTo(cam.x(0.4), 9);
		expect(py[0]).toBeCloseTo(cam.y(0.7), 9);
	});
});

describe('isoLine', () => {
	it('returns a segment every point of which has the given fitness', () => {
		const seg = isoLine(0.8, [0, 1, 0, 1]);
		expect(seg).not.toBeNull();
		const [ax, ay, bx, by] = seg!;
		expect(fitness(ax, ay)).toBeCloseTo(0.8, 9);
		expect(fitness(bx, by)).toBeCloseTo(0.8, 9);
	});

	it('reports nothing for a fitness the frame cannot contain', () => {
		expect(isoLine(9, [0, 1, 0, 1])).toBeNull();
		expect(isoLine(-3, [0, 1, 0, 1])).toBeNull();
	});

	it('agrees with the worker’s own score weighting', () => {
		expect(CATCH_WEIGHT).toBe(0.5);
		expect(fitness(1, 0)).toBeCloseTo(fitness(0, 2), 9);
	});
});

describe('niceStep', () => {
	it('picks round numbers', () => {
		expect(niceStep(0.11)).toBeCloseTo(0.1, 9);
		expect(niceStep(0.23)).toBeCloseTo(0.2, 9);
		expect(niceStep(0.6)).toBeCloseTo(0.5, 9);
		expect(niceStep(9)).toBeCloseTo(10, 9);
	});
});

describe('fieldBox', () => {
	// The plate draws each hall as a whole pendulum, and a pendulum is much
	// wider than the dot it replaced. Halls at the corners of the field used
	// to hang over the edge of the plate; these are the two cases that has to
	// keep working.
	const cases = [
		{ w: 894, h: 448, slot: 30 },
		{ w: 420, h: 340, slot: 12 },
		{ w: 1200, h: 520, slot: 40 }
	];
	it('leaves room for a whole rig at every corner', () => {
		for (const { w, h, slot } of cases) {
			const box = fieldBox(w, h, slot);
			const cam = camera([0, 1, 0, 1], box);
			const reach = slot * RIG_REACH;
			for (const [d, c] of [
				[0, 0],
				[1, 0],
				[0, 1],
				[1, 1]
			]) {
				const x = cam.x(d);
				const y = cam.y(c);
				expect(x - reach).toBeGreaterThanOrEqual(0);
				expect(x + reach).toBeLessThanOrEqual(w);
				expect(y - reach).toBeGreaterThanOrEqual(0);
				expect(y + reach).toBeLessThanOrEqual(h);
			}
		}
	});

	it('still leaves a usable field once the rigs are accounted for', () => {
		for (const { w, h, slot } of cases) {
			const box = makeBox(w, h, slot);
			expect(box.right - box.left).toBeGreaterThan(60);
			expect(box.bottom - box.top).toBeGreaterThan(40);
		}
	});

	it('keeps the whole fan inside the box, however crowded', () => {
		const { w, h, slot } = { w: 894, h: 448, slot: 14 };
		const box = fieldBox(w, h, slot);
		const cam = camera([0, 1, 0, 1], box);
		const n = 28;
		// every hall identical, which is the worst case for the fan
		const px = Float64Array.from({ length: n }, () => cam.x(0.95));
		const py = Float64Array.from({ length: n }, () => cam.y(0.5));
		separate(px, py, n, 2 * slot + 6, cam.iso, isoWindow(cam.x(0.95), cam.y(0.5), cam.iso, box));
		const reach = slot * RIG_REACH;
		for (let i = 0; i < n; i++) {
			expect(px[i]).toBeGreaterThan(-1);
			expect(px[i]).toBeLessThan(w + 1);
			expect(py[i]).toBeGreaterThan(-1);
			expect(py[i]).toBeLessThan(h + 1);
		}
		void reach;
	});
});

describe('isoWindow', () => {
	it('gives a pool in the corner the side it still has', () => {
		const box = fieldBox(894, 448, 24);
		const cam = camera([0, 1, 0, 1], box);
		// every hall at the origin, which is where every run begins
		const n = 6;
		const px = Float64Array.from({ length: n }, () => cam.x(0));
		const py = Float64Array.from({ length: n }, () => cam.y(0));
		const win = isoWindow(cam.x(0), cam.y(0), cam.iso, box, box.slack);
		const used = separate(px, py, n, 54, cam.iso, win);
		expect(used).toBeGreaterThan(0);
		for (let a = 0; a < n; a++)
			for (let b = a + 1; b < n; b++)
				expect(Math.hypot(px[a] - px[b], py[a] - py[b])).toBeGreaterThan(used * 0.99);
		// still on the canvas: the slack is a fraction of the margin the rigs
		// are drawn in, not a licence to leave the plate
		const reach = 24 * RIG_REACH;
		for (let i = 0; i < n; i++) {
			expect(px[i] - reach).toBeGreaterThan(-1);
			expect(px[i] + reach).toBeLessThan(895);
			expect(py[i] - reach).toBeGreaterThan(-1);
			expect(py[i] + reach).toBeLessThan(449);
		}
	});
});

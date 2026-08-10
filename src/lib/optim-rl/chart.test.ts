import { describe, expect, it } from 'vitest';
import { mulberry32 } from './rng';
import {
	angleBetween,
	bestPassage,
	boatSpeed,
	cellIndex,
	createBaseline,
	createTheta,
	greedyRoute,
	HEADING_NAME,
	headingAngle,
	inIrons,
	makeSea,
	NO_GO,
	N_HEADINGS,
	runPassage,
	trainStep,
	waterCells,
	stepSea,
	stepTime,
	twaOf,
	type Sea
} from './chart';

const deg = (d: number) => (d * Math.PI) / 180;

describe('the polar diagram', () => {
	it('cannot sail inside the no-go zone', () => {
		expect(boatSpeed(0)).toBe(0);
		expect(boatSpeed(NO_GO - 1e-6)).toBe(0);
		expect(boatSpeed(deg(20))).toBe(0);
	});

	it('is quickest on a beam reach, and eases off downwind', () => {
		const beam = boatSpeed(deg(90));
		expect(beam).toBeCloseTo(1, 6);
		expect(boatSpeed(deg(50))).toBeLessThan(beam);
		expect(boatSpeed(deg(180))).toBeLessThan(beam);
		// running is slower than reaching but far from stopped
		expect(boatSpeed(deg(180))).toBeGreaterThan(0.6);
	});

	it('has no cliff at the edge of the no-go zone', () => {
		expect(boatSpeed(NO_GO + 1e-4)).toBeLessThan(0.01);
	});

	it('measures the smallest angle between bearings', () => {
		expect(angleBetween(deg(350), deg(10))).toBeCloseTo(deg(20), 9);
		expect(angleBetween(deg(0), deg(180))).toBeCloseTo(Math.PI, 9);
	});
});

describe('one leg', () => {
	it('luffs head to wind and goes nowhere, and the clock still runs', () => {
		const sea = makeSea();
		expect(inIrons(sea, 0)).toBe(true); // due north, wind from the north
		const r = stepSea(sea, sea.start, 0);
		expect(r.s).toBe(sea.start);
		expect(r.irons).toBe(true);
		expect(r.r).toBeLessThan(0);
	});

	it('charges more time close-hauled than on a reach', () => {
		const sea = makeSea();
		// heading 1 is NE, 45° off a northerly — close-hauled; 2 is E, a beam reach
		expect(twaOf(sea, 1)).toBeCloseTo(deg(45), 9);
		expect(twaOf(sea, 2)).toBeCloseTo(deg(90), 9);
		expect(stepTime(sea, 1)).toBeLessThan(stepTime(sea, 2));
	});

	it('is stopped by land and by the edge of the chart, and pays for it', () => {
		const sea = makeSea({ land: new Set([cellIndex(3, 3)]) });
		const from = cellIndex(3, 2);
		const r = stepSea(sea, from, 0); // north, into the land — but north is irons
		expect(r.s).toBe(from);
		const sea2 = makeSea({ windFrom: Math.PI, land: new Set([cellIndex(3, 3)]) });
		const r2 = stepSea(sea2, from, 0); // wind from the south now, so north sails
		expect(r2.s).toBe(from);
		expect(r2.irons).toBe(false);
		expect(r2.r).toBeLessThan(0);
	});
});

describe('the best route', () => {
	it('finds a way to a harbour dead upwind, and it is not a straight line', () => {
		const sea = upwindSea();
		const { reward, path } = bestPassage(sea);
		expect(Number.isFinite(reward)).toBe(true);
		// dead upwind is 9 cells; no legal route can do it in 9 legs
		expect(path.length).toBeGreaterThan(10);
	});

	it('reports NaN when the harbour is walled off', () => {
		const sea = makeSea({ windFrom: Math.PI });
		for (let x = 0; x < sea.w; x++) sea.land.add(cellIndex(x, 8));
		expect(Number.isNaN(bestPassage(sea).reward)).toBe(true);
	});
});

/** Open water, wind from the north, harbour dead upwind of the start. */
function upwindSea(): Sea {
	return makeSea({
		start: cellIndex(7, 0),
		harbour: cellIndex(7, 9),
		land: new Set<number>(),
		shoals: new Set<number>(),
		windFrom: 0
	});
}

describe('what REINFORCE discovers', () => {
	/** The plate's own training loop, so tests and demo cannot drift apart. */
	function train(sea: Sea, episodes = 9000, lr = 0.09, seed = 4242) {
		const theta = createTheta(sea);
		const baseline = createBaseline(sea);
		const rand = mulberry32(seed);
		const water = waterCells(sea);
		let arrivals = 0;
		for (let e = 0; e < episodes; e++) {
			if (trainStep(sea, theta, baseline, rand, lr, water).end === 'harbour') arrivals++;
		}
		return { theta, arrivals };
	}

	it('learns to reach a harbour it cannot sail straight at', () => {
		const sea = upwindSea();
		const { theta } = train(sea);
		const rand = mulberry32(9);
		let arrived = 0;
		for (let k = 0; k < 200; k++) if (runPassage(sea, theta, rand).end === 'harbour') arrived++;
		expect(arrived / 200).toBeGreaterThan(0.9);
	});

	it('stops trying to sail into the wind', () => {
		const sea = upwindSea();
		const { theta } = train(sea);
		const rand = mulberry32(11);
		let irons = 0;
		let steps = 0;
		for (let k = 0; k < 100; k++) {
			const p = runPassage(sea, theta, rand);
			irons += p.ironsSteps;
			steps += p.steps;
		}
		expect(irons / steps).toBeLessThan(0.05);
	});

	/**
	 * The one that matters. Nothing in the reward mentions zigzagging; it only
	 * counts time and forbids pointing too high. If the policy comes back
	 * alternating between the two close-hauled headings, it worked out tacking
	 * on its own — which is the entire reason this world replaced a gridworld.
	 */
	it('discovers tacking: it alternates between the two close-hauled headings', () => {
		const sea = upwindSea();
		const { theta } = train(sea);
		const route = greedyRoute(sea, theta);

		expect(route.length).toBeGreaterThan(9);
		// heading 1 is NE, heading 7 is NW: the only two that make ground north
		const nePort = route.filter((a) => a === 1).length;
		const nwStar = route.filter((a) => a === 7).length;
		expect(nePort).toBeGreaterThan(0);
		expect(nwStar).toBeGreaterThan(0);

		// and it genuinely alternates rather than sailing one long slant
		let switches = 0;
		let last = -1;
		for (const a of route) {
			if (a !== 1 && a !== 7) continue;
			if (last !== -1 && a !== last) switches++;
			last = a;
		}
		expect(switches).toBeGreaterThanOrEqual(2);

		// every leg it chose is one it could actually sail
		for (const a of route) expect(inIrons(sea, a)).toBe(false);
	});

	it('sails a different shape when the wind backs', () => {
		const north = upwindSea();
		const west = { ...upwindSea(), windFrom: headingAngle(6) }; // wind from the west
		const a = greedyRoute(north, train(north).theta);
		const b = greedyRoute(west, train(west).theta);
		// the same destination, a different wind, and a different set of headings
		const setOf = (r: number[]) => new Set(r);
		expect([...setOf(a)].sort().join()).not.toBe([...setOf(b)].sort().join());
		for (const h of b) expect(twaOf(west, h)).toBeGreaterThan(NO_GO);
	});

	it('leaves a policy that is a probability distribution everywhere', () => {
		const sea = upwindSea();
		const { theta } = train(sea, 2000);
		for (let s = 0; s < sea.w * sea.h; s++) {
			let sum = 0;
			const slice = theta.subarray(s * N_HEADINGS, (s + 1) * N_HEADINGS);
			let mx = -Infinity;
			for (const v of slice) mx = Math.max(mx, v);
			for (const v of slice) sum += Math.exp(v - mx);
			expect(Number.isFinite(sum)).toBe(true);
			expect(sum).toBeGreaterThan(0);
		}
	});
	/**
	 * The robustness the reader will lean on, because spinning the compass rose
	 * is the best thing in the plate and it must not hand them a boat that has
	 * given up. Every wind, several seeds — and the reason this passes at all is
	 * the scatter: without it, the winds either side of the harbour's bearing
	 * fail outright, and which ones fail depends on the seed.
	 */
	it('arrives under every wind, on every seed', () => {
		for (const seed of [4242, 7, 2049]) {
			for (let point = 0; point < N_HEADINGS; point++) {
				const sea = makeSea({ windFrom: headingAngle(point) });
				const { theta } = train(sea, 14000, 0.09, seed);
				const rand = mulberry32(5);
				let arrived = 0;
				for (let k = 0; k < 60; k++) if (runPassage(sea, theta, rand).end === 'harbour') arrived++;
				expect(arrived / 60, `wind from ${HEADING_NAME[point]}, seed ${seed}`).toBeGreaterThan(
					0.85
				);
			}
		}
	});

	it('without the scatter, some winds strand the learner', () => {
		// Not a nicety being documented — the failure this guards against is why
		// the scatter exists, and if it ever stops happening the constant can go.
		const sea = makeSea({ windFrom: headingAngle(7) });
		const theta = createTheta(sea);
		const baseline = createBaseline(sea);
		const rand = mulberry32(4242);
		const water = waterCells(sea);
		for (let e = 0; e < 14000; e++)
			trainStep(sea, theta, baseline, rand, 0.09, water, { scatter: 0 });
		const rr = mulberry32(5);
		let arrived = 0;
		for (let k = 0; k < 60; k++) if (runPassage(sea, theta, rr).end === 'harbour') arrived++;
		expect(arrived / 60).toBeLessThan(0.5);
	});
});

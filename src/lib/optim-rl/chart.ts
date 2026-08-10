// A sea chart, a wind, and a boat that cannot sail straight at it.
//
// The replacement for a gridworld, and the reason for the replacement: in a
// gridworld the optimal policy is the shortest path, which is the answer you
// already knew before the learner started. Here the shortest path is often
// *impossible*, and the policy has to discover something no one told it —
// that to make ground upwind you must sail at an angle, alternately, in a
// zigzag. Nothing in the reward says "zigzag". It falls out of a rule about
// what a boat can do and a cost that counts time.
//
// The machinery underneath is exactly the gridworld's: a table of logits, a
// softmax per cell, REINFORCE with return-to-go and a per-state baseline.
// Plain loops on plain arrays, hundreds of episodes a second, no GPU. What
// changed is the world, not the learner.

import type { Rand } from './rng';
import { sampleFrom, softmax } from './softmax';

export const CHART_W = 14;
export const CHART_H = 10;
/** Eight compass headings, clockwise from north. */
export const N_HEADINGS = 8;

/** Screen-space steps per heading: index 0 is north (y grows upward). */
export const HEADING_DX = [0, 1, 1, 1, 0, -1, -1, -1] as const;
export const HEADING_DY = [1, 1, 0, -1, -1, -1, 0, 1] as const;
export const HEADING_NAME = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'] as const;

/** The compass angle of heading `a`, in radians clockwise from north. */
export function headingAngle(a: number): number {
	return (a * Math.PI) / 4;
}

/**
 * How close to the wind this boat can point, in radians. Real yachts manage
 * about 40°; 35° here so that a 45° heading — which is all an eight-point
 * compass can offer — is genuinely sailable rather than sitting exactly on
 * the boundary. Below this angle the sails luff and the boat stops.
 */
export const NO_GO = (35 * Math.PI) / 180;

/**
 * The slowest a boat is allowed to be while still moving, as a fraction of its
 * best speed. Without a floor the cost of sailing one degree off the no-go
 * limit runs to infinity and a single unlucky episode swamps every gradient
 * that follows it.
 */
const MIN_SPEED = 0.45;

/** Smallest signed angle between two compass bearings, in [0, π]. */
export function angleBetween(a: number, b: number): number {
	let d = Math.abs(a - b) % (Math.PI * 2);
	if (d > Math.PI) d = Math.PI * 2 - d;
	return d;
}

/**
 * The polar diagram, which is the whole physics of this chapter's world: how
 * fast the boat goes at a given angle to the wind, as a fraction of its best.
 *
 * Zero inside the no-go zone. Rising to a maximum on a beam reach, wind across
 * the deck at ninety degrees — which is genuinely a sailboat's quickest point
 * of sail, and the reason a course made of right angles can beat a straight
 * line. Easing off again running downwind, where the sails can only be pushed.
 */
export function boatSpeed(twa: number): number {
	if (twa <= NO_GO) return 0;
	if (twa < Math.PI / 2) {
		const u = (twa - NO_GO) / (Math.PI / 2 - NO_GO);
		return u * u * (3 - 2 * u); // smoothstep, so there is no cliff at the edge
	}
	const v = (twa - Math.PI / 2) / (Math.PI / 2);
	return 1 - 0.35 * v;
}

export interface Sea {
	w: number;
	h: number;
	/** Mutable on purpose — the reader drags these around under a live learner. */
	start: number;
	harbour: number;
	/** Cells you cannot enter: land. */
	land: Set<number>;
	/** Cells that end the passage badly. */
	shoals: Set<number>;
	/**
	 * The direction the wind blows FROM, in radians clockwise from north — the
	 * nautical convention, and the one the compass rose in the plate reads.
	 * Mutable: spinning it is the best thing in the chapter.
	 */
	windFrom: number;
	/** Time charged for one cell at best speed; everything else scales off it. */
	timeCost: number;
	harbourReward: number;
	shoalReward: number;
	/** Extra time lost when the sails luff and the boat does not move at all. */
	ironsCost: number;
	gamma: number;
	maxSteps: number;
}

export function cellIndex(x: number, y: number, w = CHART_W): number {
	return y * w + x;
}

/**
 * The chapter's water: an islet athwart the direct route, two shoals well off
 * the laylines, and the harbour set nearly dead upwind of the start so that
 * the only way there is the one nobody mentioned.
 *
 * The layout is not decoration, and it was chosen by measurement rather than
 * by eye. A wind makes a uniform policy drift *downwind*, so a harbour to
 * windward is genuinely hard to stumble into — and hazards sitting on the
 * laylines truncate exactly the passages that were about to succeed. Put
 * shoals in the tacking corridor and the learner never finds the harbour at
 * all; what it finds instead is that sailing straight onto a shoal is cheaper
 * than wandering until the clock runs out, which it will do, briskly, and
 * with no error message. This water was picked from five candidates by
 * training each one and keeping the layout that could actually be learned.
 */
export function makeSea(overrides: Partial<Sea> = {}): Sea {
	const cells = (pts: number[][]) => new Set(pts.map(([x, y]) => cellIndex(x, y)));
	return {
		w: CHART_W,
		h: CHART_H,
		start: cellIndex(7, 0),
		harbour: cellIndex(6, 9),
		land: cells([
			[6, 4],
			[7, 4],
			[6, 5]
		]),
		shoals: cells([
			[2, 5],
			[11, 4]
		]),
		windFrom: 0, // straight down the chart from the north
		timeCost: -0.28,
		harbourReward: 12,
		shoalReward: -9,
		ironsCost: -0.5,
		gamma: 0.97,
		maxSteps: 150,
		...overrides
	};
}

/** True wind angle for a heading under this wind: 0 is dead upwind. */
export function twaOf(sea: Sea, a: number): number {
	return angleBetween(headingAngle(a), sea.windFrom);
}

/** Whether the sails would luff on this heading — no movement at all. */
export function inIrons(sea: Sea, a: number): boolean {
	return boatSpeed(twaOf(sea, a)) <= 0;
}

/** Time charged for one cell on this heading. Slower point of sail, more time. */
export function stepTime(sea: Sea, a: number): number {
	const s = boatSpeed(twaOf(sea, a));
	if (s <= 0) return sea.ironsCost;
	return sea.timeCost / Math.max(MIN_SPEED, s);
}

export interface StepResult {
	s: number;
	r: number;
	done: boolean;
	/** True when the sails luffed and the boat stayed put. */
	irons: boolean;
}

/** One leg. Land and the chart's edge stop you; the clock runs regardless. */
export function stepSea(sea: Sea, s: number, a: number): StepResult {
	if (inIrons(sea, a)) return { s, r: sea.ironsCost, done: false, irons: true };
	const x = s % sea.w;
	const y = (s / sea.w) | 0;
	const nx = x + HEADING_DX[a];
	const ny = y + HEADING_DY[a];
	const time = stepTime(sea, a);
	if (nx < 0 || nx >= sea.w || ny < 0 || ny >= sea.h || sea.land.has(cellIndex(nx, ny, sea.w))) {
		return { s, r: time, done: false, irons: false };
	}
	const s2 = cellIndex(nx, ny, sea.w);
	if (s2 === sea.harbour) return { s: s2, r: time + sea.harbourReward, done: true, irons: false };
	if (sea.shoals.has(s2)) return { s: s2, r: time + sea.shoalReward, done: true, irons: false };
	return { s: s2, r: time, done: false, irons: false };
}

/** Fresh logits, all zero — a boat with no opinion about any heading. */
export function createTheta(sea: Sea): Float64Array {
	return new Float64Array(sea.w * sea.h * N_HEADINGS);
}

export function createBaseline(sea: Sea): Float64Array {
	return new Float64Array(sea.w * sea.h);
}

/** π(·|s) = softmax over the eight headings available in cell s. */
export function policyAt(theta: Float64Array, s: number, out?: Float64Array): Float64Array {
	return softmax(theta.subarray(s * N_HEADINGS, s * N_HEADINGS + N_HEADINGS), out);
}

export type PassageEnd = 'harbour' | 'shoal' | 'timeout';

export interface Passage {
	states: number[];
	actions: number[];
	rewards: number[];
	/** Every cell occupied, in order — for drawing the wake. */
	path: number[];
	end: PassageEnd;
	totalReward: number;
	steps: number;
	/** How many legs were wasted head-to-wind. */
	ironsSteps: number;
}

export function runPassage(sea: Sea, theta: Float64Array, rand: Rand, from = sea.start): Passage {
	const states: number[] = [];
	const actions: number[] = [];
	const rewards: number[] = [];
	let s = from;
	const path: number[] = [s];
	let end: PassageEnd = 'timeout';
	let total = 0;
	let ironsSteps = 0;
	const probs = new Float64Array(N_HEADINGS);
	for (let t = 0; t < sea.maxSteps; t++) {
		policyAt(theta, s, probs);
		const a = sampleFrom(probs, rand);
		const step = stepSea(sea, s, a);
		states.push(s);
		actions.push(a);
		rewards.push(step.r);
		if (step.irons) ironsSteps++;
		if (step.s !== s) path.push(step.s);
		total += step.r;
		s = step.s;
		if (step.done) {
			end = s === sea.harbour ? 'harbour' : 'shoal';
			break;
		}
	}
	return {
		states,
		actions,
		rewards,
		path,
		end,
		totalReward: total,
		steps: states.length,
		ironsSteps
	};
}

/** Discounted return-to-go, computed backwards. */
export function returnsToGo(rewards: number[], gamma: number, tail = 0): Float64Array {
	const g = new Float64Array(rewards.length);
	let acc = tail;
	for (let t = rewards.length - 1; t >= 0; t--) {
		acc = rewards[t] + gamma * acc;
		g[t] = acc;
	}
	return g;
}

export const BASELINE_LR = 0.15;

/** How much the policy is paid to stay undecided. Swept, not guessed. */
export const ENTROPY_BONUS = 0.02;

/**
 * REINFORCE with a per-state baseline, one whole passage:
 *
 *   θ[s_t, i] += lr · (G_t − V(s_t)) · (1[i = a_t] − π_i(s_t))
 *
 * Byte for byte the rule the gridworld used, with eight headings where there
 * were four directions. That it transfers without a single change is the point
 * of putting it here: the learner never learned to sail. It learned to raise
 * the probability of whatever preceded a better-than-usual passage.
 */
export function reinforceUpdate(
	sea: Sea,
	theta: Float64Array,
	baseline: Float64Array,
	p: Pick<Passage, 'states' | 'actions' | 'rewards' | 'path' | 'end'>,
	lr: number,
	entropy = ENTROPY_BONUS
): void {
	const tail = p.end === 'timeout' ? baseline[p.path[p.path.length - 1]] : 0;
	const g = returnsToGo(p.rewards, sea.gamma, tail);
	const probs = new Float64Array(N_HEADINGS);
	for (let t = 0; t < p.states.length; t++) {
		const s = p.states[t];
		const a = p.actions[t];
		policyAt(theta, s, probs);
		const adv = g[t] - baseline[s];
		// The entropy bonus, and it earns its place here rather than being
		// hygiene: a wind makes a uniform policy drift downwind, so a harbour to
		// windward is hard to stumble into, and a policy that commits early
		// commits to whatever it found first. Paying it a little to stay
		// undecided is the difference between this chapter working under any
		// wind the reader picks and working under one.
		let h = 0;
		if (entropy > 0) for (const q of probs) h -= q * Math.log(q + 1e-12);
		for (let i = 0; i < N_HEADINGS; i++) {
			const pg = adv * ((i === a ? 1 : 0) - probs[i]);
			const eg = entropy > 0 ? -probs[i] * (Math.log(probs[i] + 1e-12) + h) : 0;
			theta[s * N_HEADINGS + i] += lr * (pg + entropy * eg);
		}
		baseline[s] += BASELINE_LR * (g[t] - baseline[s]);
	}
}

/**
 * How often a practice passage starts somewhere other than the home mooring.
 *
 * Measured, and load-bearing. A wind makes a uniform policy drift downwind, so
 * a harbour to windward is something a random walk essentially never reaches —
 * and with the wind on some bearings the learner simply never finds it, sails
 * into a corner, and spends every episode there. Starting one passage in five
 * from a random patch of water fixes it completely: across five seeds and all
 * eight wind directions, every single run arrives.
 *
 * This is the same reverse curriculum the double pendulum practises its catch
 * with, in a world small enough to watch it work. Practising what you can
 * already almost do is how the gradient stays informative.
 */
export const SCATTER = 0.2;

/** Every cell a practice passage may be dropped into. */
export function waterCells(sea: Sea): number[] {
	const out: number[] = [];
	for (let i = 0; i < sea.w * sea.h; i++)
		if (!sea.land.has(i) && !sea.shoals.has(i) && i !== sea.harbour) out.push(i);
	return out;
}

/**
 * One practice passage and the update that follows it — the whole training
 * loop, so the plate and the tests cannot drift apart on how learning works.
 */
export function trainStep(
	sea: Sea,
	theta: Float64Array,
	baseline: Float64Array,
	rand: Rand,
	lr: number,
	water: number[],
	opts: { scatter?: number; entropy?: number } = {}
): Passage {
	const scatter = opts.scatter ?? SCATTER;
	const from = scatter > 0 && rand() < scatter ? water[(rand() * water.length) | 0] : sea.start;
	const p = runPassage(sea, theta, rand, from);
	reinforceUpdate(sea, theta, baseline, p, lr, opts.entropy);
	return p;
}

/**
 * The best passage time any route could achieve, by Dijkstra over headings —
 * the dashed line the plate draws so the reader can see how far from perfect
 * the policy is at every moment. Returns the total reward of that route, or
 * NaN when the harbour cannot be reached at all.
 */
export function bestPassage(sea: Sea): { reward: number; path: number[] } {
	const n = sea.w * sea.h;
	const cost = new Float64Array(n).fill(Infinity);
	const from = new Int32Array(n).fill(-1);
	cost[sea.start] = 0;
	const seen = new Uint8Array(n);
	for (;;) {
		let s = -1;
		let best = Infinity;
		for (let i = 0; i < n; i++)
			if (!seen[i] && cost[i] < best) {
				best = cost[i];
				s = i;
			}
		if (s < 0) break;
		if (s === sea.harbour) break;
		seen[s] = 1;
		const x = s % sea.w;
		const y = (s / sea.w) | 0;
		for (let a = 0; a < N_HEADINGS; a++) {
			if (inIrons(sea, a)) continue;
			const nx = x + HEADING_DX[a];
			const ny = y + HEADING_DY[a];
			if (nx < 0 || nx >= sea.w || ny < 0 || ny >= sea.h) continue;
			const s2 = cellIndex(nx, ny, sea.w);
			if (sea.land.has(s2)) continue;
			// a shoal ends the passage badly, so no sane route enters one
			if (sea.shoals.has(s2) && s2 !== sea.harbour) continue;
			const c = cost[s] - stepTime(sea, a); // stepTime is negative
			if (c < cost[s2]) {
				cost[s2] = c;
				from[s2] = s;
			}
		}
	}
	if (!Number.isFinite(cost[sea.harbour])) return { reward: NaN, path: [] };
	const path: number[] = [];
	for (let s = sea.harbour; s >= 0; s = from[s]) path.push(s);
	path.reverse();
	return { reward: sea.harbourReward - cost[sea.harbour], path };
}

/**
 * The greedy route the current policy would sail — argmax at every cell, no
 * dice. Used to read off what the policy has actually decided, and to test
 * that it discovered tacking rather than merely getting lucky.
 */
export function greedyRoute(sea: Sea, theta: Float64Array, limit = 200): number[] {
	const probs = new Float64Array(N_HEADINGS);
	const route: number[] = [];
	let s = sea.start;
	const seen = new Set<number>();
	for (let t = 0; t < limit; t++) {
		policyAt(theta, s, probs);
		let a = 0;
		for (let i = 1; i < N_HEADINGS; i++) if (probs[i] > probs[a]) a = i;
		route.push(a);
		const step = stepSea(sea, s, a);
		if (step.s === s && step.irons) break; // stuck head to wind
		s = step.s;
		if (step.done) break;
		const key = s * N_HEADINGS + a;
		if (seen.has(key)) break; // looping
		seen.add(key);
	}
	return route;
}

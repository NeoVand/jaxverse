// The inverted pendulum on a cart — the classic Barto–Sutton–Anderson task,
// with the standard constants everyone's cartpole uses. A linear policy on
// the four state numbers, squeezed through softmax over {push left, push
// right}, is enough to balance it; REINFORCE with return-to-go and a
// running baseline finds the weights. Plain arithmetic, no GPU.

import type { Rand } from './rng';
import { sampleFrom, softmax } from './softmax';
import { returnsToGo } from './gridworld';

export const GRAVITY = 9.8;
export const CART_MASS = 1.0;
export const POLE_MASS = 0.1;
export const POLE_HALF = 0.5; // half the pole length, metres
export const FORCE_MAG = 10.0;
export const DT = 0.02;
export const X_LIMIT = 2.4; // the track ends here
export const TH_LIMIT = (12 * Math.PI) / 180; // fallen past 12° = episode over
export const MAX_STEPS = 500;
export const GAMMA = 0.99;

const TOTAL_MASS = CART_MASS + POLE_MASS;
const PML = POLE_MASS * POLE_HALF;

export interface CartState {
	x: number; // cart position
	xd: number; // cart velocity
	th: number; // pole angle from vertical (radians, + leans right)
	thd: number; // pole angular velocity
}

/** Gym's reset: every state number uniform in ±0.05 — barely off balance. */
export function resetCart(rand: Rand): CartState {
	const u = () => (rand() * 2 - 1) * 0.05;
	return { x: u(), xd: u(), th: u(), thd: u() };
}

/**
 * One physics tick under a horizontal force on the cart (policy push plus
 * any shove the reader lands). Semi-implicit Euler: velocities first, then
 * positions — noticeably more stable than plain Euler at dt = 0.02.
 */
export function physicsStep(s: CartState, force: number): void {
	const sin = Math.sin(s.th);
	const cos = Math.cos(s.th);
	const temp = (force + PML * s.thd * s.thd * sin) / TOTAL_MASS;
	const thAcc =
		(GRAVITY * sin - cos * temp) / (POLE_HALF * (4 / 3 - (POLE_MASS * cos * cos) / TOTAL_MASS));
	const xAcc = temp - (PML * thAcc * cos) / TOTAL_MASS;
	s.xd += xAcc * DT;
	s.x += s.xd * DT;
	s.thd += thAcc * DT;
	s.th += s.thd * DT;
}

/** The episode is over when the pole falls past ±12° or the cart leaves the track. */
export function isFailed(s: CartState): boolean {
	return Math.abs(s.th) > TH_LIMIT || Math.abs(s.x) > X_LIMIT;
}

export const N_FEATURES = 4;
export const CART_ACTIONS = 2; // 0 = push left, 1 = push right

// Features scaled to comparable magnitude so one learning rate serves all
// four weights — position spans ±2.4 while the angle spans ±0.21.
const FEAT_SCALE = [1 / X_LIMIT, 1 / 3, 1 / TH_LIMIT, 1 / 3] as const;

/** State → scaled feature vector, written into `out`. */
export function cartFeatures(s: CartState, out: Float64Array): Float64Array {
	out[0] = s.x * FEAT_SCALE[0];
	out[1] = s.xd * FEAT_SCALE[1];
	out[2] = s.th * FEAT_SCALE[2];
	out[3] = s.thd * FEAT_SCALE[3];
	return out;
}

/** Fresh policy weights, all zero — push left and push right equally likely. */
export function createCartTheta(): Float64Array {
	return new Float64Array(CART_ACTIONS * N_FEATURES);
}

/** Per-timestep running baseline — "usual return from step t onward". */
export function createCartBaseline(): Float64Array {
	return new Float64Array(MAX_STEPS);
}

/** π(·|s) = softmax of the two linear scores θ_a · features. */
export function cartPolicy(
	theta: Float64Array,
	feats: Float64Array,
	out?: Float64Array
): Float64Array {
	const logits = new Float64Array(CART_ACTIONS);
	for (let a = 0; a < CART_ACTIONS; a++) {
		let z = 0;
		for (let i = 0; i < N_FEATURES; i++) z += theta[a * N_FEATURES + i] * feats[i];
		logits[a] = z;
	}
	return softmax(logits, out);
}

export function actionForce(a: number): number {
	return a === 0 ? -FORCE_MAG : FORCE_MAG;
}

export interface CartEpisode {
	/** Features of every visited state, T×4, flat. */
	feats: Float64Array;
	actions: number[];
	/** Steps survived — also the undiscounted return, at +1 per step. */
	steps: number;
}

/** Roll one headless episode under the current policy. */
export function runCartEpisode(theta: Float64Array, rand: Rand, maxSteps = MAX_STEPS): CartEpisode {
	const s = resetCart(rand);
	const feats = new Float64Array(maxSteps * N_FEATURES);
	const actions: number[] = [];
	const f = new Float64Array(N_FEATURES);
	const probs = new Float64Array(CART_ACTIONS);
	let t = 0;
	while (t < maxSteps) {
		cartFeatures(s, f);
		feats.set(f, t * N_FEATURES);
		cartPolicy(theta, f, probs);
		const a = sampleFrom(probs, rand);
		actions.push(a);
		physicsStep(s, actionForce(a));
		t++;
		if (isFailed(s)) break;
	}
	return { feats, actions, steps: t };
}

export const CART_BASELINE_LR = 0.1;

/**
 * REINFORCE with baseline over one whole episode (reward is +1 per step, so
 * G_t = 1 + γ + … over the survived tail):
 *
 *   θ[a, i] += lr · (G_t − b_t) · (1[a = a_t] − π_a(s_t)) · feat_i
 *
 * The baseline is a running mean of G_t *per timestep* — the continuous-state
 * cousin of the gridworld's per-state table. Advantages are then divided by
 * their own within-episode standard deviation: with γ = 0.99 a fresh episode
 * has G_0 ≈ 99 against a baseline of 0, and hundreds of unscaled steps of
 * that size drive θ to saturation in a single episode — sometimes saturated
 * wrong, permanently. Whitening bounds the step; direction is untouched.
 * Reaching the 500-step cap is treated as terminal, the standard convention:
 * with only positive rewards there is no idle-loop pathology for truncation
 * to feed.
 */
export function cartReinforceUpdate(
	theta: Float64Array,
	baseline: Float64Array,
	ep: CartEpisode,
	lr: number,
	gamma = GAMMA
): void {
	const T = ep.steps;
	const rewards = new Array<number>(T).fill(1);
	const g = returnsToGo(rewards, gamma);
	const adv = new Float64Array(T);
	let m = 0;
	for (let t = 0; t < T; t++) {
		adv[t] = g[t] - baseline[t];
		baseline[t] += CART_BASELINE_LR * (g[t] - baseline[t]);
		m += adv[t];
	}
	m /= T;
	let v = 0;
	for (let t = 0; t < T; t++) v += (adv[t] - m) * (adv[t] - m);
	const sd = Math.sqrt(v / T) + 1e-6;
	const probs = new Float64Array(CART_ACTIONS);
	const f = new Float64Array(N_FEATURES);
	for (let t = 0; t < T; t++) {
		for (let i = 0; i < N_FEATURES; i++) f[i] = ep.feats[t * N_FEATURES + i];
		cartPolicy(theta, f, probs);
		const a = ep.actions[t];
		const w = (lr * adv[t]) / sd;
		for (let b = 0; b < CART_ACTIONS; b++) {
			const coef = w * ((b === a ? 1 : 0) - probs[b]);
			for (let i = 0; i < N_FEATURES; i++) theta[b * N_FEATURES + i] += coef * f[i];
		}
	}
}

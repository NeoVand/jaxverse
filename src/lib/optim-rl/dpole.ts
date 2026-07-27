// The double pendulum swing-up on a sliding hinge — two point masses on
// massless rods, stacked on a pivot that slides along a rail. The links
// start hanging straight down; the only control is a horizontal force on
// the hinge, so the policy must pump energy into the stack and then catch
// it upright, steering the top link *through* the bottom one the whole way.
// Reward is dense — the height of the tip, every step — episodes are a
// fixed length, and nothing ever "fails": the policy simply earns more the
// longer it keeps the stack tall. A linear softmax policy over {push left,
// coast, push right} on trigonometric features is enough; REINFORCE with a
// per-timestep baseline and whitened advantages finds the weights. Plain
// arithmetic, no GPU.
//
// Dynamics come from the Lagrangian of q = [x, θ1, θ2] (angles from
// vertical), giving the symmetric 3×3 system D(q)·q̈ = b(q, q̇, u):
//
//   D = [ M+m1+m2         (m1+m2)ℓ1c1      m2ℓ2c2      ]
//       [ (m1+m2)ℓ1c1     (m1+m2)ℓ1²       m2ℓ1ℓ2c12   ]
//       [ m2ℓ2c2          m2ℓ1ℓ2c12        m2ℓ2²       ]
//
//   b = [ u + (m1+m2)ℓ1s1·θ̇1² + m2ℓ2s2·θ̇2²
//         (m1+m2)gℓ1s1 − m2ℓ1ℓ2s12·θ̇2²
//         m2gℓ2s2 + m2ℓ1ℓ2s12·θ̇1²          ]
//
// (c1 = cos θ1, s12 = sin(θ1−θ2), …). Integration is RK4 in substeps: the
// stage keeps simulating after a fall, and a swinging double pendulum is
// exactly the flow that punishes a lazy integrator.

import type { Rand } from './rng';
import { sampleFrom, softmax } from './softmax';
import { returnsToGo } from './gridworld';

export const GRAVITY = 9.8;
export const HINGE_MASS = 1.0; // the sliding dot
export const M1 = 0.1; // lower bob
export const M2 = 0.05; // upper bob — lighter than the lower one
export const L1 = 0.5; // lower rod, metres
export const L2 = 0.35; // upper rod — shorter, too: a lighter, shorter top
// link tames the chaos and widens the balance basin. Every published
// swing-up rig picks its mass ratios kindly; so do we.
export const FORCE_MAG = 11.0;
export const DT = 0.02; // one control tick — the policy acts at 50 Hz
const SUBSTEPS = 4; // RK4 at 200 Hz keeps the chaotic regime honest
export const X_LIMIT = 2.4; // the rail ends here — a soft bumper, not a failure
export const TH_LIMIT = (18 * Math.PI) / 180; // the drawn target cones, nothing more
export const MAX_STEPS = 400; // 8 seconds — time to swing up and hold
export const GAMMA = 0.99;

const MT = HINGE_MASS + M1 + M2;
const M12 = M1 + M2;

export interface DpoleState {
	x: number; // hinge position on the rail
	xd: number; // hinge velocity
	th1: number; // lower link angle from vertical (+ leans right)
	th1d: number;
	th2: number; // upper link angle from vertical
	th2d: number;
}

/** Where an episode begins: hanging (the honest start), balanced at the
 * top (learn the hold), or at the brink — both links near the top, tilted
 * and moving, exactly the awkward hand-off between swing and catch. The
 * reverse curriculum: practice the goal and its doorstep, not just the
 * journey. */
export type DpoleStart = 0 | 1 | 2; // 0 = hanging, 1 = up, 2 = brink

export function resetDpole(rand: Rand, kind: DpoleStart = 0): DpoleState {
	const u = (m: number) => (rand() * 2 - 1) * m;
	if (kind === 1)
		return { x: u(0.04), xd: u(0.04), th1: u(0.04), th1d: u(0.04), th2: u(0.04), th2d: u(0.04) };
	if (kind === 2)
		return { x: u(0.5), xd: u(0.5), th1: u(0.5), th1d: u(1), th2: u(0.5), th2d: u(1.5) };
	return {
		x: u(0.04),
		xd: u(0.04),
		th1: Math.PI + u(0.04),
		th1d: u(0.04),
		th2: Math.PI + u(0.04),
		th2d: u(0.04)
	};
}

/** q̈ = D⁻¹b by explicit 3×3 elimination — written into `out`. */
function accel(s: DpoleState, force: number, out: Float64Array): void {
	const s1 = Math.sin(s.th1);
	const c1 = Math.cos(s.th1);
	const s2 = Math.sin(s.th2);
	const c2 = Math.cos(s.th2);
	const s12 = Math.sin(s.th1 - s.th2);
	const c12 = Math.cos(s.th1 - s.th2);

	const d11 = MT;
	const d12 = M12 * L1 * c1;
	const d13 = M2 * L2 * c2;
	const d22 = M12 * L1 * L1;
	const d23 = M2 * L1 * L2 * c12;
	const d33 = M2 * L2 * L2;

	const b1 = force + M12 * L1 * s1 * s.th1d * s.th1d + M2 * L2 * s2 * s.th2d * s.th2d;
	const b2 = M12 * GRAVITY * L1 * s1 - M2 * L1 * L2 * s12 * s.th2d * s.th2d;
	const b3 = M2 * GRAVITY * L2 * s2 + M2 * L1 * L2 * s12 * s.th1d * s.th1d;

	// symmetric 3×3 solve by cofactors
	const a11 = d22 * d33 - d23 * d23;
	const a12 = d13 * d23 - d12 * d33;
	const a13 = d12 * d23 - d13 * d22;
	const a22 = d11 * d33 - d13 * d13;
	const a23 = d12 * d13 - d11 * d23;
	const a33 = d11 * d22 - d12 * d12;
	const det = d11 * a11 + d12 * a12 + d13 * a13;

	out[0] = (a11 * b1 + a12 * b2 + a13 * b3) / det;
	out[1] = (a12 * b1 + a22 * b2 + a23 * b3) / det;
	out[2] = (a13 * b1 + a23 * b2 + a33 * b3) / det;
}

const K1 = new Float64Array(3);
const K2 = new Float64Array(3);
const K3 = new Float64Array(3);
const K4 = new Float64Array(3);
const TMP: DpoleState = { x: 0, xd: 0, th1: 0, th1d: 0, th2: 0, th2d: 0 };

function rk4Sub(s: DpoleState, force: number, h: number): void {
	// stage derivatives of the velocities; positions advance by the velocities
	accel(s, force, K1);
	TMP.x = s.x + (h / 2) * s.xd;
	TMP.xd = s.xd + (h / 2) * K1[0];
	TMP.th1 = s.th1 + (h / 2) * s.th1d;
	TMP.th1d = s.th1d + (h / 2) * K1[1];
	TMP.th2 = s.th2 + (h / 2) * s.th2d;
	TMP.th2d = s.th2d + (h / 2) * K1[2];
	accel(TMP, force, K2);
	const xd2 = TMP.xd;
	const th1d2 = TMP.th1d;
	const th2d2 = TMP.th2d;
	TMP.x = s.x + (h / 2) * xd2;
	TMP.xd = s.xd + (h / 2) * K2[0];
	TMP.th1 = s.th1 + (h / 2) * th1d2;
	TMP.th1d = s.th1d + (h / 2) * K2[1];
	TMP.th2 = s.th2 + (h / 2) * th2d2;
	TMP.th2d = s.th2d + (h / 2) * K2[2];
	accel(TMP, force, K3);
	const xd3 = TMP.xd;
	const th1d3 = TMP.th1d;
	const th2d3 = TMP.th2d;
	TMP.x = s.x + h * xd3;
	TMP.xd = s.xd + h * K3[0];
	TMP.th1 = s.th1 + h * th1d3;
	TMP.th1d = s.th1d + h * K3[1];
	TMP.th2 = s.th2 + h * th2d3;
	TMP.th2d = s.th2d + h * K3[2];
	accel(TMP, force, K4);
	const xd4 = TMP.xd;
	const th1d4 = TMP.th1d;
	const th2d4 = TMP.th2d;

	s.x += (h / 6) * (s.xd + 2 * xd2 + 2 * xd3 + xd4);
	s.th1 += (h / 6) * (s.th1d + 2 * th1d2 + 2 * th1d3 + th1d4);
	s.th2 += (h / 6) * (s.th2d + 2 * th2d2 + 2 * th2d3 + th2d4);
	s.xd += (h / 6) * (K1[0] + 2 * K2[0] + 2 * K3[0] + K4[0]);
	s.th1d += (h / 6) * (K1[1] + 2 * K2[1] + 2 * K3[1] + K4[1]);
	s.th2d += (h / 6) * (K1[2] + 2 * K2[2] + 2 * K3[2] + K4[2]);
}

/** One control tick under a horizontal force on the hinge. The rail's ends
 * are soft bumpers, not a failure: the hinge bounces and play continues. */
export function physicsStep(s: DpoleState, force: number): void {
	const h = DT / SUBSTEPS;
	for (let i = 0; i < SUBSTEPS; i++) rk4Sub(s, force, h);
	if (Math.abs(s.x) > X_LIMIT) {
		s.x = Math.sign(s.x) * X_LIMIT;
		s.xd *= -0.3;
	}
}

/** Normalized tip height: −1 hanging, +1 fully upright — the reward's core. */
export function tipHeight(s: DpoleState): number {
	return (L1 * Math.cos(s.th1) + L2 * Math.cos(s.th2)) / (L1 + L2);
}

/** Both links inside the drawn cones — "caught", for the reader's scoreboard. */
export function isUpright(s: DpoleState): boolean {
	const a1 = Math.atan2(Math.sin(s.th1), Math.cos(s.th1));
	const a2 = Math.atan2(Math.sin(s.th2), Math.cos(s.th2));
	return Math.abs(a1) < TH_LIMIT && Math.abs(a2) < TH_LIMIT;
}

/** Per-step reward: the tip's height, plus a catch bonus that is sharply
 * peaked at dead vertical, discounted by angular speed, and — crucially —
 * five times taller than anything the swing can earn. Inside the cones
 * "higher" is nearly flat, so without the peak the policy drifts to the
 * cone's edge and slides out; and without the ×5 the policy learns to
 * park the lower link upright at half height forever, because the last
 * leg of the climb passes through a valley of awkward, low-reward states
 * that a modest summit never pays back. A whisper of a dock for hugging
 * the bumpers keeps the swing-up mid-rail. */
export function dpoleReward(s: DpoleState): number {
	const h01 = (tipHeight(s) + 1) / 2;
	const calm = 1 / (1 + 0.5 * (Math.abs(s.th1d) + Math.abs(s.th2d)));
	return h01 + 5 * h01 ** 8 * calm - 0.01 * (Math.abs(s.x) / X_LIMIT);
}

/** The policy decides every tick — balancing the top of the stack needs
 * the full 50 Hz; slower cadences cannot hold the catch. */
export const ACTION_REPEAT = 1;
export const DECISIONS = MAX_STEPS / ACTION_REPEAT;

export const N_FEATURES = 16;
// Five graded pushes. The swing-up needs the full ±11 N to pump energy in;
// the catch needs a whisper — with only full-strength pushes available,
// every correction at the top overshoots the basin, and the policy learns
// the sad truth that not touching it is best. The quarter-strength pushes
// are what make the catch physically holdable. ("Do nothing" is still a
// trap early on; the entropy bonus below keeps the policy from freezing
// into it.)
export const ACTION_FORCES = [-1, -0.25, 0, 0.25, 1] as const;
export const DPOLE_ACTIONS = ACTION_FORCES.length;

const clip1 = (v: number) => Math.max(-1, Math.min(1, v));
const wrap = (th: number) => Math.atan2(Math.sin(th), Math.cos(th));

/** State → feature vector, written into `out`. Two dashboards, one policy:
 * swing gauges (sin/cos of each angle, velocities) that fade out near the
 * top, and fine balance gauges (wrapped angles and velocities, amplified
 * and clipped) that fade in there. The gate g — the tip's height to the
 * fourth power — does the fading. Without it the two skills fight over
 * the same weights: far from the top the clipped gauges saturate into
 * constant flags, near the top cos θ ≈ 1 acts as a stray bias, and a
 * linear policy is forced into one compromise controller that can neither
 * pump cleanly nor hold. Gated, it is two controllers in one rulebook. */
export function dpoleFeatures(s: DpoleState, out: Float64Array): Float64Array {
	const h01 = (tipHeight(s) + 1) / 2;
	const g = h01 ** 4;
	out[0] = s.x / X_LIMIT;
	out[1] = s.xd / 3;
	out[2] = (1 - g) * Math.sin(s.th1);
	out[3] = (1 - g) * Math.cos(s.th1);
	out[4] = ((1 - g) * s.th1d) / 6;
	out[5] = (1 - g) * Math.sin(s.th2);
	out[6] = (1 - g) * Math.cos(s.th2);
	out[7] = ((1 - g) * s.th2d) / 8;
	out[8] = g * clip1(wrap(s.th1) / TH_LIMIT);
	out[9] = g * clip1(wrap(s.th2) / TH_LIMIT);
	out[10] = g * clip1(s.th1d / 1.5);
	out[11] = g * clip1(s.th2d / 2);
	// wide copies of the same gauges, clipping at ±45° instead of ±18°:
	// the catch is decided at 20–40° of tilt, where the fine gauges above
	// are already pegged and the swing gauges are faded — without these
	// the policy is nearly blind at exactly the moment that matters
	out[12] = g * clip1(wrap(s.th1) / (2.5 * TH_LIMIT));
	out[13] = g * clip1(wrap(s.th2) / (2.5 * TH_LIMIT));
	out[14] = g * clip1(s.th1d / 3.5);
	out[15] = g * clip1(s.th2d / 5);
	return out;
}

// ── the policy: one weight per (action, feature) pair — 5×12 = 60 numbers ──
// No hidden layer. The zoomed features saturate into on/off flags during
// the swing and read like a ruler near the top, which is exactly the
// nonlinearity the two regimes need; a linear vote over them can both
// pump and catch. (A small MLP was tried and learned far slower — the
// classic price of learning features while the policy gradient screams.)
export const N_PARAMS = DPOLE_ACTIONS * N_FEATURES;

/** Fresh policy weights, all zero — every push exactly equally likely,
 * the honest coin flip the chapter opens with. */
export function createDpoleTheta(): Float64Array {
	return new Float64Array(N_PARAMS);
}

/** Forward pass: writes probabilities into `probs`. */
function forward(theta: Float64Array, feats: Float64Array, probs: Float64Array): void {
	const logits = LOGITS;
	for (let a = 0; a < DPOLE_ACTIONS; a++) {
		let z = 0;
		for (let i = 0; i < N_FEATURES; i++) z += theta[a * N_FEATURES + i] * feats[i];
		logits[a] = z;
	}
	softmax(logits, probs);
}
const LOGITS = new Float64Array(DPOLE_ACTIONS);

/** Three per-decision running baselines — "usual return from step t
 * onward", kept separately per start kind (hanging, up, brink): their
 * returns live on different planets, and sharing one baseline would grade
 * every action by the luck of its start. Each track carries one extra
 * slot: a running RMS of advantages, the normalizer. */
export function createDpoleBaseline(): Float64Array {
	return new Float64Array(3 * (DECISIONS + 1));
}

/** π(·|s) = softmax of the network's three scores. */
export function dpolePolicy(
	theta: Float64Array,
	feats: Float64Array,
	out?: Float64Array
): Float64Array {
	const probs = out ?? new Float64Array(DPOLE_ACTIONS);
	forward(theta, feats, probs);
	return probs;
}

export function actionForce(a: number): number {
	return ACTION_FORCES[a] * FORCE_MAG;
}

export interface DpoleEpisode {
	/** Features of every decision state, T×8, flat. */
	feats: Float64Array;
	actions: number[];
	/** Per-decision rewards — the tip's height, mostly. */
	rewards: Float64Array;
	/** Episode length in decisions — fixed for hanging and brink starts,
	 * cut short at the first fall for balanced starts. */
	steps: number;
	/** How this episode began: hanging, up, or at the brink. */
	kind: DpoleStart;
	/** Undiscounted return, Σ rewards — the sparkline's number. */
	ret: number;
	/** Physics ticks spent caught inside the cones — the real scoreboard. */
	caught: number;
}

/** Curriculum mix: half the practice is the real journey from hanging;
 * a quarter starts balanced (learn the hold); a quarter starts at the
 * brink (learn the catch). */
export function drawStart(rand: Rand): DpoleStart {
	const r = rand();
	return r < 0.5 ? 0 : r < 0.75 ? 1 : 2;
}

/** Roll one headless episode under the current policy. Hanging and brink
 * starts run the full fixed length — there is nothing to fail, only reward
 * to gather. Balanced starts end at the first fall: their return is
 * survival itself, and letting them run on would bury the balance signal
 * under four hundred steps of chaotic-swing luck. */
export function runDpoleEpisode(theta: Float64Array, rand: Rand, kind?: DpoleStart): DpoleEpisode {
	const k = kind ?? drawStart(rand);
	const s = resetDpole(rand, k);
	const feats = new Float64Array(DECISIONS * N_FEATURES);
	const rewards = new Float64Array(DECISIONS);
	const actions: number[] = [];
	const f = new Float64Array(N_FEATURES);
	const probs = new Float64Array(DPOLE_ACTIONS);
	let ret = 0;
	let caught = 0;
	let t = 0;
	while (t < DECISIONS) {
		dpoleFeatures(s, f);
		feats.set(f, t * N_FEATURES);
		dpolePolicy(theta, f, probs);
		const a = sampleFrom(probs, rand);
		actions.push(a);
		let r = 0;
		for (let k = 0; k < ACTION_REPEAT; k++) {
			physicsStep(s, actionForce(a));
			r += dpoleReward(s) / ACTION_REPEAT;
			if (isUpright(s)) caught++;
		}
		rewards[t] = r;
		ret += r;
		t++;
		if (k === 1 && !isUpright(s)) break;
	}
	return { feats, actions, rewards, steps: t, kind: k, ret, caught };
}

export const DPOLE_BASELINE_LR = 0.1;
/** Gentle weight decay per update: keeps the softmax from saturating, so
 * the policy never stops exploring — REINFORCE's classic failure here is a
 * premature freeze, not a wrong answer. */
export const DPOLE_DECAY = 1e-3;
/** Entropy bonus, same purpose from the other side: a small standing
 * reward for staying undecided, so exploration survives the early stretch
 * where every push honestly looks worse than the last. */
export const DPOLE_ENTROPY = 0.015;

/**
 * REINFORCE with baseline over one whole episode (reward is the tip's
 * height, so G_t is "how tall the near future stays from here"):
 *
 *   θ[a, i] += lr · (G_t − b_t) · (1[a = a_t] − π_a(s_t)) · feat_i
 *
 * The baseline is a running mean of G_t *per decision*; advantages are then
 * divided by a running RMS of advantages across episodes, which bounds the
 * step without touching its direction. The scale must be cross-episode: a
 * quiet episode's advantages are genuinely small, and dividing them by
 * their own tiny spread would blow them up — stillness would reinforce
 * itself, and the policy would freeze at the bottom forever.
 */
export function dpoleReinforceUpdate(
	theta: Float64Array,
	baseline: Float64Array,
	eps: DpoleEpisode[],
	lr: number,
	gamma = GAMMA,
	decay = DPOLE_DECAY,
	beta = DPOLE_ENTROPY
): void {
	const grad = new Float64Array(N_PARAMS);
	const probs = new Float64Array(DPOLE_ACTIONS);
	const gz = new Float64Array(DPOLE_ACTIONS);
	const f = new Float64Array(N_FEATURES);
	for (const ep of eps) {
		const T = ep.steps;
		const off = ep.kind * (DECISIONS + 1);
		const g = returnsToGo(Array.from(ep.rewards.subarray(0, T)), gamma);
		const adv = new Float64Array(T);
		let sq = 0;
		for (let t = 0; t < T; t++) {
			adv[t] = g[t] - baseline[off + t];
			baseline[off + t] += DPOLE_BASELINE_LR * (g[t] - baseline[off + t]);
			sq += adv[t] * adv[t];
		}
		const epRms = Math.sqrt(sq / T);
		const prev = baseline[off + DECISIONS];
		const rms = prev === 0 ? epRms : prev * 0.99 + epRms * 0.01;
		baseline[off + DECISIONS] = rms;
		const scale = Math.max(rms, 1e-3);
		for (let t = 0; t < T; t++) {
			for (let i = 0; i < N_FEATURES; i++) f[i] = ep.feats[t * N_FEATURES + i];
			forward(theta, f, probs);
			const a = ep.actions[t];
			const w = adv[t] / (scale * eps.length);
			// ∇ logits of w·log π(a) + β·H(π): the score-function trick plus
			// the entropy bonus
			let hent = 0;
			for (let b = 0; b < DPOLE_ACTIONS; b++)
				if (probs[b] > 1e-12) hent -= probs[b] * Math.log(probs[b]);
			for (let b = 0; b < DPOLE_ACTIONS; b++) {
				const dH = probs[b] > 1e-12 ? -probs[b] * (Math.log(probs[b]) + hent) : 0;
				gz[b] = w * ((b === a ? 1 : 0) - probs[b]) + (beta / eps.length) * dH;
			}
			for (let b = 0; b < DPOLE_ACTIONS; b++) {
				const row = b * N_FEATURES;
				for (let i = 0; i < N_FEATURES; i++) grad[row + i] += gz[b] * f[i];
			}
		}
	}
	// the decay keeps the softmax soft, so exploration never dies —
	// REINFORCE's classic failure here is a premature freeze
	for (let i = 0; i < N_PARAMS; i++) theta[i] = theta[i] * (1 - decay) + lr * grad[i];
}

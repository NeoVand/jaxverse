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
import { returnsToGo } from './chart';

export const GRAVITY = 9.8;
export const HINGE_MASS = 1.0; // the sliding dot
export const M1 = 0.1; // lower bob
export const M2 = 0.05; // upper bob — lighter than the lower one
export const L1 = 0.5; // lower rod, metres
export const L2 = 0.35; // upper rod — shorter, too: a lighter, shorter top
// link tames the chaos and widens the balance basin. Every published
// swing-up rig picks its mass ratios kindly; so do we.
export const FORCE_MAG = 20.0;
export const DT = 0.02; // one control tick — the policy acts at 50 Hz
const SUBSTEPS = 4; // RK4 at 200 Hz keeps the chaotic regime honest
export const X_LIMIT = 2.4; // the rail ends here — a soft bumper, not a failure
// Drag at the two pins (and a whisper on the slide). The pin drag is
// QUADRATIC in spin — τ = −c·θ̇|θ̇| — because it must be two things at
// once: negligible at honest pumping speeds (~5 rad/s it steals a fraction
// of a percent per swing) and lethal to a runaway propeller (at 40 rad/s
// it bites 64× harder, bleeding the whirl out in ~5 s). Linear friction
// cannot do both. Without any, the rig is a perpetual-motion machine: one
// missed catch leaves the stack over-energized FOREVER, spinning in a
// state the policy never practiced from, and the live show never recovers.
const B_X = 0.05; // N·s/m on the slide
const C_1 = 2e-4; // N·m·s²/rad² at the lower pin
const C_2 = 5e-5; // and the upper
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
	// balanced starts get REAL perturbations (up to ~17°, real drift): the
	// hold's basin is exactly the target the catch must hand into, and a
	// hold taught only from ±2° grips so tightly that every slightly-off
	// catch — and every reader shove — is beyond saving. Wide, sloppy
	// starts here directly buy catch success and shove recovery.
	if (kind === 1)
		return { x: u(0.3), xd: u(0.4), th1: u(0.3), th1d: u(0.8), th2: u(0.3), th2d: u(1.2) };
	// the brink — but never from here: brink episodes replay real delivered
	// states (see DeliveryBuffer). This fallback only feeds the curriculum
	// before the first delivery ever lands.
	if (kind === 2)
		return { x: u(0.5), xd: u(0.5), th1: u(0.4), th1d: u(2.5), th2: u(0.55), th2d: u(3.5) };
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

	const b1 = force + M12 * L1 * s1 * s.th1d * s.th1d + M2 * L2 * s2 * s.th2d * s.th2d - B_X * s.xd;
	const b2 =
		M12 * GRAVITY * L1 * s1 -
		M2 * L1 * L2 * s12 * s.th2d * s.th2d -
		C_1 * s.th1d * Math.abs(s.th1d);
	const b3 =
		M2 * GRAVITY * L2 * s2 + M2 * L1 * L2 * s12 * s.th1d * s.th1d - C_2 * s.th2d * Math.abs(s.th2d);

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

/** Total mechanical energy of the links (kinetic + potential), treating the
 * hinge's own sliding energy as the actuator's business, not the stack's. */
export function mechanicalEnergy(s: DpoleState): number {
	const v1x = s.xd + L1 * Math.cos(s.th1) * s.th1d;
	const v1y = -L1 * Math.sin(s.th1) * s.th1d;
	const v2x = v1x + L2 * Math.cos(s.th2) * s.th2d;
	const v2y = v1y - L2 * Math.sin(s.th2) * s.th2d;
	const T = 0.5 * M1 * (v1x * v1x + v1y * v1y) + 0.5 * M2 * (v2x * v2x + v2y * v2y);
	const V =
		M1 * GRAVITY * L1 * Math.cos(s.th1) +
		M2 * GRAVITY * (L1 * Math.cos(s.th1) + L2 * Math.cos(s.th2));
	return T + V;
}

/** The energy of standing perfectly still at the top… */
export const E_TOP = M1 * GRAVITY * L1 + M2 * GRAVITY * (L1 + L2);
/** …and of hanging perfectly still at the bottom — the span normalizes. */
export const E_SPAN = 2 * E_TOP;

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
	const spin = Math.abs(s.th1d) + Math.abs(s.th2d);
	const calm = 1 / (1 + 0.5 * spin);
	// the energy term: closeness of the stack's mechanical energy to the
	// value it has standing still at the top. Height alone cannot grade a
	// good pump — halfway through a rising swing the tip is low but the
	// energy is right — and it also scolds overshoot: arriving with spin to
	// spare is not progress toward standing.
	const eErr = Math.abs(mechanicalEnergy(s) - E_TOP) / E_SPAN;
	if (!isHighRegime(s)) {
		// Below the hand-off, the per-step reward is a whisper — shaping
		// toward height and the right energy, minus a small tax on the
		// clock. Anything louder and the policy farms it: an earlier draft
		// paid ~1/tick for cruising at top energy, which made *delivering*
		// (and ending the episode) a pay cut, so the stack learned to swing
		// forever, beautifully, and never arrive. Down here the delivery
		// bonus must be the only real income. The penalties below are the
		// stability half of the bargain:
		// — excess spin: free up to the ~10 rad/s an honest pump needs at
		//   the bottom, then quadratic — a propeller's good-looking flashes
		//   near the top must never out-earn its cost down here;
		// — energy surplus: pumping PAST the top's energy is not neutral,
		//   it is the whirl being wound up — braking must pay immediately;
		// — the rail's last stretch: leaning on a bumper is not a strategy.
		const overSpin = Math.max(0, spin - 10);
		const surplus = Math.max(0, (mechanicalEnergy(s) - E_TOP) / E_SPAN);
		const edge = Math.max(0, Math.abs(s.x) / X_LIMIT - 0.7);
		return (
			0.1 * h01 +
			0.1 * Math.max(0, 1 - eErr) -
			0.2 -
			0.01 * (Math.abs(s.x) / X_LIMIT) -
			Math.min(0.6, 0.02 * overSpin * overSpin) -
			0.3 * Math.min(2, surplus) -
			2 * edge * edge
		);
	}
	// Above the hand-off: height, energy, and a catch bonus sharply peaked
	// at dead vertical, discounted by calm² — a fast fly-through must be
	// worth almost nothing next to standing still. The spin tax kills the
	// orbit (drift past the summit each lap, skimming reward without ever
	// catching); it lives only up here because at the bottom of an honest
	// pump the links MUST move ~8 rad/s. Capped: since fly-throughs no
	// longer end the episode, a 40 rad/s whirl would otherwise stack a
	// −70/tick monster that swamps the baselines — the CAP keeps the tax at
	// "you lost the whole upright bonus and then some", which is plenty.
	const over = Math.max(0, spin - 3);
	return (
		h01 +
		Math.max(0, 1 - eErr) +
		5 * h01 ** 8 * calm * calm -
		Math.min(6, 0.05 * over * over) -
		0.01 * (Math.abs(s.x) / X_LIMIT)
	);
}

/** The policy decides every tick — balancing the top of the stack needs
 * the full 50 Hz; slower cadences cannot hold the catch. */
export const ACTION_REPEAT = 1;
export const DECISIONS = MAX_STEPS / ACTION_REPEAT;

export const N_FEATURES = 33;
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

/** The hand-off: above this tip height the catch half of the rulebook
 * drives instead of the swing half. */
export const HANDOFF_H01 = 0.82;
/** And above this, the hold half: catching (aggressive, tilted, moving)
 * and holding (gentle trims around dead vertical) are different jobs, and
 * one linear controller cannot be both — with the two sharing weights,
 * every catch lesson was a hold un-lesson, and the hold capped at a
 * quarter of its solo skill. */
export const CORE_H01 = 0.96;

export function isHighRegime(s: DpoleState): boolean {
	return (tipHeight(s) + 1) / 2 > HANDOFF_H01;
}

/** A crossing only counts as a DELIVERY when it arrives slow enough to
 * catch. Without this gate a swing episode ended at its first fly-through
 * — so a policy whirling at 40 rad/s could never even practice the bleed
 * (every braking attempt was cut short at the next crossing), and the live
 * stage, which never resets, stayed a propeller forever. The threshold is
 * strict on purpose, and tuned by trial: at 6 rad/s the median arrival was
 * caught one in ten. Tightening below 4 backfires — the swing never
 * DISCOVERS deliveries that slow (0/100 at 2.5, seed-dependent failure at
 * 3), so nothing downstream trains at all. Four is the workable gate:
 * the swing brakes to ~3.7 rad/s and still finds the bonus reliably. */
export function isCatchable(s: DpoleState): boolean {
	return Math.abs(s.th1d) + Math.abs(s.th2d) < 4;
}

export function isDelivery(s: DpoleState): boolean {
	return isHighRegime(s) && isCatchable(s);
}

/** Once the balance half has the wheel it keeps it until the tip truly
 * drops out — a lower bar than the hand-off, so the regimes don't chatter
 * at the boundary. */
export const REGIME_OUT_H01 = 0.7;

export function hasDroppedOut(s: DpoleState): boolean {
	return (tipHeight(s) + 1) / 2 < REGIME_OUT_H01;
}

/** The terminal prize of a swing episode: it ends the moment the tip
 * crosses the hand-off height, and this grades WHAT it delivered there.
 * Three factors, all learned the hard way. Spin: a slow arrival is worth
 * an order of magnitude more than a fly-through. Lean: the tip-height gate
 * alone admits states with the links folded 40–60° off vertical — a CEM
 * planner with perfect knowledge and continuous force could catch only
 * 7 of 30 such "deliveries", so paying full price for them trained the
 * swing to produce garbage the catch could never redeem. Rail room: an
 * arrival at the bumper leaves the cart nowhere to run under the stack;
 * this factor is also what finally made camping at the rail ends a losing
 * strategy rather than a quirk. */
export function deliveryBonus(s: DpoleState): number {
	const spin = Math.abs(s.th1d) + Math.abs(s.th2d);
	const lean = Math.abs(wrap(s.th1)) + Math.abs(wrap(s.th2));
	const room = 1 - Math.abs(s.x) / X_LIMIT;
	const upFactor = 1 / (1 + 2 * lean * lean);
	const railFactor = 0.25 + 0.75 * Math.min(1, room / 0.5);
	// The grade is floored, and the floor is load-bearing: ungated, a
	// folded wall-hugging arrival paid ~6 % of full price — for lucky RNG
	// streams that's a fine gradient toward cleaner arrivals, but unlucky
	// streams never earn enough from their first sloppy deliveries to learn
	// that delivering pays AT ALL (measured: seed 512 learned the swing,
	// seed 513 sat at 0/100 forever). With the floor, any delivery is worth
	// a real prize and a clean one is worth ~6× more.
	return (100 / (1 + 0.1 * spin * spin)) * Math.max(0.15, upFactor * railFactor);
}

/** Is a delivered state worth REPLAYING for brink drills? Stricter than
 * the delivery gate itself: the gate must stay loose so hanging drills can
 * still discover the bonus, but the replay buffer must not be poisoned by
 * folded or wall-pinned arrivals that no controller can catch (see
 * deliveryBonus). Training the catch on uncatchable starts caps the brink
 * win rate and freezes the α anneal — that, measured, was the whole
 * "stuck at one-in-ten catches" plateau. */
export function isPrimeDelivery(s: DpoleState): boolean {
	const lean = Math.abs(wrap(s.th1)) + Math.abs(wrap(s.th2));
	return lean < 0.9 && Math.abs(s.x) < 0.6 * X_LIMIT;
}

/** A ring of states the swing actually delivered at the hand-off height.
 * Brink episodes replay these, so the balance half practices exactly the
 * arrivals it will face — not some hand-invented distribution of "tilted
 * and moving", which turned out to be mostly uncatchable noise that washed
 * the balance weights out. The curriculum anneals itself: as the swing
 * learns calmer deliveries, the replayed starts get calmer too. */
export class DeliveryBuffer {
	private buf: DpoleState[] = [];
	private head = 0;
	constructor(private cap = 256) {}
	get size(): number {
		return this.buf.length;
	}
	push(s: DpoleState): void {
		const copy = { ...s };
		if (this.buf.length < this.cap) this.buf.push(copy);
		else {
			this.buf[this.head] = copy;
			this.head = (this.head + 1) % this.cap;
		}
	}
	sample(rand: Rand): DpoleState | null {
		if (this.buf.length === 0) return null;
		return { ...this.buf[Math.floor(rand() * this.buf.length)] };
	}
}

/** Blend a state toward standing perfectly still at the top. */
function scaleToward(s: DpoleState, a: number): DpoleState {
	return {
		x: s.x * a,
		xd: s.xd * a,
		th1: wrap(s.th1) * a,
		th1d: s.th1d * a,
		th2: wrap(s.th2) * a,
		th2d: s.th2d * a
	};
}

/** The whole training recipe in one object: the curriculum mix, the
 * delivery replay, and — the piece that finally made the catch learnable —
 * a REVERSE curriculum on the brink. Brink episodes replay delivered
 * states scaled toward vertical by α; α starts small (well inside the
 * hold's basin, so the catch succeeds and the gradient says something) and
 * anneals toward 1 (the raw delivery) only as fast as the success rate
 * allows. Without it, brink episodes fail wholesale, and a hundred
 * thousand all-failure episodes teach exactly nothing. */
/** A tumbling, often over-energized start — the state the live stage
 * actually lives in after a missed catch. Swing drills that only start at
 * rest teach a policy that can pump but never bleed: on the live stage one
 * missed catch then turns into an endless forced propeller, orbiting the
 * top at 40 rad/s and delivering nothing but fly-throughs. A slice of
 * swing drills starts mid-tumble instead, so "too much energy — brake,
 * then arrive slowly" is a practiced move, not a novel emergency. */
export function tumbleStart(rand: Rand): DpoleState {
	const u = (m: number) => (rand() * 2 - 1) * m;
	return {
		x: u(1.2),
		xd: u(2),
		th1: rand() * 2 * Math.PI,
		th1d: u(8),
		th2: rand() * 2 * Math.PI,
		th2d: u(12)
	};
}

export class DpoleCurriculum {
	readonly deliveries = new DeliveryBuffer();
	/** How much of the raw delivered state the brink replays. */
	alpha = 0.25;
	/** Recent brink success rate (caught for 2 s), EMA. */
	winRate = 0;
	/** Draw the next episode: pick a start kind, replay + scale for brink,
	 * record deliveries and successes, adapt α. */
	next(theta: Float64Array, rand: Rand): DpoleEpisode {
		const k = drawStart(rand);
		let start: DpoleState | undefined;
		if (k === 2) {
			const d = this.deliveries.sample(rand);
			// a band of difficulties up to α, not just the frontier — the
			// easy end keeps refreshing what the hard end builds on
			if (d) start = scaleToward(d, this.alpha * (0.5 + 0.5 * rand()));
		} else if (k === 0 && rand() < 0.35) {
			start = tumbleStart(rand);
		}
		const ep = runDpoleEpisode(theta, rand, k, start);
		// replay only prime deliveries — the rest are graded (poorly) by the
		// bonus but must not become the catch's training diet
		if (ep.delivered && isPrimeDelivery(ep.delivered)) this.deliveries.push(ep.delivered);
		if (ep.kind === 2 && start) {
			const win = ep.caught >= 100 ? 1 : 0;
			this.winRate += 0.05 * (win - this.winRate);
			if (this.winRate > 0.55 && this.alpha < 1) {
				this.alpha = Math.min(1, this.alpha + 0.05);
				this.winRate = 0.35; // re-earn the next notch
			} else if (this.winRate < 0.08 && this.alpha > 0.25) {
				// the retreat: when a rough patch wrecks the catch, ease the
				// starts back toward vertical and rebuild — without this, one
				// collapse is forever, because all-failure episodes teach
				// nothing to climb back on
				this.alpha = Math.max(0.25, this.alpha - 0.05);
				this.winRate = 0.35;
			}
		}
		return ep;
	}
}

/** State → feature vector, written into `out`. THREE dashboards, one
 * policy, and HARD switches between them by tip height: swing gauges below
 * the hand-off, catch gauges on the rim (0.82–0.96), hold gauges in the
 * core. Because no two blocks are ever lit together, their weights train
 * on disjoint data — three controllers in one rulebook, no fighting over
 * shared weights. This came the hard way: with swing and balance faded
 * smoothly into each other, the mid region was a compromise neither skill
 * owned; with catch and hold sharing one block, every catch lesson was a
 * hold un-lesson. */
export function dpoleFeatures(s: DpoleState, out: Float64Array): Float64Array {
	out.fill(0);
	const h01 = (tipHeight(s) + 1) / 2;
	if (h01 <= HANDOFF_H01) {
		// ── the swing dashboard ──
		out[0] = s.x / X_LIMIT;
		out[1] = s.xd / 3;
		out[2] = Math.sin(s.th1);
		out[3] = Math.cos(s.th1);
		out[4] = s.th1d / 6;
		out[5] = Math.sin(s.th2);
		out[6] = Math.cos(s.th2);
		out[7] = s.th2d / 8;
		// the energy gauges. eGap is how far the stack's mechanical energy
		// is below the top's — the fuel still missing. The pump gauges are
		// the classical energy-shaping law (push when θ̇·cos θ and the gap
		// agree) served as ready-made dials: a single positive weight on one
		// is a swing-up controller. The policy still has to learn that
		// weight, and the catch — but not to reinvent mechanics from coin
		// flips.
		const eGap = clip1((E_TOP - mechanicalEnergy(s)) / E_SPAN);
		out[8] = eGap;
		out[9] = clip1(s.th1d * Math.cos(s.th1) * 0.3) * eGap;
		out[10] = clip1(s.th2d * Math.cos(s.th2) * 0.25) * eGap;
	} else if (h01 <= CORE_H01) {
		// ── the catch dashboard: wide gauges (pegged at ±45°), the fold
		// between the links, and the energy surplus to bleed ──
		out[11] = s.x / X_LIMIT;
		out[12] = s.xd / 3;
		out[13] = clip1(wrap(s.th1) / (2.5 * TH_LIMIT));
		out[14] = clip1(wrap(s.th2) / (2.5 * TH_LIMIT));
		out[15] = clip1(s.th1d / 3.5);
		out[16] = clip1(s.th2d / 5);
		out[17] = clip1(wrap(s.th1 - s.th2) / (2.5 * TH_LIMIT));
		out[18] = clip1((mechanicalEnergy(s) - E_TOP) / (0.25 * E_SPAN));
		// cross gauges — angle × its own rate — because the catch is the one
		// moment a linear read isn't enough: leaning 20° falling IN and
		// leaning 20° falling OUT need opposite pushes, and no weighting of
		// separate angle and rate dials can tell them apart. A product dial
		// can: this is a quadratic controller exactly (and only) where the
		// job is hardest. Measured, these took the catch-per-delivery from
		// roughly one in four to one in two.
		out[19] = clip1(wrap(s.th1) * s.th1d * 1.2);
		out[20] = clip1(wrap(s.th2) * s.th2d * 0.8);
		out[21] = clip1(wrap(s.th1 - s.th2) * (s.th1d - s.th2d) * 0.8);
		out[22] = clip1(s.xd * wrap(s.th1) * 1.5);
	} else {
		// ── the hold dashboard: fine gauges pegged at ±18°, wide ones at
		// ±45° for the moments a gust (or the reader) tips it ──
		out[23] = s.x / X_LIMIT;
		out[24] = s.xd / 3;
		out[25] = clip1(wrap(s.th1) / TH_LIMIT);
		out[26] = clip1(wrap(s.th2) / TH_LIMIT);
		out[27] = clip1(s.th1d / 1.5);
		out[28] = clip1(s.th2d / 2);
		out[29] = clip1(wrap(s.th1) / (2.5 * TH_LIMIT));
		out[30] = clip1(wrap(s.th2) / (2.5 * TH_LIMIT));
		out[31] = clip1(s.th1d / 3.5);
		out[32] = clip1(s.th2d / 5);
	}
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
	/** The state a swing episode delivered at the hand-off, for the replay
	 * buffer — null for other kinds or if it never arrived. */
	delivered: DpoleState | null;
}

/** Curriculum mix: half the practice is the real journey from hanging;
 * a quarter starts balanced (learn the hold); a quarter starts at the
 * brink (learn the catch). */
export function drawStart(rand: Rand): DpoleStart {
	const r = rand();
	return r < 0.5 ? 0 : r < 0.75 ? 1 : 2;
}

/** Roll one headless episode under the current policy. The episode
 * boundaries ARE the division of labour: a hanging start ends the moment
 * the tip crosses the hand-off height, collecting the delivery bonus — it
 * trains only the swing gauges' weights and never grades the catch. Up
 * and brink starts end when the tip truly drops out — they train only the
 * balance gauges' weights. Without this cut, the two skills share one
 * time-indexed baseline, and a lucky fly-through during a swing episode
 * reinforces whatever random flailing it did up there onto the balance
 * weights — confident lessons in nonsense. */
export function runDpoleEpisode(
	theta: Float64Array,
	rand: Rand,
	kind?: DpoleStart,
	start?: DpoleState
): DpoleEpisode {
	const k = kind ?? drawStart(rand);
	const s = start ? { ...start } : resetDpole(rand, k);
	const feats = new Float64Array(DECISIONS * N_FEATURES);
	const rewards = new Float64Array(DECISIONS);
	const actions: number[] = [];
	const f = new Float64Array(N_FEATURES);
	const probs = new Float64Array(DPOLE_ACTIONS);
	let ret = 0;
	let caught = 0;
	let t = 0;
	let delivered: DpoleState | null = null;
	while (t < DECISIONS) {
		dpoleFeatures(s, f);
		feats.set(f, t * N_FEATURES);
		dpolePolicy(theta, f, probs);
		const a = sampleFrom(probs, rand);
		actions.push(a);
		let r = 0;
		for (let j = 0; j < ACTION_REPEAT; j++) {
			physicsStep(s, actionForce(a));
			r += dpoleReward(s) / ACTION_REPEAT;
			if (isUpright(s)) caught++;
		}
		if (k === 0 && isDelivery(s)) {
			rewards[t] = r + deliveryBonus(s);
			ret += rewards[t];
			t++;
			delivered = { ...s };
			break;
		}
		rewards[t] = r;
		ret += r;
		t++;
		if (k !== 0 && hasDroppedOut(s)) break;
	}
	return { feats, actions, rewards, steps: t, kind: k, ret, caught, delivered };
}

export const DPOLE_BASELINE_LR = 0.1;
/** Faint weight decay per update — a hedge against runaway logits, and no
 * more. At 1e-3 it was an eroder: once the baselines converge and the
 * advantages go quiet, a decay the gradient can't outrun melts a perfected
 * policy back into mush over tens of thousands of updates. */
export const DPOLE_DECAY = 1e-4;
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
			// clamp the whitened advantage: when the curriculum shifts, the
			// first episodes under the new regime can be many RMS's from the
			// running baseline, and one unclamped mega-update saturates the
			// softmax into a wrong answer it never recovers from
			const w = Math.max(-3, Math.min(3, adv[t] / scale)) / eps.length;
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

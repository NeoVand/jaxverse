// The swarm: what happens *between* the practice halls.
//
// Each hall is a complete REINFORCE learner — its own θ, RNG, baseline,
// curriculum — and its own gradient is first-order in its own trajectories:
// "given what I just did, push my weights this way." That gradient knows
// nothing about the other five learners. Everything here is the second
// layer, the part that exists only because there are several of them.
//
// Three channels, deliberately different in kind:
//
//   EXPERIENCE goes to everyone. A hall that lands a delivery cleaner than
//   any it has managed announces the state, and every other hall files it on
//   a small guest shelf to practise from. A place to start is not an opinion
//   about what to do, so this costs nobody their independence — but it is
//   kept off their own replay buffers, because a pool sharing into one
//   buffer learns to catch the leader's arrivals and never its own.
//
//   GRADIENTS go downhill. The 165 numbers behind a find are posted to the
//   halls doing worse, who step a fraction of the way along them.
//
//   WEIGHTS go wherever a hall decides. This file only supplies the PRIOR —
//   a capped ranking of the peers who are both ahead and improving — and
//   each hall's own social learner (in the worker) decides on top of it
//   whether to take anyone's advice at all, graded by whether it climbed the
//   pool's ranking afterwards.
//
// That last channel is a correction, not a flourish. Influence applied
// indiscriminately was measured to make the pool fail TOGETHER: mean pool
// fitness 0.33, against 0.82 for six halls sharing nothing, because a hall
// mid-discovery that keeps being dragged toward the leader never finishes
// its own idea. REINFORCE discovery lives on a stream compounding its own
// lucky exploration, and six streams averaged into one compound nobody's.
// The race IS the insurance; a hall that can refuse is a hall that can keep
// its share of it.
//
// The prior leans on how fast a peer is *improving* (`rise`, the distance
// its fitness has opened above its own slow average) as much as on how good
// it currently is — so the hall that just found something is the one the
// others are inclined toward, seconds before it is nominally the champion.

export interface SwarmOpts {
	/** How far behind the leader a hall must be (as a fraction of the
	 * leader's fitness) before any weight drift reaches it at all. Inside
	 * this band the halls are genuinely racing and stay independent. */
	gate: number;
	/** Pull applied to a hopeless hall, per pull round. */
	maxPull: number;
	/** Softmax temperature on the fitness gap between peers. */
	tau: number;
	/** How loudly "improving right now" speaks next to "already good". */
	riseWeight: number;
	/** Ceiling on a prior logit, so the instinct ranks the peers without
	 * drowning out what a hall has learned about being influenced. */
	priorCap: number;
}

export const SWARM: SwarmOpts = {
	gate: 0.35,
	maxPull: 0.18,
	tau: 0.12,
	riseWeight: 12,
	priorCap: 1.6
};

/** How hard hall `score` gets dragged toward the halls ahead of it, given
 * the leader's `best`. Zero for the leader and for anyone still in the
 * race; ramping to `maxPull` for a hall that has learned nothing. The pull
 * is self-limiting — once a laggard's θ has drifted onto the blend, the
 * difference it lerps along is zero, so there is no thrash to damp. */
export function pullStrength(score: number, best: number, o: SwarmOpts = SWARM): number {
	const norm = Math.max(best, 0.05);
	const gap = Math.max(0, (best - score) / norm);
	if (gap <= o.gate) return 0;
	return o.maxPull * Math.min(1, (gap - o.gate) / (1 - o.gate));
}

/**
 * Hall `w`'s attention over its peers, written into `out` (length = number
 * of halls). Mass only ever lands on halls that are strictly ahead; the row
 * sums to 1, or to 0 when nobody is ahead — which is exactly the leader's
 * case, and the caller should then skip the pull entirely.
 */
export function attention(
	scores: readonly number[],
	rises: readonly number[],
	w: number,
	out: Float64Array,
	o: SwarmOpts = SWARM
): number {
	const n = scores.length;
	out.fill(0);
	let top = -Infinity;
	for (let j = 0; j < n; j++) {
		if (j === w || scores[j] <= scores[w]) continue;
		const z = (scores[j] - scores[w]) / o.tau + o.riseWeight * rises[j];
		if (z > top) top = z;
	}
	if (top === -Infinity) return 0;
	let sum = 0;
	for (let j = 0; j < n; j++) {
		if (j === w || scores[j] <= scores[w]) continue;
		const e = Math.exp((scores[j] - scores[w]) / o.tau + o.riseWeight * rises[j] - top);
		out[j] = e;
		sum += e;
	}
	for (let j = 0; j < n; j++) out[j] /= sum;
	return 1;
}

/**
 * The same logits `attention` softmaxes, handed over raw — the PRIOR each
 * hall starts from before it has any social experience of its own. A hall
 * learns a correction on top of these (see the social layer in the worker),
 * so the pool begins with the sensible instinct "listen to whoever is ahead
 * and moving" and then finds out, from its own rank, whether that instinct
 * pays. `out` gets one entry per hall; peers that are not ahead get
 * −Infinity, which no learned correction can rescue: a hall never takes
 * advice from someone doing worse than it is.
 */
export function priorLogits(
	scores: readonly number[],
	rises: readonly number[],
	w: number,
	out: Float64Array,
	o: SwarmOpts = SWARM
): void {
	for (let j = 0; j < scores.length; j++) {
		if (j === w || scores[j] <= scores[w]) {
			out[j] = -Infinity;
			continue;
		}
		// Capped, and the cap is the point. Uncapped, a hall twice as good as
		// you scores a logit near ten, which no amount of social experience
		// can argue with inside a session — the prior would BE the policy and
		// the learned part would be decoration. Held to a couple of nats it
		// ranks the peers, and what the hall learns about being influenced
		// decides whether it listens to any of them.
		const z = (scores[j] - scores[w]) / o.tau + o.riseWeight * rises[j];
		out[j] = Math.min(o.priorCap, z);
	}
}

/** The blended target: Σⱼ αⱼ·θⱼ, written into `out`. */
export function blendInto(
	out: Float64Array,
	thetas: readonly Float64Array[],
	weights: Float64Array
): void {
	out.fill(0);
	for (let j = 0; j < thetas.length; j++) {
		const a = weights[j];
		if (a === 0) continue;
		const t = thetas[j];
		for (let i = 0; i < out.length; i++) out[i] += a * t[i];
	}
}

/** Fitness measured against a hall's own slow average: how far it has just
 * pulled above its recent self. A hall that has been good for a long time
 * has a rise of ~0; a hall that just found something has a spike.
 *
 * It has to be a good deal slower than the fitness it is subtracting, or the
 * two averages track each other and every rise is zero. Fitness now settles
 * over about a second; this settles over several. */
export const RISE_EMA = 0.004;

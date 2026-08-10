// Learning a reward from comparisons, and then optimizing against it.
//
// Everything the "Teaching Taste" chapter computes lives here, in plain
// TypeScript, because none of it is big enough to be worth a GPU: an
// ornament is six numbers, the judge fitted to your clicks is four hundred,
// and the policy optimized against that judge is twelve. The chapter's whole
// argument is that the *smallness* is not the point — a proxy with four
// hundred parameters is gamed by exactly the mechanism that games one with
// four hundred billion.
//
// Three objects, in the order the chapter meets them:
//
//   1. a GENE — six numbers in [0,1] that draw one ornament;
//   2. a JUDGE — a small MLP fitted to pairwise verdicts by Bradley–Terry,
//      the same maximum-likelihood fit that turns chess results into Elo;
//   3. a POLICY — a Gaussian over ornaments, ascended against the judge under
//      a KL leash back to the reference the comparisons were drawn from.
//
// The third is the RLHF objective with a policy class small enough that the
// KL has a closed form, so the leash in this chapter is the real thing rather
// than an illustration of one.

import type { Rand } from './rng';

// ── 1 · the gene ───────────────────────────────────────────────────────────

export const N_GENES = 6;

export interface GeneMeta {
	key: string;
	/** What the reader sees in an axis picker. */
	label: string;
	/** One line, for the tooltip — what moving this gene does to the drawing. */
	note: string;
}

/** The six dials that draw an ornament, in the order they sit in the vector. */
export const GENES: readonly GeneMeta[] = [
	{ key: 'arms', label: 'arms', note: 'how many petals the rosette repeats around its centre' },
	{
		key: 'curl',
		label: 'curl',
		note: 'how far each petal spirals, from swept back to swept forward'
	},
	{ key: 'reach', label: 'reach', note: 'how far the petals extend past the medallion' },
	{ key: 'weight', label: 'weight', note: 'stroke thickness — the ink the engraver spends' },
	{ key: 'rings', label: 'rings', note: 'how many concentric rings are set inside the flower' },
	{ key: 'ink', label: 'ink', note: 'the colour, from ultramarine through to vermilion' }
];

/** A drawable ornament: six numbers, each in [0,1]. */
export type Gene = Float64Array;

/**
 * The chapter works in an unconstrained space u ∈ ℝ⁶ and squashes to genes
 * with a logistic, so that a *Gaussian* policy over u is a well-behaved
 * distribution over ornaments — no walls to bump into, and a KL that closes.
 */
export function geneFromU(u: Float64Array, out?: Gene): Gene {
	const g = out ?? new Float64Array(N_GENES);
	for (let i = 0; i < N_GENES; i++) g[i] = 1 / (1 + Math.exp(-u[i]));
	return g;
}

/** The inverse, for placing a known ornament back into policy space. */
export function uFromGene(g: Gene, out?: Float64Array): Float64Array {
	const u = out ?? new Float64Array(N_GENES);
	for (let i = 0; i < N_GENES; i++) {
		const p = Math.min(1 - 1e-6, Math.max(1e-6, g[i]));
		u[i] = Math.log(p / (1 - p));
	}
	return u;
}

/**
 * The reference distribution: the pile every comparison was drawn from, and
 * the thing the leash later measures drift against.
 *
 * The width is load-bearing in two places, so it is worth the paragraph. Too
 * wide and the logit-normal flattens into something indistinguishable from
 * uniform — π_ref draws as a plateau, the leash plate's "boost, don't replace"
 * picture has nothing to boost, and, worse, the optimizer can never reach
 * anywhere the judge has not already seen data, so over-optimization has no
 * room to happen. Too narrow and every ornament looks like every other and
 * there is nothing to have an opinion about. At 1.15 the density is a visible
 * bell, 95% of each gene lands in [0.09, 0.91], and the corners of the space
 * are reachable but unvisited — which is exactly the situation preference data
 * is collected in.
 */
export const REF_SIGMA = 1.15;

export function sampleRefU(rand: Rand, out?: Float64Array): Float64Array {
	const u = out ?? new Float64Array(N_GENES);
	for (let i = 0; i < N_GENES; i++) u[i] = REF_SIGMA * gauss(rand);
	return u;
}

export function sampleRefGene(rand: Rand): Gene {
	return geneFromU(sampleRefU(rand));
}

/** Box–Muller, one draw at a time — a standard normal from a uniform Rand. */
export function gauss(rand: Rand): number {
	let a = rand();
	if (a < 1e-12) a = 1e-12;
	return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * rand());
}

// ── 2 · the judge ──────────────────────────────────────────────────────────

/** Hidden width. Small on purpose: 401 parameters, fitted to a few dozen clicks. */
export const JUDGE_H = 16;

export interface Judge {
	/** [H, 6] */ w1: Float64Array;
	/** [H] */ b1: Float64Array;
	/** [H, H] */ w2: Float64Array;
	/** [H] */ b2: Float64Array;
	/** [H] */ w3: Float64Array;
	b3: number;
	/**
	 * Where the judge's scores sit once fitted, measured over the reference
	 * distribution. Bradley–Terry pins down score *differences* and nothing
	 * else, so an unstandardized r has an arbitrary origin and scale — and β,
	 * the leash length, is priced in those units. Standardizing is what makes
	 * a β of 1 mean the same thing from one reader to the next.
	 */
	mean: number;
	std: number;
}

export const JUDGE_N_PARAMS =
	JUDGE_H * N_GENES + JUDGE_H + JUDGE_H * JUDGE_H + JUDGE_H + JUDGE_H + 1;

export function createJudge(rand: Rand): Judge {
	// Xavier-ish: keep the initial score field flat, so the taste map starts
	// honestly featureless rather than showing structure nobody voted for.
	const fill = (n: number, fanIn: number) => {
		const a = new Float64Array(n);
		const s = Math.sqrt(1 / fanIn);
		for (let i = 0; i < n; i++) a[i] = (rand() * 2 - 1) * s;
		return a;
	};
	return {
		w1: fill(JUDGE_H * N_GENES, N_GENES),
		b1: new Float64Array(JUDGE_H),
		w2: fill(JUDGE_H * JUDGE_H, JUDGE_H),
		b2: new Float64Array(JUDGE_H),
		w3: fill(JUDGE_H, JUDGE_H),
		b3: 0,
		mean: 0,
		std: 1
	};
}

/** Scratch space, so scoring a 120 × 96 taste map allocates nothing. */
interface Act {
	x: Float64Array;
	h1: Float64Array;
	h2: Float64Array;
}
const act = (): Act => ({
	x: new Float64Array(N_GENES),
	h1: new Float64Array(JUDGE_H),
	h2: new Float64Array(JUDGE_H)
});

/** Genes live in [0,1]; the network prefers them centred. */
function inputOf(g: Gene, x: Float64Array): Float64Array {
	for (let i = 0; i < N_GENES; i++) x[i] = g[i] * 2 - 1;
	return x;
}

function forwardInto(j: Judge, g: Gene, a: Act): number {
	inputOf(g, a.x);
	for (let k = 0; k < JUDGE_H; k++) {
		let s = j.b1[k];
		for (let i = 0; i < N_GENES; i++) s += j.w1[k * N_GENES + i] * a.x[i];
		a.h1[k] = Math.tanh(s);
	}
	for (let k = 0; k < JUDGE_H; k++) {
		let s = j.b2[k];
		for (let i = 0; i < JUDGE_H; i++) s += j.w2[k * JUDGE_H + i] * a.h1[i];
		a.h2[k] = Math.tanh(s);
	}
	let out = j.b3;
	for (let k = 0; k < JUDGE_H; k++) out += j.w3[k] * a.h2[k];
	return out;
}

const scratch = act();

/** The judge's raw opinion of one ornament. */
export function rawScore(j: Judge, g: Gene): number {
	return forwardInto(j, g, scratch);
}

/** The same opinion in standardized units — what β is priced in. */
export function score(j: Judge, g: Gene): number {
	return (forwardInto(j, g, scratch) - j.mean) / j.std;
}

/** ∂r/∂gene, by hand — the pathwise gradient the optimizer climbs. */
export function scoreGrad(j: Judge, g: Gene, out?: Float64Array): Float64Array {
	const a = scratch;
	forwardInto(j, g, a);
	const d2 = new Float64Array(JUDGE_H);
	for (let k = 0; k < JUDGE_H; k++) d2[k] = j.w3[k] * (1 - a.h2[k] * a.h2[k]);
	const d1 = new Float64Array(JUDGE_H);
	for (let i = 0; i < JUDGE_H; i++) {
		let s = 0;
		for (let k = 0; k < JUDGE_H; k++) s += d2[k] * j.w2[k * JUDGE_H + i];
		d1[i] = s * (1 - a.h1[i] * a.h1[i]);
	}
	const grad = out ?? new Float64Array(N_GENES);
	for (let i = 0; i < N_GENES; i++) {
		let s = 0;
		for (let k = 0; k < JUDGE_H; k++) s += d1[k] * j.w1[k * N_GENES + i];
		// chain through inputOf's ×2, and report in standardized units
		grad[i] = (2 * s) / j.std;
	}
	return grad;
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** One recorded verdict: the reader saw two ornaments and chose the winner. */
export interface Pair {
	winner: Gene;
	loser: Gene;
}

/**
 * P(the reader picks `a` over `b`), under this judge. Bradley–Terry: the
 * chance depends on nothing whatsoever except the gap between the two scores,
 * squashed. Elo, with ornaments where the chess players usually go.
 */
export function preferProb(j: Judge, a: Gene, b: Gene): number {
	return sigmoid(rawScore(j, a) - rawScore(j, b));
}

interface AdamState {
	m: Float64Array;
	v: Float64Array;
	t: number;
}

/** Every parameter of the judge, viewed as one flat list — Adam wants a vector. */
function flatViews(j: Judge): Float64Array[] {
	return [j.w1, j.b1, j.w2, j.b2, j.w3];
}

export interface FitResult {
	/** Mean pairwise loss at the end of the fit. */
	loss: number;
	/** Share of the training pairs the judge now ranks the way the reader did. */
	fit: number;
}

/**
 * Fit the judge to the pile of verdicts by maximum likelihood: turn the knobs
 * until the comparisons the reader actually made come out as unsurprising as
 * possible. Full-batch Adam — a few dozen pairs is not a dataset that needs
 * minibatching, and determinism makes the chapter's numbers reproducible.
 *
 * Weight decay is doing real work here and the chapter says so: four hundred
 * parameters fitted to thirty clicks will happily invent structure nobody
 * voted for, and the invented structure is precisely what the optimizer of
 * the next plate goes hunting for.
 */
export function fitJudge(
	j: Judge,
	pairs: readonly Pair[],
	opts: { steps?: number; lr?: number; decay?: number; refSample?: readonly Gene[] } = {}
): FitResult {
	// Measured, not guessed: swept over decay ∈ [1e-3, 6e-2] and steps ∈ {200,
	// 320} against a stand-in reader, scoring both held-out pairs and the
	// prequential calls the plate actually reports. This corner won both, and
	// the whole surface is flat to about a point, which is itself worth
	// knowing — the judge's competence here is set by how many comparisons it
	// has, not by how hard it was tuned.
	const steps = opts.steps ?? 200;
	const lr = opts.lr ?? 0.05;
	const decay = opts.decay ?? 6e-3;
	if (pairs.length === 0) return { loss: Math.LN2, fit: 0.5 };

	const parts = flatViews(j);
	const grads = parts.map((p) => new Float64Array(p.length));
	const states: AdamState[] = parts.map((p) => ({
		m: new Float64Array(p.length),
		v: new Float64Array(p.length),
		t: 0
	}));
	// j.b3 is deliberately never trained, and the reason is worth knowing:
	// Bradley–Terry only ever sees a *difference* of two scores, so an additive
	// constant on every score cancels exactly. The final bias is unidentifiable
	// from preference data — its gradient is identically zero — which is the
	// same fact that lets DPO's partition function vanish two plates later.

	const aw = act();
	const al = act();
	let loss = 0;

	for (let step = 0; step < steps; step++) {
		for (const g of grads) g.fill(0);
		loss = 0;

		for (const pair of pairs) {
			// Two forward passes into separate activation buffers, because the
			// backward pass needs both sides of the comparison at once.
			const rw = forwardInto(j, pair.winner, aw);
			const rl = forwardInto(j, pair.loser, al);

			const p = sigmoid(rw - rl);
			loss += -Math.log(Math.max(1e-12, p));
			// d/d(margin) of −log σ(margin) is −(1 − σ) = σ − 1
			const dm = p - 1;

			// the winner pushes with +dm, the loser with −dm: one backward pass
			// per side, with the sign of the margin flipped between them
			accumulate(j, grads, aw, dm);
			accumulate(j, grads, al, -dm);
		}

		const n = pairs.length;
		loss /= n;
		for (const g of grads) for (let i = 0; i < g.length; i++) g[i] /= n;

		for (let pi = 0; pi < parts.length; pi++) {
			adamStep(parts[pi], grads[pi], states[pi], lr, pi % 2 === 0 ? decay : 0);
		}
	}

	// Standardize, so β means the same thing for every reader.
	const sample = opts.refSample;
	if (sample && sample.length > 1) {
		let m = 0;
		for (const g of sample) m += forwardInto(j, g, aw);
		m /= sample.length;
		let s = 0;
		for (const g of sample) {
			const d = forwardInto(j, g, aw) - m;
			s += d * d;
		}
		j.mean = m;
		j.std = Math.max(1e-6, Math.sqrt(s / sample.length));
	}

	let right = 0;
	for (const pair of pairs) if (rawScore(j, pair.winner) > rawScore(j, pair.loser)) right++;
	return { loss, fit: right / pairs.length };
}

/** Backprop one side of a comparison into the gradient buffers, weighted by dm. */
function accumulate(j: Judge, grads: Float64Array[], a: Act, dm: number): void {
	const { x, h1, h2 } = a;
	const [gw1, gb1, gw2, gb2, gw3] = grads;
	const d2 = new Float64Array(JUDGE_H);
	for (let k = 0; k < JUDGE_H; k++) {
		gw3[k] += dm * h2[k];
		d2[k] = dm * j.w3[k] * (1 - h2[k] * h2[k]);
		gb2[k] += d2[k];
	}
	const d1 = new Float64Array(JUDGE_H);
	for (let k = 0; k < JUDGE_H; k++)
		for (let i = 0; i < JUDGE_H; i++) gw2[k * JUDGE_H + i] += d2[k] * h1[i];
	for (let i = 0; i < JUDGE_H; i++) {
		let s = 0;
		for (let k = 0; k < JUDGE_H; k++) s += d2[k] * j.w2[k * JUDGE_H + i];
		d1[i] = s * (1 - h1[i] * h1[i]);
		gb1[i] += d1[i];
	}
	for (let k = 0; k < JUDGE_H; k++)
		for (let i = 0; i < N_GENES; i++) gw1[k * N_GENES + i] += d1[k] * x[i];
}

function adamStep(p: Float64Array, g: Float64Array, s: AdamState, lr: number, decay: number): void {
	s.t++;
	const b1 = 0.9;
	const b2 = 0.999;
	const c1 = 1 - Math.pow(b1, s.t);
	const c2 = 1 - Math.pow(b2, s.t);
	for (let i = 0; i < p.length; i++) {
		const gi = g[i] + decay * p[i];
		s.m[i] = b1 * s.m[i] + (1 - b1) * gi;
		s.v[i] = b2 * s.v[i] + (1 - b2) * gi * gi;
		p[i] -= (lr * (s.m[i] / c1)) / (Math.sqrt(s.v[i] / c2) + 1e-8);
	}
}

// ── 3 · the policy, and the leash ──────────────────────────────────────────

/**
 * A diagonal Gaussian over ornament space. Twelve numbers: where to aim, and
 * how loosely. Initialized *as* the reference, so "before optimization" and
 * "the distribution the comparisons came from" are the same object.
 */
export interface Policy {
	mu: Float64Array;
	/** log σ, so the spread can never go negative however hard we push it. */
	logSigma: Float64Array;
}

export function createPolicy(): Policy {
	return {
		mu: new Float64Array(N_GENES),
		logSigma: new Float64Array(N_GENES).fill(Math.log(REF_SIGMA))
	};
}

export function clonePolicy(p: Policy): Policy {
	return { mu: Float64Array.from(p.mu), logSigma: Float64Array.from(p.logSigma) };
}

export function samplePolicyU(p: Policy, rand: Rand, out?: Float64Array): Float64Array {
	const u = out ?? new Float64Array(N_GENES);
	for (let i = 0; i < N_GENES; i++) u[i] = p.mu[i] + Math.exp(p.logSigma[i]) * gauss(rand);
	return u;
}

export function samplePolicyGene(p: Policy, rand: Rand): Gene {
	return geneFromU(samplePolicyU(p, rand));
}

/**
 * KL(π ‖ π_ref) in nats, closed form for two diagonal Gaussians. This is the
 * chapter's x-axis: how far the policy has travelled from the distribution its
 * training data came from — which is the only honest ruler for "how hard did
 * we optimize", and the one Gao et al. measured over-optimization against.
 *
 * Note the ordering, because it is the whole safety property: the average is
 * taken under π, our own behaviour, so the penalty only ever sees the places
 * we actually go.
 */
export function klToRef(p: Policy): number {
	let kl = 0;
	const s0 = REF_SIGMA;
	for (let i = 0; i < N_GENES; i++) {
		const s = Math.exp(p.logSigma[i]);
		kl += Math.log(s0 / s) + (s * s + p.mu[i] * p.mu[i]) / (2 * s0 * s0) - 0.5;
	}
	return kl;
}

export interface AscendOpts {
	/** Leash length. 0 is the unleashed optimizer of the Goodhart plate. */
	beta: number;
	/** Draws per step — the expectation is estimated, like everything here. */
	samples?: number;
	lr?: number;
}

/**
 * One optimizer step on  E_{g∼π}[ r(g) ] − β · KL(π ‖ π_ref).
 *
 * The expectation is differentiated by the *reparameterization* trick rather
 * than by REINFORCE: draw the noise first, write u = μ + σ·ε, and the sample
 * becomes an ordinary differentiable function of the knobs. That is available
 * here precisely because the judge is a network we can differentiate — the
 * luxury a language model, whose actions are discrete tokens, never gets.
 */
export function ascendStep(j: Judge, p: Policy, rand: Rand, opts: AscendOpts): void {
	const n = opts.samples ?? 16;
	const lr = opts.lr ?? 0.06;
	const beta = opts.beta;
	const gMu = new Float64Array(N_GENES);
	const gLs = new Float64Array(N_GENES);
	const u = new Float64Array(N_GENES);
	const gene = new Float64Array(N_GENES);
	const dr = new Float64Array(N_GENES);
	const eps = new Float64Array(N_GENES);

	for (let s = 0; s < n; s++) {
		for (let i = 0; i < N_GENES; i++) {
			eps[i] = gauss(rand);
			u[i] = p.mu[i] + Math.exp(p.logSigma[i]) * eps[i];
		}
		geneFromU(u, gene);
		scoreGrad(j, gene, dr);
		for (let i = 0; i < N_GENES; i++) {
			// chain through the logistic squash: dg/du = g(1−g)
			const dgdu = gene[i] * (1 - gene[i]);
			const d = dr[i] * dgdu;
			gMu[i] += d;
			gLs[i] += d * eps[i] * Math.exp(p.logSigma[i]);
		}
	}

	const s0sq = REF_SIGMA * REF_SIGMA;
	// Scaling the whole objective by 1/(1+β) moves the optimum nowhere — it is
	// a step-size policy, not a change of problem. Without it a short leash is
	// a stiff spring, and one fixed stride across a stiff spring is the
	// Prologue's divergence: the step outruns the valley it is measuring, and
	// the reader's β slider explodes instead of tightening.
	const k = lr / (1 + beta);
	for (let i = 0; i < N_GENES; i++) {
		const sig = Math.exp(p.logSigma[i]);
		// −β ∇KL, differentiated in closed form
		const dklMu = p.mu[i] / s0sq;
		const dklLs = (sig * sig) / s0sq - 1;
		p.mu[i] += k * (gMu[i] / n - beta * dklMu);
		p.logSigma[i] += k * (gLs[i] / n - beta * dklLs);
		// keep the spread in a sane band: a policy that collapses to a point
		// stops being a distribution, and one that explodes stops being a policy
		p.logSigma[i] = Math.max(Math.log(0.04), Math.min(Math.log(3.2), p.logSigma[i]));
	}
}

/** Expected standardized score under π, estimated the only way available. */
export function expectedScore(j: Judge, p: Policy, rand: Rand, n = 96): number {
	let s = 0;
	for (let i = 0; i < n; i++) s += score(j, samplePolicyGene(p, rand));
	return s / n;
}

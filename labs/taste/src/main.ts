// Lab 7 — teaching taste, and then losing it.
//
// The chapter's argument, condensed and run against a stand-in reader so it
// needs nobody's clicks. A hidden "true taste" scores six-number ornaments;
// comparisons are drawn from it; a Bradley–Terry judge is fitted to those
// comparisons and to nothing else; and then an optimizer is turned loose on
// the judge. The judge's own score climbs forever. The hidden taste — which is
// what anyone actually wanted, and which the optimizer never sees — rises,
// peaks, and falls.
//
// That gap is the whole chapter, and it is about a hundred lines of arithmetic.
// Plain loops on plain arrays: ZERO dependencies beyond vite.

const out = document.getElementById('out') as HTMLPreElement;

const G = 6; // genes per ornament
const H = 16; // hidden width of the judge

let seed = 20240;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0), seed / 4294967296);
const gauss = () => {
	const a = Math.max(1e-12, rand());
	return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * rand());
};

// ── the reference: where ornaments come from, and what the leash measures ────
const REF_SIGMA = 1.15;
const squash = (u: Float64Array) => {
	const g = new Float64Array(G);
	for (let i = 0; i < G; i++) g[i] = 1 / (1 + Math.exp(-u[i]));
	return g;
};
const sampleRef = () => {
	const u = new Float64Array(G);
	for (let i = 0; i < G; i++) u[i] = REF_SIGMA * gauss();
	return squash(u);
};

// ── the hidden taste. The judge never sees this; only comparisons drawn from
// it. Change it and everything downstream changes with it. ──────────────────
const TRUE_BEST = [0.28, 0.62, 0.8, 0.22, 0.4, 0.55];
function trueTaste(g: Float64Array): number {
	let s = 0;
	for (let i = 0; i < G; i++) s -= (g[i] - TRUE_BEST[i]) ** 2;
	return s * 3;
}

// ── the judge: 6 → 16 → 16 → 1, tanh, 401 parameters ────────────────────────
const w1 = new Float64Array(H * G);
const b1 = new Float64Array(H);
const w2 = new Float64Array(H * H);
const b2 = new Float64Array(H);
const w3 = new Float64Array(H);
const b3 = 0;
for (let i = 0; i < w1.length; i++) w1[i] = (rand() * 2 - 1) / Math.sqrt(G);
for (let i = 0; i < w2.length; i++) w2[i] = (rand() * 2 - 1) / Math.sqrt(H);
for (let i = 0; i < w3.length; i++) w3[i] = (rand() * 2 - 1) / Math.sqrt(H);

const x = new Float64Array(G);
const h1 = new Float64Array(H);
const h2 = new Float64Array(H);

function judge(g: Float64Array): number {
	for (let i = 0; i < G; i++) x[i] = g[i] * 2 - 1;
	for (let k = 0; k < H; k++) {
		let s = b1[k];
		for (let i = 0; i < G; i++) s += w1[k * G + i] * x[i];
		h1[k] = Math.tanh(s);
	}
	for (let k = 0; k < H; k++) {
		let s = b2[k];
		for (let i = 0; i < H; i++) s += w2[k * H + i] * h1[i];
		h2[k] = Math.tanh(s);
	}
	let o = b3;
	for (let k = 0; k < H; k++) o += w3[k] * h2[k];
	return o;
}

/** ∂judge/∂gene, by hand — what the optimizer climbs. */
function judgeGrad(g: Float64Array): Float64Array {
	judge(g);
	const d2 = new Float64Array(H);
	for (let k = 0; k < H; k++) d2[k] = w3[k] * (1 - h2[k] * h2[k]);
	const d1 = new Float64Array(H);
	for (let i = 0; i < H; i++) {
		let s = 0;
		for (let k = 0; k < H; k++) s += d2[k] * w2[k * H + i];
		d1[i] = s * (1 - h1[i] * h1[i]);
	}
	const grad = new Float64Array(G);
	for (let i = 0; i < G; i++) {
		let s = 0;
		for (let k = 0; k < H; k++) s += d1[k] * w1[k * G + i];
		grad[i] = 2 * s;
	}
	return grad;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

// ── fitting: Bradley–Terry by maximum likelihood, full-batch Adam ────────────
type Pair = { w: Float64Array; l: Float64Array };

const P = [w1, b1, w2, b2, w3];
const Mm = P.map((p) => new Float64Array(p.length));
const Vv = P.map((p) => new Float64Array(p.length));
let adamT = 0;
// b3 is never trained on purpose: Bradley–Terry only sees a *difference* of two
// scores, so an additive constant on every score cancels exactly. The final
// bias is unidentifiable from preference data — its gradient is identically
// zero — which is the same fact that makes DPO's partition function vanish.

function fit(pairs: Pair[], steps = 200, lr = 0.05, decay = 6e-3): number {
	const Gr = P.map((p) => new Float64Array(p.length));
	let loss = 0;
	for (let step = 0; step < steps; step++) {
		for (const g of Gr) g.fill(0);
		loss = 0;
		for (const pair of pairs) {
			const rw = judge(pair.w);
			const xw = Float64Array.from(x);
			const aw = Float64Array.from(h1);
			const bw = Float64Array.from(h2);
			const rl = judge(pair.l);
			const p = sigmoid(rw - rl);
			loss += -Math.log(Math.max(1e-12, p));
			const dm = p - 1; // d/d(margin) of −log σ(margin)
			back(Gr, xw, aw, bw, dm);
			back(Gr, x, h1, h2, -dm);
		}
		const n = pairs.length;
		loss /= n;
		for (const g of Gr) for (let i = 0; i < g.length; i++) g[i] /= n;

		adamT++;
		const c1 = 1 - Math.pow(0.9, adamT);
		const c2 = 1 - Math.pow(0.999, adamT);
		for (let pi = 0; pi < P.length; pi++) {
			const p = P[pi];
			const g = Gr[pi];
			const m = Mm[pi];
			const v = Vv[pi];
			const wd = pi % 2 === 0 ? decay : 0; // weights decay, biases don't
			for (let i = 0; i < p.length; i++) {
				const gi = g[i] + wd * p[i];
				m[i] = 0.9 * m[i] + 0.1 * gi;
				v[i] = 0.999 * v[i] + 0.001 * gi * gi;
				p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
			}
		}
	}
	return loss;
}

function back(
	Gr: Float64Array[],
	xs: Float64Array,
	a1: Float64Array,
	a2: Float64Array,
	dm: number
) {
	const [gw1, gb1, gw2, gb2, gw3] = Gr;
	const d2 = new Float64Array(H);
	for (let k = 0; k < H; k++) {
		gw3[k] += dm * a2[k];
		d2[k] = dm * w3[k] * (1 - a2[k] * a2[k]);
		gb2[k] += d2[k];
	}
	for (let k = 0; k < H; k++) for (let i = 0; i < H; i++) gw2[k * H + i] += d2[k] * a1[i];
	for (let i = 0; i < H; i++) {
		let s = 0;
		for (let k = 0; k < H; k++) s += d2[k] * w2[k * H + i];
		const d = s * (1 - a1[i] * a1[i]);
		gb1[i] += d;
		for (let j = 0; j < G; j++) gw1[i * G + j] += d * xs[j];
	}
}

// ── the policy: a diagonal Gaussian over ornaments, and its leash ────────────
type Policy = { mu: Float64Array; ls: Float64Array };
const newPolicy = (): Policy => ({
	mu: new Float64Array(G),
	ls: new Float64Array(G).fill(Math.log(REF_SIGMA))
});

/** KL(π ‖ π_ref) in nats — closed form for two diagonal Gaussians. */
function kl(p: Policy): number {
	let k = 0;
	for (let i = 0; i < G; i++) {
		const s = Math.exp(p.ls[i]);
		k += Math.log(REF_SIGMA / s) + (s * s + p.mu[i] * p.mu[i]) / (2 * REF_SIGMA ** 2) - 0.5;
	}
	return k;
}

/** One reparameterized step on E[judge] − β·KL. Dice rolled before the knobs. */
function ascend(p: Policy, beta: number, n = 16, lr = 0.06) {
	const gMu = new Float64Array(G);
	const gLs = new Float64Array(G);
	for (let s = 0; s < n; s++) {
		const eps = new Float64Array(G);
		const u = new Float64Array(G);
		for (let i = 0; i < G; i++) {
			eps[i] = gauss();
			u[i] = p.mu[i] + Math.exp(p.ls[i]) * eps[i];
		}
		const g = squash(u);
		const dr = judgeGrad(g);
		for (let i = 0; i < G; i++) {
			const d = dr[i] * g[i] * (1 - g[i]); // chain through the squash
			gMu[i] += d;
			gLs[i] += d * eps[i] * Math.exp(p.ls[i]);
		}
	}
	const k = lr / (1 + beta); // step-size policy, not a new objective
	for (let i = 0; i < G; i++) {
		const sig = Math.exp(p.ls[i]);
		p.mu[i] += k * (gMu[i] / n - (beta * p.mu[i]) / REF_SIGMA ** 2);
		p.ls[i] += k * (gLs[i] / n - beta * ((sig * sig) / REF_SIGMA ** 2 - 1));
		p.ls[i] = Math.max(Math.log(0.04), Math.min(Math.log(3.2), p.ls[i]));
	}
}

const sampleFrom = (p: Policy) => {
	const u = new Float64Array(G);
	for (let i = 0; i < G; i++) u[i] = p.mu[i] + Math.exp(p.ls[i]) * gauss();
	return squash(u);
};
const meanOf = (p: Policy, f: (g: Float64Array) => number, n = 400) => {
	let s = 0;
	for (let i = 0; i < n; i++) s += f(sampleFrom(p));
	return s / n;
};

// ── running it ───────────────────────────────────────────────────────────────
const STOPS = [0, 0.5, 1, 2, 4, 8, 16];
const bar = (v: number, lo: number, hi: number, width = 22) =>
	'█'.repeat(Math.max(0, Math.round(((v - lo) / (hi - lo || 1)) * width)));

function newPair(): Pair {
	const a = sampleRef();
	const b = sampleRef();
	return trueTaste(a) >= trueTaste(b) ? { w: a, l: b } : { w: b, l: a };
}

function heldOut(n = 400): number {
	let hit = 0;
	for (let i = 0; i < n; i++) {
		const p = newPair();
		if (judge(p.w) > judge(p.l)) hit++;
	}
	return hit / n;
}

function sweep(beta: number) {
	const p = newPolicy();
	const rows: { kl: number; proxy: number; gold: number }[] = [];
	const snap = () =>
		rows.push({
			kl: kl(p),
			proxy: meanOf(p, judge),
			gold: meanOf(p, trueTaste)
		});
	snap();
	let next = 1;
	for (let step = 0; step < 20000 && next < STOPS.length; step++) {
		ascend(p, beta);
		if (kl(p) >= STOPS[next]) {
			snap();
			next++;
		}
	}
	// A leashed run simply stops travelling; report where it settled.
	if (rows.length === 1) snap();
	return rows;
}

async function main() {
	const L: string[] = [];
	const say = async (...s: string[]) => {
		L.push(...s);
		out.textContent = L.join('\n');
		await new Promise((r) => setTimeout(r));
	};

	await say('fitting a judge to comparisons from a taste it never sees', '');
	const pairs: Pair[] = [];
	for (let k = 1; k <= 30; k++) {
		pairs.push(newPair());
		const loss = fit(pairs);
		if (k % 6 === 0)
			await say(
				`  pairs ${String(k).padStart(2)}   loss ${loss.toFixed(3)}   ` +
					`calls ${(heldOut() * 100).toFixed(0)}% of unseen comparisons right`
			);
	}

	await say(
		'',
		'the judge is now a stand-in for the taste. turn an optimizer loose on it,',
		'with no leash at all, and watch the two part company.',
		'',
		'   KL    the judge says          the taste says'
	);
	const free = sweep(0);
	const pl = Math.min(...free.map((r) => r.proxy));
	const ph = Math.max(...free.map((r) => r.proxy));
	const gl = Math.min(...free.map((r) => r.gold));
	const gh = Math.max(...free.map((r) => r.gold));
	for (const r of free) {
		await say(
			`  ${r.kl.toFixed(1).padStart(4)}   ${r.proxy >= 0 ? '+' : ''}${r.proxy.toFixed(2)} ${bar(r.proxy, pl, ph).padEnd(23)} ${r.gold >= 0 ? '+' : ''}${r.gold.toFixed(2)} ${bar(r.gold, gl, gh)}`
		);
	}
	const peak = free.reduce((a, b) => (b.gold > a.gold ? b : a));
	const first = free[0];
	const last = free[free.length - 1];
	// Say only what the numbers say. Whether the run ends up worse than it
	// started depends on the seed, and a lab that overstates its own result is
	// exactly the failure this chapter is about.
	const gained = peak.gold - first.gold;
	const given = peak.gold - last.gold;
	const verdict =
		last.gold < first.gold
			? `by the end it was worse than where it started.`
			: `by the end it had handed back ${((given / (gained || 1)) * 100).toFixed(0)}% of what it gained.`;
	await say(
		'',
		`the judge's score peaked at ${last.kl.toFixed(1)} nats — the far end, as it always will.`,
		`the taste peaked at ${peak.kl.toFixed(1)} nats, and ${verdict}`,
		'nothing in the left-hand column marks the spot. that is the whole chapter.',
		''
	);

	await say(
		'now the same optimizer, on a leash of length β:',
		'',
		'     β     KL travelled   the taste says'
	);
	for (const beta of [0, 0.1, 0.25, 0.5, 1, 2, 4]) {
		const rows = sweep(beta);
		const last = rows[rows.length - 1];
		await say(
			`  ${beta.toFixed(2).padStart(4)}   ${last.kl.toFixed(2).padStart(6)}         ${last.gold >= 0 ? '+' : ''}${last.gold.toFixed(2)} ${bar(last.gold, gl, gh)}`
		);
	}
	await say(
		'',
		'somewhere in that column is a β that beats both ends. it is not at either end,',
		'and no run that only watches the judge could ever find it.',
		'',
		'things to break: move TRUE_BEST, shrink the comparison budget below 30,',
		'set decay to 0 in fit(), or widen REF_SIGMA and watch the judge lose its grip.'
	);
}

void main();

// Lab 6 — learning from reward, with no tensors anywhere.
//
// The chapter's sea chart, condensed. A boat picks one of eight compass
// headings each leg. It cannot sail within 35° of the wind, and it is quickest
// with the wind on the beam, so the cost of a leg depends on its angle — which
// is all the physics there is. The harbour sits nearly dead upwind, so there
// is no route that points at it.
//
// Nothing below mentions zigzagging. Watch the policy work it out.
//
// Plain loops on plain arrays: ZERO dependencies beyond vite.

const out = document.getElementById('out') as HTMLPreElement;

// ── the water ────────────────────────────────────────────────────────────────
const W = 14;
const H = 10;
const at = (x: number, y: number) => y * W + x;
const START = at(7, 0);
const HARBOUR = at(6, 9);
const LAND = new Set([at(6, 4), at(7, 4), at(6, 5)]);
const SHOALS = new Set([at(2, 5), at(11, 4)]);

const A = 8; // compass points
const DX = [0, 1, 1, 1, 0, -1, -1, -1];
const DY = [1, 1, 0, -1, -1, -1, 0, 1];
const NAME = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
const GLYPH = ['|', '/', '-', '\\', '|', '/', '-', '\\'];

/** Direction the wind blows FROM, radians clockwise from north. Change me. */
const WIND_FROM = 0;
const NO_GO = (35 * Math.PI) / 180;
const MIN_SPEED = 0.45;

const TIME_COST = -0.28;
const HARBOUR_REWARD = 12;
const SHOAL_REWARD = -9;
const IRONS_COST = -0.5;
const GAMMA = 0.97;
const MAX_STEPS = 150;
const SCATTER = 0.2;
const ENTROPY = 0.02;

const angleBetween = (a: number, b: number) => {
	let d = Math.abs(a - b) % (Math.PI * 2);
	if (d > Math.PI) d = Math.PI * 2 - d;
	return d;
};

/** The polar diagram: how fast the boat goes at a given angle to the wind. */
function boatSpeed(twa: number): number {
	if (twa <= NO_GO) return 0;
	if (twa < Math.PI / 2) {
		const u = (twa - NO_GO) / (Math.PI / 2 - NO_GO);
		return u * u * (3 - 2 * u);
	}
	return 1 - 0.35 * ((twa - Math.PI / 2) / (Math.PI / 2));
}

const twaOf = (a: number) => angleBetween((a * Math.PI) / 4, WIND_FROM);
const inIrons = (a: number) => boatSpeed(twaOf(a)) <= 0;
const stepTime = (a: number) => {
	const s = boatSpeed(twaOf(a));
	return s <= 0 ? IRONS_COST : TIME_COST / Math.max(MIN_SPEED, s);
};

function stepSea(s: number, a: number): { s: number; r: number; done: boolean } {
	if (inIrons(a)) return { s, r: IRONS_COST, done: false };
	const x = s % W;
	const y = (s / W) | 0;
	const nx = x + DX[a];
	const ny = y + DY[a];
	const t = stepTime(a);
	if (nx < 0 || nx >= W || ny < 0 || ny >= H || LAND.has(at(nx, ny)))
		return { s, r: t, done: false };
	const s2 = at(nx, ny);
	if (s2 === HARBOUR) return { s: s2, r: t + HARBOUR_REWARD, done: true };
	if (SHOALS.has(s2)) return { s: s2, r: t + SHOAL_REWARD, done: true };
	return { s: s2, r: t, done: false };
}

// ── the policy: a table of logits, softmaxed per cell ────────────────────────
const theta = new Float64Array(W * H * A); // all zero: no opinion about anything
const baseline = new Float64Array(W * H); // running mean of returns per cell

function policyAt(s: number, probs: Float64Array) {
	let mx = -Infinity;
	for (let i = 0; i < A; i++) mx = Math.max(mx, theta[s * A + i]);
	let sum = 0;
	for (let i = 0; i < A; i++) {
		probs[i] = Math.exp(theta[s * A + i] - mx);
		sum += probs[i];
	}
	for (let i = 0; i < A; i++) probs[i] /= sum;
}

let seed = 4242;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0), seed / 4294967296);

const WATER: number[] = [];
for (let i = 0; i < W * H; i++) if (!LAND.has(i) && !SHOALS.has(i) && i !== HARBOUR) WATER.push(i);

// ── one passage, and the REINFORCE update that follows it ────────────────────
function sailAndLearn(lr: number): { total: number; arrived: boolean } {
	// One passage in five starts somewhere other than the mooring. Without this
	// the boat never finds a harbour to windward at all — a uniform policy
	// drifts DOWNwind — and settles for sailing onto a shoal, which is cheaper
	// than running out of clock. Delete the next line to watch that happen.
	const from = rand() < SCATTER ? WATER[(rand() * WATER.length) | 0] : START;

	const states: number[] = [];
	const actions: number[] = [];
	const rewards: number[] = [];
	let s = from;
	let arrived = false;
	let total = 0;
	const probs = new Float64Array(A);
	for (let t = 0; t < MAX_STEPS; t++) {
		policyAt(s, probs);
		let r = rand();
		let a = A - 1;
		for (let i = 0; i < A; i++) {
			r -= probs[i];
			if (r <= 0) {
				a = i;
				break;
			}
		}
		const step = stepSea(s, a);
		states.push(s);
		actions.push(a);
		rewards.push(step.r);
		total += step.r;
		s = step.s;
		if (step.done) {
			arrived = s === HARBOUR;
			break;
		}
	}

	// return-to-go, computed backwards; truncated passages bootstrap with V
	const g = new Float64Array(rewards.length);
	let acc = states.length === MAX_STEPS ? baseline[s] : 0;
	for (let t = rewards.length - 1; t >= 0; t--) {
		acc = rewards[t] + GAMMA * acc;
		g[t] = acc;
	}

	// θ[s,i] += lr · [ (G_t − V(s)) · (1[i=a] − π_i)  +  entropy · ∂H/∂θ_i ]
	for (let t = 0; t < states.length; t++) {
		const st = states[t];
		const a = actions[t];
		policyAt(st, probs);
		const adv = g[t] - baseline[st];
		let h = 0;
		for (const q of probs) h -= q * Math.log(q + 1e-12);
		for (let i = 0; i < A; i++) {
			const pg = adv * ((i === a ? 1 : 0) - probs[i]);
			const eg = -probs[i] * (Math.log(probs[i] + 1e-12) + h); // stay curious
			theta[st * A + i] += lr * (pg + ENTROPY * eg);
		}
		baseline[st] += 0.15 * (g[t] - baseline[st]);
	}
	return { total, arrived: arrived && from === START };
}

// ── drawing ──────────────────────────────────────────────────────────────────
/** The greedy route: argmax at every cell, no dice. */
function route(): number[] {
	const probs = new Float64Array(A);
	const out: number[] = [];
	let s = START;
	const seen = new Set<number>();
	for (let t = 0; t < 120; t++) {
		policyAt(s, probs);
		let a = 0;
		for (let i = 1; i < A; i++) if (probs[i] > probs[a]) a = i;
		out.push(a);
		const step = stepSea(s, a);
		if (step.s === s) break;
		s = step.s;
		if (step.done || seen.has(s)) break;
		seen.add(s);
	}
	return out;
}

function chart(): string {
	const g = Array.from({ length: H }, () => Array(W).fill(' ·'));
	for (const c of LAND) g[(c / W) | 0][c % W] = ' #';
	for (const c of SHOALS) g[(c / W) | 0][c % W] = ' x';
	let s = START;
	for (const a of route()) {
		const step = stepSea(s, a);
		if (step.s === s) break;
		s = step.s;
		g[(s / W) | 0][s % W] = ' ' + GLYPH[a];
	}
	g[(START / W) | 0][START % W] = ' S';
	g[(HARBOUR / W) | 0][HARBOUR % W] = ' H';
	return g
		.slice()
		.reverse()
		.map((r) => r.join(''))
		.join('\n');
}

async function main() {
	const lines: string[] = [];
	const recent: number[] = [];
	const hits: number[] = [];
	const nogo = [...Array(A).keys()].filter(inIrons).map((a) => NAME[a]);
	for (let ep = 1; ep <= 14000; ep++) {
		const { total, arrived } = sailAndLearn(0.09);
		recent.push(total);
		hits.push(arrived ? 1 : 0);
		if (recent.length > 300) {
			recent.shift();
			hits.shift();
		}
		if (ep % 1000 === 0 || ep === 1) {
			const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
			const arr = (hits.reduce((a, b) => a + b, 0) / hits.length) * 100;
			lines.push(
				`episode ${String(ep).padStart(5)} · mean return ${mean.toFixed(2).padStart(7)} · arrives ${arr.toFixed(0).padStart(3)}%`
			);
			out.textContent =
				`wind from ${((WIND_FROM * 180) / Math.PI).toFixed(0)}° · cannot sail ${nogo.join(', ')}\n` +
				`S start · H harbour · # land · x shoal · the line is the greedy route\n\n` +
				`${chart()}\n\n` +
				lines.join('\n');
			await new Promise((r) => setTimeout(r));
		}
	}
	const r = route();
	const tacks = r.filter((a, i) => i > 0 && a !== r[i - 1]).length;
	out.textContent +=
		`\n\nheadings: ${r.map((a) => NAME[a]).join(' ')}\n` +
		`${tacks} changes of heading in ${r.length} legs — nothing in the reward asked for that.\n\n` +
		`things to break: set WIND_FROM to Math.PI (wind behind you) and watch the zigzag\n` +
		`vanish; set SCATTER to 0 and watch some winds strand the boat entirely; drop\n` +
		`NO_GO to zero and the whole manoeuvre stops being necessary.`;
}

void main();

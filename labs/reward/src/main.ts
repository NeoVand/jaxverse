// Lab 6 — learning from reward, with no tensors anywhere.
//
// The chapter's gridworld, condensed: an 8×6 field with a treasure, three
// pits, a step tax, and two wall fences. The policy is a table of logits —
// four numbers per cell — and learning is REINFORCE with return-to-go and a
// per-state baseline. Plain loops on plain arrays: this lab has ZERO
// dependencies beyond vite. Watch the arrows sharpen from confusion to route.

const out = document.getElementById('out') as HTMLPreElement;

// ── the world ────────────────────────────────────────────────────────────────
const W = 8;
const H = 6;
const at = (x: number, y: number) => y * W + x;
const START = at(0, 0);
const GOAL = at(7, 5);
const WALLS = new Set([at(2, 1), at(2, 2), at(2, 3), at(5, 2), at(5, 3), at(5, 4)]);
const PITS = new Set([at(1, 1), at(4, 1), at(6, 4)]);
const STEP_COST = -0.15;
const GOAL_REWARD = 10;
const PIT_REWARD = -8;
const GAMMA = 0.96;
const MAX_STEPS = 200;

// actions 0..3: up, down, left, right — y grows upward
const DX = [0, 0, -1, 1];
const DY = [1, -1, 0, 0];

function stepEnv(s: number, a: number): { s: number; r: number; done: boolean } {
	const x = s % W;
	const y = (s / W) | 0;
	let nx = x + DX[a];
	let ny = y + DY[a];
	if (nx < 0 || nx >= W || ny < 0 || ny >= H || WALLS.has(at(nx, ny))) {
		nx = x;
		ny = y;
	}
	const s2 = at(nx, ny);
	if (s2 === GOAL) return { s: s2, r: STEP_COST + GOAL_REWARD, done: true };
	if (PITS.has(s2)) return { s: s2, r: STEP_COST + PIT_REWARD, done: true };
	return { s: s2, r: STEP_COST, done: false };
}

// ── the policy: a table of logits, softmaxed per cell ────────────────────────
const theta = new Float64Array(W * H * 4); // all zero: a uniform, curious policy
const baseline = new Float64Array(W * H); // running mean of returns per state

function policyAt(s: number, probs: Float64Array) {
	let mx = -Infinity;
	for (let i = 0; i < 4; i++) mx = Math.max(mx, theta[s * 4 + i]);
	let sum = 0;
	for (let i = 0; i < 4; i++) {
		probs[i] = Math.exp(theta[s * 4 + i] - mx);
		sum += probs[i];
	}
	for (let i = 0; i < 4; i++) probs[i] /= sum;
}

let seed = 7;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0), seed / 4294967296);

// ── one episode + the REINFORCE update ───────────────────────────────────────
function runAndLearn(lr: number): { total: number; end: string } {
	const states: number[] = [];
	const actions: number[] = [];
	const rewards: number[] = [];
	let s = START;
	let end = 'timeout';
	let total = 0;
	const probs = new Float64Array(4);
	for (let t = 0; t < MAX_STEPS; t++) {
		policyAt(s, probs);
		let r = rand();
		let a = 3;
		for (let i = 0; i < 4; i++) {
			r -= probs[i];
			if (r <= 0) {
				a = i;
				break;
			}
		}
		const step = stepEnv(s, a);
		states.push(s);
		actions.push(a);
		rewards.push(step.r);
		total += step.r;
		s = step.s;
		if (step.done) {
			end = s === GOAL ? 'goal' : 'pit';
			break;
		}
	}
	// return-to-go, computed backwards; timeouts bootstrap with V(s_final)
	const g = new Float64Array(rewards.length);
	let acc = end === 'timeout' ? baseline[s] : 0;
	for (let t = rewards.length - 1; t >= 0; t--) {
		acc = rewards[t] + GAMMA * acc;
		g[t] = acc;
	}
	// θ[s,i] += lr · (G_t − V(s)) · (1[i=a] − π_i(s)) — the policy gradient
	for (let t = 0; t < states.length; t++) {
		const st = states[t];
		const a = actions[t];
		policyAt(st, probs);
		const adv = g[t] - baseline[st];
		for (let i = 0; i < 4; i++) theta[st * 4 + i] += lr * adv * ((i === a ? 1 : 0) - probs[i]);
		baseline[st] += 0.15 * (g[t] - baseline[st]); // the baseline chases fresh returns
	}
	return { total, end };
}

// ── ASCII rendering: the argmax arrow of every free cell ────────────────────
const ARROWS = ['↑', '↓', '←', '→'];
function drawPolicy(): string {
	const rows: string[] = [];
	const probs = new Float64Array(4);
	for (let y = H - 1; y >= 0; y--) {
		let row = '';
		for (let x = 0; x < W; x++) {
			const s = at(x, y);
			if (s === GOAL) row += ' ◆ ';
			else if (WALLS.has(s)) row += ' █ ';
			else if (PITS.has(s)) row += ' ○ ';
			else {
				policyAt(s, probs);
				let best = 0;
				for (let i = 1; i < 4; i++) if (probs[i] > probs[best]) best = i;
				// confidence shows as presence: faint policies print a dot
				row += probs[best] > 0.5 ? ` ${ARROWS[best]} ` : ' · ';
			}
		}
		rows.push(row);
	}
	return rows.join('\n');
}

async function main() {
	const lines: string[] = [];
	const recent: number[] = [];
	let wins = 0;
	for (let ep = 1; ep <= 6000; ep++) {
		const { total, end } = runAndLearn(0.08);
		recent.push(total);
		if (recent.length > 200) recent.shift();
		if (end === 'goal') wins++;
		if (ep % 500 === 0 || ep === 1) {
			const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
			lines.push(
				`episode ${String(ep).padStart(4)} · mean return (last 200): ${mean.toFixed(2).padStart(6)} ` +
					`· goals so far: ${wins}`
			);
			out.textContent =
				`start bottom-left → ◆ treasure · ○ pits end the episode · █ walls\n` +
				`arrows are each cell's favorite move; · means still undecided\n\n` +
				`${drawPolicy()}\n\n` +
				lines.join('\n');
			await new Promise((r) => setTimeout(r));
		}
	}
	out.textContent += `\n\ndone — best possible return is ${(GOAL_REWARD + STEP_COST * 12).toFixed(2)} (12 steps). Move the treasure, add pits, or zero the baseline and see what breaks.`;
}

void main();

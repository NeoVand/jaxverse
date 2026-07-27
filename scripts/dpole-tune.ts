// Scratch harness for the swing-up. Run: npx tsx scripts/dpole-tune.ts
// Env: EPISODES LR GAMMA SEEDS BATCH BETA DECAY UP PROBE_UP
import {
	actionForce,
	ACTION_REPEAT,
	createDpoleBaseline,
	createDpoleTheta,
	DECISIONS,
	DpoleCurriculum,
	DPOLE_ACTIONS,
	dpoleFeatures,
	dpolePolicy,
	dpoleReinforceUpdate,
	isUpright,
	N_FEATURES,
	physicsStep,
	resetDpole,
	runDpoleEpisode,
	tipHeight,
	type DpoleEpisode,
	type DpoleStart
} from '../src/lib/optim-rl/dpole';
import { mulberry32 } from '../src/lib/optim-rl/rng';
import { sampleFrom } from '../src/lib/optim-rl/softmax';

const LR = Number(process.env.LR ?? 0.15);
const GAMMA = process.env.GAMMA ? Number(process.env.GAMMA) : undefined;
const SEEDS = (process.env.SEEDS ?? '512,42').split(',').map(Number);
const EPISODES = Number(process.env.EPISODES ?? 40000);
const BATCH = Number(process.env.BATCH ?? 8);
const DECAY = process.env.DECAY ? Number(process.env.DECAY) : undefined;
const BETA = process.env.BETA ? Number(process.env.BETA) : undefined;
const UP = process.env.UP ? (Number(process.env.UP) as DpoleStart) : undefined;
const PROBE_UP = (Number(process.env.PROBE_UP ?? 0) as DpoleStart) ?? 0;
const WIN = 2000;

/** Eval episode from a hanging (or PROBE_UP) start: entropy, mix, mean
 * height, ticks caught. */
function probe(theta: Float64Array, rand: () => number): string {
	const s = resetDpole(rand, PROBE_UP);
	const f = new Float64Array(N_FEATURES);
	const p = new Float64Array(DPOLE_ACTIONS);
	let ent = 0;
	let h = 0;
	let caught = 0;
	const mix = new Array(DPOLE_ACTIONS).fill(0);
	for (let t = 0; t < DECISIONS; t++) {
		dpoleFeatures(s, f);
		dpolePolicy(theta, f, p);
		for (let a = 0; a < DPOLE_ACTIONS; a++) if (p[a] > 1e-9) ent -= p[a] * Math.log(p[a]);
		const a = sampleFrom(p, rand);
		mix[a]++;
		for (let k = 0; k < ACTION_REPEAT; k++) physicsStep(s, actionForce(a));
		h += tipHeight(s);
		if (isUpright(s)) caught++;
	}
	return `H${(ent / DECISIONS).toFixed(2)} mix${mix.map((m) => Math.round((m / DECISIONS) * 100)).join('/')} h${(h / DECISIONS).toFixed(2)} C${caught}`;
}

for (const seed of SEEDS) {
	const theta = createDpoleTheta();
	const baseline = createDpoleBaseline();
	const rand = mulberry32(seed);
	const t0 = performance.now();
	const sumRet = [0, 0, 0];
	const sumCaught = [0, 0, 0];
	const sumSteps = [0, 0, 0];
	const n = [0, 0, 0];
	console.log(`seed ${seed}:`);
	const cur = new DpoleCurriculum();
	for (let e = BATCH; e <= EPISODES; e += BATCH) {
		const eps: DpoleEpisode[] = [];
		for (let i = 0; i < BATCH; i++) {
			eps.push(UP !== undefined ? runDpoleEpisode(theta, rand, UP) : cur.next(theta, rand));
		}
		dpoleReinforceUpdate(theta, baseline, eps, LR, GAMMA, DECAY, BETA);
		for (const ep of eps) {
			sumRet[ep.kind] += ep.ret;
			sumCaught[ep.kind] += ep.caught;
			sumSteps[ep.kind] += ep.steps;
			n[ep.kind]++;
		}
		if (e % WIN < BATCH) {
			const per = (arr: number[], k: number) => (arr[k] / Math.max(n[k], 1)).toFixed(0);
			console.log(
				`  ${e}: hangR${per(sumRet, 0)} upT${per(sumSteps, 1)} upC${per(sumCaught, 1)} brT${per(sumSteps, 2)} brC${per(sumCaught, 2)} a${cur.alpha.toFixed(2)} w${cur.winRate.toFixed(2)} [${probe(theta, rand)}]`
			);
			for (let k = 0; k < 3; k++) {
				sumRet[k] = 0;
				sumCaught[k] = 0;
				sumSteps[k] = 0;
				n[k] = 0;
			}
		}
	}
	console.log(`  (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
}

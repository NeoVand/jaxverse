// Scratch probe: after training, what fraction of DELIVERIES convert into
// a held stand? This is the bottleneck behind slow recovery: the swing
// re-delivers fine after a shove, but each arrival is only caught with
// probability p — and the visible "struggle" is the geometric wait for a
// success. Run: npx tsx scripts/dpole-catch.ts
import {
	createDpoleTheta,
	createDpoleBaseline,
	dpoleReinforceUpdate,
	dpoleFeatures,
	dpolePolicy,
	physicsStep,
	resetDpole,
	actionForce,
	isUpright,
	isDelivery,
	tumbleStart,
	DpoleCurriculum,
	DECISIONS,
	DPOLE_ACTIONS,
	N_FEATURES,
	type DpoleEpisode,
	type DpoleState
} from '../src/lib/optim-rl/dpole';
import { mulberry32 } from '../src/lib/optim-rl/rng';

const EPISODES = Number(process.env.EPISODES ?? 150_000);
const SEED = Number(process.env.SEED ?? 512);

const theta = createDpoleTheta();
const baseline = createDpoleBaseline();
const cur = new DpoleCurriculum();
const rand = mulberry32(SEED);
for (let e = 0; e < EPISODES; e += 8) {
	const batch: DpoleEpisode[] = [];
	for (let i = 0; i < 8; i++) batch.push(cur.next(theta, rand));
	dpoleReinforceUpdate(theta, baseline, batch, 0.15);
}
console.log(`trained ${EPISODES} eps, alpha=${cur.alpha.toFixed(2)}`);

const f = new Float64Array(N_FEATURES);
const probs = new Float64Array(DPOLE_ACTIONS);
function act(s: DpoleState): number {
	dpoleFeatures(s, f);
	dpolePolicy(theta, f, probs);
	let u = rand();
	for (let a = 0; a < DPOLE_ACTIONS - 1; a++) {
		u -= probs[a];
		if (u <= 0) return a;
	}
	return DPOLE_ACTIONS - 1;
}

/** Swing from `start` until delivery (≤800 ticks), then run 400 more and
 * ask: did a 100-tick hold form? Repeat 100×. */
function catchRate(start: () => DpoleState, label: string): void {
	let delivered = 0;
	let caughtHold = 0;
	const spins: number[] = [];
	for (let e = 0; e < 100; e++) {
		const s = start();
		let arrived = false;
		for (let t = 0; t < 800; t++) {
			physicsStep(s, actionForce(act(s)));
			if (isDelivery(s)) {
				arrived = true;
				break;
			}
		}
		if (!arrived) continue;
		delivered++;
		spins.push(Math.abs(s.th1d) + Math.abs(s.th2d));
		let streak = 0;
		for (let t = 0; t < DECISIONS; t++) {
			physicsStep(s, actionForce(act(s)));
			streak = isUpright(s) ? streak + 1 : 0;
			if (streak >= 100) break;
		}
		if (streak >= 100) caughtHold++;
	}
	spins.sort((a, b) => a - b);
	console.log(
		`${label}: delivered ${delivered}/100, caught ${caughtHold}/${delivered} (${((100 * caughtHold) / Math.max(1, delivered)).toFixed(0)}%), median arrival spin ${spins[Math.floor(spins.length / 2)]?.toFixed(1)}`
	);
}

catchRate(() => resetDpole(rand, 0), 'from rest   ');
catchRate(() => tumbleStart(rand), 'from tumble ');

/**
 * Diagnostic only: known dynamics + privileged physical pose cost.
 * Run with `node tools/world-oracle-bench.ts` (Node 24+).
 * Nothing in src/ may import this benchmark or use its plans as learned behavior.
 */
import {
	ARM_DT,
	armPoints,
	initialArm,
	poseError,
	stepArm,
	wrapAngle,
	type ArmState
} from '../src/lib/world/simulator.ts';

interface Search {
	name: string;
	horizon: number;
	hold: boolean;
	candidates: number;
	rounds: number;
}

interface Result {
	search: string;
	start: number;
	goal: number;
	seed: number;
	initialError: number;
	minimumError: number;
	finalError: number;
	finalSpeed: number;
	firstReached: number | null;
	firstDwell: number | null;
	longestDwell: number;
	tipTravel: number;
	minimumFoldGap: number;
	meanPlanningMs: number;
}

const POSE_TOLERANCE = 0.15;
const DWELL_STEPS = 5;
const BUDGET = 40;
const GOALS = [
	{ q1: -0.9, q2: 1 },
	{ q1: -1.8, q2: 1.8 },
	{ q1: -0.7, q2: 2.65 }
];
const STARTS: ArmState[] = [
	initialArm(),
	{ ...initialArm(), v1: 0.5, v2: -0.5 },
	{ q1: -1.7, q2: 1.2, v1: -0.2, v2: 0.4 }
];
const SEARCHES: Search[] = [
	{ name: 'h6-terminal-64x3', horizon: 6, hold: false, candidates: 64, rounds: 3 },
	{ name: 'h12-terminal-64x3', horizon: 12, hold: false, candidates: 64, rounds: 3 },
	{ name: 'h12-hold-64x3', horizon: 12, hold: true, candidates: 64, rounds: 3 },
	{ name: 'h20-hold-64x3', horizon: 20, hold: true, candidates: 64, rounds: 3 },
	{ name: 'h12-hold-128x4', horizon: 12, hold: true, candidates: 128, rounds: 4 }
];

// Same generators as the production CEM; importing the model would bring JAX into this oracle.
function random(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function normal(rand: () => number): number {
	return Math.sqrt(-2 * Math.log(Math.max(1e-9, rand()))) * Math.cos(2 * Math.PI * rand());
}

function oracleAction(
	state: ArmState,
	goal: (typeof GOALS)[number],
	search: Search,
	seed: number
): readonly [number, number] {
	const width = 2 * Math.ceil(search.horizon / 2);
	const eliteCount = Math.max(4, Math.floor(search.candidates / 8));
	const rand = random(seed);
	const mean = new Float32Array(width);
	const deviation = new Float32Array(width).fill(0.8);
	const window = search.hold ? Math.min(4, search.horizon) : 1;
	let bestCost = Infinity;
	let best: readonly [number, number] = [0, 0];
	for (let round = 0; round < search.rounds; round++) {
		const samples = new Float32Array(search.candidates * width);
		const costs = new Float64Array(search.candidates);
		for (let b = 0; b < search.candidates; b++) {
			for (let k = 0; k < width; k++)
				samples[b * width + k] =
					b === 0 ? mean[k] : Math.max(-1, Math.min(1, mean[k] + deviation[k] * normal(rand)));
			let predicted = state;
			let cost = 0;
			for (let t = 0; t < search.horizon; t++) {
				const k = b * width + Math.floor(t / 2) * 2;
				predicted = stepArm(predicted, [samples[k], samples[k + 1]]);
				if (t >= search.horizon - window) cost += poseError(predicted, goal) ** 2 / window;
			}
			costs[b] = cost;
			if (cost < bestCost) {
				bestCost = cost;
				best = [samples[b * width], samples[b * width + 1]];
			}
		}
		const order = Array.from({ length: search.candidates }, (_, i) => i).sort(
			(a, b) => costs[a] - costs[b]
		);
		for (let k = 0; k < width; k++) {
			let mu = 0;
			for (let e = 0; e < eliteCount; e++) mu += samples[order[e] * width + k] / eliteCount;
			let variance = 0;
			for (let e = 0; e < eliteCount; e++)
				variance += (samples[order[e] * width + k] - mu) ** 2 / eliteCount;
			mean[k] = mu;
			deviation[k] = Math.max(0.08, Math.sqrt(variance));
		}
	}
	return best;
}

function run(search: Search, start: number, goalIndex: number, seed: number): Result {
	let state = { ...STARTS[start] };
	const goal = GOALS[goalIndex];
	const initialError = poseError(state, goal);
	let minimumError = initialError;
	let firstReached: number | null = null;
	let firstDwell: number | null = null;
	let dwell = 0;
	let longestDwell = 0;
	let tipTravel = 0;
	let minimumFoldGap = Math.PI;
	let planningMs = 0;
	for (let t = 0; t < BUDGET; t++) {
		const before = performance.now();
		const action = oracleAction(state, goal, search, seed * 1000 + t);
		planningMs += performance.now() - before;
		const next = stepArm(state, action);
		const oldTip = armPoints(state).tip;
		const newTip = armPoints(next).tip;
		tipTravel += Math.hypot(newTip.x - oldTip.x, newTip.y - oldTip.y);
		state = next;
		minimumFoldGap = Math.min(minimumFoldGap, Math.PI - Math.abs(wrapAngle(state.q2)));
		const error = poseError(state, goal);
		minimumError = Math.min(minimumError, error);
		if (error <= POSE_TOLERANCE) {
			dwell++;
			if (firstReached === null) firstReached = t + 1;
			if (dwell >= DWELL_STEPS && firstDwell === null) firstDwell = t + 1;
		} else dwell = 0;
		longestDwell = Math.max(longestDwell, dwell);
	}
	return {
		search: search.name,
		start,
		goal: goalIndex,
		seed,
		initialError,
		minimumError,
		finalError: poseError(state, goal),
		finalSpeed: Math.hypot(state.v1, state.v2),
		firstReached,
		firstDwell,
		longestDwell,
		tipTravel,
		minimumFoldGap,
		meanPlanningMs: planningMs / BUDGET
	};
}

const selected = process.argv[2];
const results: Result[] = [];
for (const search of SEARCHES.filter((search) => !selected || search.name.includes(selected))) {
	for (let start = 0; start < STARTS.length; start++)
		for (let goal = 0; goal < GOALS.length; goal++)
			for (const seed of [7, 19, 37]) results.push(run(search, start, goal, seed));
	const group = results.filter((result) => result.search === search.name);
	const mean = (key: 'finalError' | 'finalSpeed' | 'meanPlanningMs') =>
		group.reduce((sum, result) => sum + result[key], 0) / group.length;
	console.error(
		JSON.stringify({
			search: search.name,
			runs: group.length,
			reached: group.filter((result) => result.firstReached !== null).length,
			dwelled: group.filter((result) => result.firstDwell !== null).length,
			finalWithinTolerance: group.filter((result) => result.finalError <= POSE_TOLERANCE).length,
			meanFinalError: mean('finalError'),
			meanFinalSpeed: mean('finalSpeed'),
			meanPlanningMs: mean('meanPlanningMs')
		})
	);
}
console.log(
	JSON.stringify(
		{
			diagnostic: 'Known simulator dynamics and physical full-pose cost; no learned model.',
			criteria: {
				poseToleranceRadians: POSE_TOLERANCE,
				dwellSteps: DWELL_STEPS,
				dt: ARM_DT,
				budget: BUDGET
			},
			goals: GOALS,
			starts: STARTS,
			results
		},
		null,
		2
	)
);

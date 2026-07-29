/// <reference lib="webworker" />
// A practice hall for the double pendulum, running off the main thread.
// The pool RACES rather than averages: each hall is a fully independent
// learner — its own θ, RNG, baseline, curriculum — running the exact
// single-thread recipe the tests validate. The stage watches every hall's
// fitness and always performs with the champion's θ. Laggards eventually
// ADOPT the champion's weights (population-based training). Averaging the
// halls' updates into one θ was tried first and failed outright: REINFORCE
// discovery lives on a stream compounding its own lucky exploration, and
// six streams summed into one θ compound nobody's.
//
// Protocol (main → hall):
//   { type: 'boot',  seed, lr }       once, right after creation
//   { type: 'run' }                   start (or resume) practising
//   { type: 'stop' }                  pause; all state kept
//   { type: 'lr', lr }                learning-rate slider moved
//   { type: 'adopt', theta, deliverEMA, drillEMA }   take the champion's θ
//   { type: 'live', episode }         a segment played out on the stage
// and (hall → main), after every round:
//   { type: 'report', theta, score, deliverEMA, drillEMA, episodes, drills, alpha }
import {
	createDpoleBaseline,
	createDpoleTheta,
	DpoleCurriculum,
	dpoleReinforceUpdate,
	type DpoleEpisode,
	type DpoleState
} from './dpole';
import { mulberry32, type Rand } from './rng';

const BATCHES_PER_ROUND = 6; // 48 episodes ≈ 25 ms per report
const EMA = 0.02; // fitness horizon ≈ the last few hundred episodes

const theta = createDpoleTheta();
const baseline = createDpoleBaseline();
const curriculum = new DpoleCurriculum();
let rand: Rand = mulberry32(1);
let lr = 0.15;
let running = false;
let deliverEMA = 0; // recent fraction of swing drills that delivered
let drillEMA = 0; // recent caught-ticks per brink drill, as a fraction

function round() {
	if (!running) return;
	const drills: number[] = [];
	let episodes = 0;
	for (let b = 0; b < BATCHES_PER_ROUND; b++) {
		const batch: DpoleEpisode[] = [];
		for (let i = 0; i < 8; i++) batch.push(curriculum.next(theta, rand));
		dpoleReinforceUpdate(theta, baseline, batch, lr);
		episodes += batch.length;
		for (const ep of batch) {
			if (ep.kind === 0) deliverEMA += EMA * ((ep.delivered ? 1 : 0) - deliverEMA);
			if (ep.kind === 2) {
				drillEMA += EMA * (ep.caught / 400 - drillEMA);
				drills.push(ep.caught);
			}
		}
	}
	postMessage({
		type: 'report',
		theta: Float64Array.from(theta),
		score: deliverEMA + 0.5 * drillEMA,
		deliverEMA,
		drillEMA,
		episodes,
		drills,
		alpha: curriculum.alpha
	});
	// yield to the message queue between rounds, so stop/adopt/lr land
	setTimeout(round, 0);
}

type HallMessage =
	| { type: 'boot'; seed: number; lr: number }
	| { type: 'run' }
	| { type: 'stop' }
	| { type: 'lr'; lr: number }
	| { type: 'adopt'; theta: Float64Array; deliverEMA: number; drillEMA: number }
	| { type: 'live'; episode: DpoleEpisode }
	| { type: 'delivery'; state: DpoleState };

self.onmessage = (e: MessageEvent<HallMessage>) => {
	const m = e.data;
	if (m.type === 'boot') {
		rand = mulberry32(m.seed);
		lr = m.lr;
	} else if (m.type === 'run') {
		if (!running) {
			running = true;
			round();
		}
	} else if (m.type === 'stop') {
		running = false;
	} else if (m.type === 'lr') {
		lr = m.lr;
	} else if (m.type === 'adopt') {
		theta.set(m.theta);
		// the fitness credit travels with the weights — otherwise the hall
		// would be adopted over and over while its own EMAs catch up
		deliverEMA = m.deliverEMA;
		drillEMA = m.drillEMA;
	} else if (m.type === 'live') {
		// the stage's own segments — shoves included — teach every hall
		dpoleReinforceUpdate(theta, baseline, [m.episode], lr);
	} else if (m.type === 'delivery') {
		// a delivery witnessed on the live stage becomes a rehearsable drill
		curriculum.deliveries.push(m.state);
	}
};

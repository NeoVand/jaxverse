/// <reference lib="webworker" />
// A practice hall for the double pendulum, running off the main thread.
// The pool RACES rather than averages: each hall is a fully independent
// learner — its own θ, RNG, baseline, curriculum — running the exact
// single-thread recipe the tests validate. The stage watches every hall's
// fitness and always performs with the champion's θ.
//
// What passes BETWEEN halls is in ./swarm.ts, and it is two things. A hall
// ANNOUNCES a find — a delivery cleaner than any it has managed before —
// and the broker hands that state to every other hall, which drops it
// straight into its replay buffer and starts rehearsing someone else's
// discovery. And a hall that has fallen well behind is PULLED, a little at
// a time, toward a fitness-weighted blend of the halls ahead of it. The
// leader is pulled by nobody: averaging the whole pool into one θ was tried
// first and failed outright, because REINFORCE discovery lives on a stream
// compounding its own lucky exploration, and six streams summed into one θ
// compound nobody's. Downhill-only drift keeps the front of the race pure
// and rescues the tail, which is the only part that needed rescuing.
//
// Protocol (main → hall):
//   { type: 'boot',  seed, lr }       once, right after creation
//   { type: 'run' }                   start (or resume) practising
//   { type: 'stop' }                  pause; all state kept
//   { type: 'lr', lr }                learning-rate slider moved
//   { type: 'peers', thetas, cand, prior, rank, n, gate, deliverEMAs,
//     drillEMAs, now }                 the weights of the halls ahead of this
//                                     one and the prior over them; the hall
//                                     picks one (or refuses) and learns from it
//   { type: 'delivery', state, grad? }  a discovery to rehearse, and the
//                                     gradient of the episode that found it
//   { type: 'live', episode }         a segment played out on the stage
// and (hall → main), after every round:
//   { type: 'report', theta, score, deliverEMA, drillEMA, showDeliver,
//     showDrill, episodes, drills,
//     alpha, finds }
import {
	createDpoleBaseline,
	createDpoleTheta,
	deliveryBonus,
	DpoleCurriculum,
	dpoleGradient,
	dpoleReinforceUpdate,
	isPrimeDelivery,
	type DpoleEpisode,
	type DpoleState
} from './dpole';
import { mulberry32, type Rand } from './rng';

const BATCHES_PER_ROUND = 6; // 48 episodes ≈ 25 ms per report
// Fitness horizon — and this is a CONTROL signal, not a display value.
//
// It was briefly 0.003, on the reasoning that at 0.02 the delivery rate
// averages only its last fifty swing attempts and the residual sampling
// noise (σ ≈ 0.04 on a rate of 0.9) was visible as the pool twitching on the
// field. Both halves of that were true and the conclusion was still wrong,
// because `score` is not only drawn. It is what the halls are RANKED by, and
// rank is the entire reward of the social learner below: a decision is graded
// on whether it climbed the ranking over the next SOC_HORIZON. Slowing this
// average flattens the ranking exactly during the first seconds, when every
// hall is still near zero and the differences that ought to separate them are
// smaller than the lag — so the social learner spends the discovery window
// being taught by noise, and then acts on what it learned with a step of
// SOC_KAPPA.
//
// Measured, six halls, matched wall-clock and matched episode counts: at 0.02
// the performing rig first holds the stack at 30 s and has 3,690 held ticks by
// 90 s. At 0.003 it first holds at 80 s and has 764. The twitch belongs to the
// follower that draws the field, which is where it is now damped.
const EMA = 0.02;
const FINDS_PER_ROUND = 2; // announcements are rare by construction; cap anyway
// A find must beat the hall's own best by this much to be worth everyone's
// attention — and that bar sags very slowly, so a hall whose policy has
// drifted can rediscover its own ground instead of going silent forever.
const FIND_MARGIN = 1.03;
const FIND_FORGET = 0.999;
// How much of another hall's gradient to step along. Small on purpose: the
// episode behind it was collected under THEIR policy, so for us it is
// off-policy and the honest correction (an importance ratio we do not
// compute) is somewhere below one. A third of a step in a direction that
// just worked, from a hall that just proved it works, is the trade.
const SHARED_GRAD = 0.35;

// ── the social layer ──
// Every hall also runs a second, tiny learner whose only question is WHO TO
// LISTEN TO. Its actions are "take hall j's weights a step" and "refuse
// everyone"; its policy is a softmax over one logit per peer plus a refusal
// logit, added to the prior the broker sends (ahead of me, and improving);
// and its reward is the one thing a competitor cares about — did I climb the
// pool's ranking, or fall. It is REINFORCE again, three lines of it, on a
// decision the pendulum policy never sees.
//
// This is not decoration on top of the swarm; it is the fix for something
// measured. Influence applied indiscriminately made the pool fail TOGETHER
// (mean pool fitness 0.33 against 0.82 for no sharing at all), because a
// hall mid-discovery that keeps being dragged toward the leader never
// finishes its own idea. A hall being hurt by advice can now learn to refuse
// it, and one being helped can learn to ask for more.
const SOC_LR = 0.5; // social learning rate
const SOC_KAPPA = 0.16; // how far one accepted influence moves θ
const SOC_BASE_LR = 0.05; // the social baseline's own EMA
const SOC_HORIZON = 2400; // ms before a social decision is graded
const SOC_SKEPTICISM = 2.2; // the opening bias toward refusing everyone
let socLogit: Float64Array | null = null; // learned: one per peer, + refuse
let socBase = 0; // "how much do I usually climb?"
let socProbs: Float64Array | null = null; // last decision distribution, drawn
const socPending: {
	pick: number;
	cand: Int32Array;
	probs: Float64Array;
	rank: number;
	at: number;
}[] = [];

const theta = createDpoleTheta();
const baseline = createDpoleBaseline();
const curriculum = new DpoleCurriculum();
let rand: Rand = mulberry32(1);
/** The social learner draws here and NOT from `rand`.
 *
 * `rand` is the training stream: the curriculum pulls from it to choose every
 * start state. Sampling a social decision out of it too means the episodes a
 * hall practises depend on how many peer messages have arrived, and peer
 * messages arrive on wall-clock timing — so the run stops being reproducible
 * from its seed, which is the only reason the seed exists. (This drew from
 * `Math.random()` once, which was not reproducible either; moving it onto the
 * training stream fixed the wrong half of that.) Same seed, separate stream:
 * the decisions replay, and the training they interleave with does not move. */
let socRand: Rand = mulberry32(2);
let lr = 0.15;
let running = false;
let deliverEMA = 0; // recent fraction of swing drills that delivered
let drillEMA = 0; // recent caught-ticks per brink drill, as a fraction
/** The same two skills, averaged over a much longer window, for DRAWING only.
 *
 * The fast pair above is a control signal — it sets `score`, which sets the
 * ranking, which is the social learner's whole reward — so it has to answer
 * within the couple of seconds a social decision is graded over. At that speed
 * it carries real sampling noise, around 0.04 on a rate of 0.9, and the field
 * draws each hall AT its two skills, so that noise is the pool visibly
 * shaking.
 *
 * Slowing the fast pair fixes the picture and breaks the learning; damping it
 * in the follower that draws does not work either, because that follower
 * adapts its rate to the distance still to cover and at this noise level the
 * NOISE is the distance — it reads a tremble as a hall that has moved and
 * hurries to keep up. Two averages of the same two numbers is the way out: the
 * slow pair is a true reading of an earlier moment rather than a point
 * invented between two readings, so the rungs still measure exactly what they
 * appear to measure. It lags by about a second. */
let showDeliver = 0;
let showDrill = 0;
const SHOW_EMA = 0.003;
let bestFind = 0; // the finest delivery this hall has ever announced

function round() {
	if (!running) return;
	const drills: number[] = [];
	const finds: { state: DpoleState; bonus: number; grad: Float64Array }[] = [];
	let episodes = 0;
	for (let b = 0; b < BATCHES_PER_ROUND; b++) {
		const batch: DpoleEpisode[] = [];
		for (let i = 0; i < 8; i++) batch.push(curriculum.next(theta, rand));
		dpoleReinforceUpdate(theta, baseline, batch, lr);
		episodes += batch.length;
		for (const ep of batch) {
			if (ep.kind === 0) {
				deliverEMA += EMA * ((ep.delivered ? 1 : 0) - deliverEMA);
				showDeliver += SHOW_EMA * ((ep.delivered ? 1 : 0) - showDeliver);
			}
			if (ep.kind === 2) {
				drillEMA += EMA * (ep.caught / 400 - drillEMA);
				showDrill += SHOW_EMA * (ep.caught / 400 - showDrill);
				drills.push(ep.caught);
			}
			// a personal best worth telling the others about. Prime only: the
			// buffer the neighbours will rehearse from must not be poisoned
			// with folded or wall-pinned arrivals nobody can catch
			if (ep.delivered && finds.length < FINDS_PER_ROUND && isPrimeDelivery(ep.delivered)) {
				const bonus = deliveryBonus(ep.delivered);
				if (bonus > bestFind * FIND_MARGIN) {
					bestFind = bonus;
					// the announcement carries both halves of what was learned:
					// the STATE it arrived in, to rehearse from, and the
					// GRADIENT of the episode that got there — 165 numbers the
					// others can step along the moment they receive them, where
					// the episode itself would be a hundred kilobytes. The
					// gradient is recomputed against this hall's own baselines
					// rather than reused from the batch, so it is the lesson of
					// this one episode and nothing else.
					finds.push({
						state: ep.delivered,
						bonus,
						grad: dpoleGradient(theta, baseline, [ep])
					});
				}
			}
		}
	}
	bestFind *= FIND_FORGET;
	postMessage({
		type: 'report',
		theta: Float64Array.from(theta),
		score: deliverEMA + 0.5 * drillEMA,
		deliverEMA,
		drillEMA,
		showDeliver,
		showDrill,
		episodes,
		drills,
		alpha: curriculum.alpha,
		finds,
		social: socProbs ? Float64Array.from(socProbs) : null
	});
	// yield to the message queue between rounds, so stop/adopt/lr land
	setTimeout(round, 0);
}

/** One turn of the social learner: grade the decisions that have come due,
 * then make a new one.
 *
 * The broker sends CANDIDATES, not the whole pool — the handful of halls
 * actually ahead of this one, capped. With a pool of thirty that keeps the
 * message a few kilobytes instead of a megabyte, and it costs nothing real:
 * a hall was never going to take advice from someone behind it. The learned
 * logits are still indexed by hall, so an opinion about hall 7 survives the
 * rounds where hall 7 is not a candidate. */
function social(m: {
	thetas: Float64Array;
	cand: Int32Array;
	prior: Float64Array;
	deliverEMAs: Float64Array;
	drillEMAs: Float64Array;
	rank: number;
	n: number;
	gate: number;
	now: number;
}) {
	const n = m.n;
	const K = m.cand.length;
	const P = theta.length;
	if (!socLogit || socLogit.length !== n + 1) {
		const grown = new Float64Array(n + 1);
		// A hall starts SKEPTICAL, not neutral: refusing everyone is the
		// opening bias, and being influenced is the thing that has to earn
		// its place. Measured the other way round — a pool that starts
		// credulous spends the first half-minute being dragged around before
		// its social learner has graded enough decisions to object, and by
		// then the discovery it was in the middle of is gone.
		grown[n] = socLogit ? socLogit[socLogit.length - 1] : SOC_SKEPTICISM;
		// the pool can be resized live; opinions about the halls that were
		// already here survive it
		if (socLogit) for (let k = 0; k < Math.min(n, socLogit.length - 1); k++) grown[k] = socLogit[k];
		else grown[n] = SOC_SKEPTICISM;
		socLogit = grown;
		socPending.length = 0;
	}

	// ── grade what has come due ──
	// The reward is rank, exactly as a competitor would score it: points for
	// standing above the halls you were below. Positive means the decision
	// (to listen to somebody, or to refuse everybody) was followed by a
	// climb; the baseline makes it "better than my usual", the same trick
	// the pendulum policy uses one level down.
	while (socPending.length && m.now - socPending[0].at >= SOC_HORIZON) {
		const d = socPending.shift();
		if (!d) break;
		const r = (d.rank - m.rank) / Math.max(1, n - 1);
		const adv = r - socBase;
		socBase += SOC_BASE_LR * (r - socBase);
		for (let c = 0; c < d.cand.length; c++) {
			const id = d.cand[c];
			if (id < n) socLogit[id] += SOC_LR * adv * ((id === d.pick ? 1 : 0) - d.probs[c]);
		}
		socLogit[n] += SOC_LR * adv * ((d.pick === n ? 1 : 0) - d.probs[d.cand.length]);
	}

	// ── decide: one action per candidate, plus "refuse everyone" ──
	const z = new Float64Array(K + 1);
	let top = -Infinity;
	for (let c = 0; c < K; c++) {
		z[c] = m.prior[c] + socLogit[m.cand[c]];
		if (z[c] > top) top = z[c];
	}
	z[K] = socLogit[n];
	if (z[K] > top) top = z[K];
	const probs = new Float64Array(K + 1);
	let sum = 0;
	for (let c = 0; c <= K; c++) {
		probs[c] = Number.isFinite(z[c]) ? Math.exp(z[c] - top) : 0;
		sum += probs[c];
	}
	for (let c = 0; c <= K; c++) probs[c] /= sum;
	// the hall's own seeded stream — its social one, see above
	let u = socRand();
	let slot = K;
	for (let c = 0; c <= K; c++) {
		u -= probs[c];
		if (u <= 0) {
			slot = c;
			break;
		}
	}
	const pick = slot < K ? m.cand[slot] : n; // n means "refuse everyone"
	socPending.push({ pick, cand: m.cand, probs, rank: m.rank, at: m.now });

	// what the main thread draws: one dense row over the whole pool, so the
	// graph is this hall's own probabilities and not a re-derivation of them
	const dense = new Float64Array(n + 1);
	for (let c = 0; c < K; c++) dense[m.cand[c]] = probs[c];
	dense[n] = probs[K];
	socProbs = dense;

	// ── act ──
	// Refusal is free and costs nothing but the chance to catch up. Taking
	// advice moves this hall a step toward that peer, scaled by how far
	// behind it is: a hall at the front barely moves even when it chooses to
	// listen, because there is little there for it to gain and a great deal
	// — its own independent stream — for it to lose.
	if (slot >= K) return;
	const k = SOC_KAPPA * m.gate;
	const off = slot * P;
	for (let i = 0; i < P; i++) theta[i] += k * (m.thetas[off + i] - theta[i]);
	deliverEMA += k * (m.deliverEMAs[slot] - deliverEMA);
	drillEMA += k * (m.drillEMAs[slot] - drillEMA);
	// the drawn pair moves with them, or the mark on the field would go on
	// describing a policy this hall has partly stepped away from
	showDeliver += k * (m.deliverEMAs[slot] - showDeliver);
	showDrill += k * (m.drillEMAs[slot] - showDrill);
}

type HallMessage =
	| { type: 'boot'; seed: number; lr: number }
	| { type: 'run' }
	| { type: 'stop' }
	| { type: 'lr'; lr: number }
	| {
			type: 'peers';
			thetas: Float64Array;
			cand: Int32Array;
			prior: Float64Array;
			deliverEMAs: Float64Array;
			drillEMAs: Float64Array;
			rank: number;
			n: number;
			gate: number;
			now: number;
	  }
	| { type: 'live'; episode: DpoleEpisode }
	| { type: 'delivery'; state: DpoleState; grad?: Float64Array };

self.onmessage = (e: MessageEvent<HallMessage>) => {
	const m = e.data;
	if (m.type === 'boot') {
		rand = mulberry32(m.seed);
		// a different stream from the same seed, not a different seed: two halls
		// booted alike still differ, and one hall replays exactly
		socRand = mulberry32((m.seed ^ 0x9e3779b9) >>> 0);
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
	} else if (m.type === 'peers') {
		social(m);
	} else if (m.type === 'live') {
		// the stage's own segments — shoves included — teach every hall
		dpoleReinforceUpdate(theta, baseline, [m.episode], lr);
	} else if (m.type === 'delivery') {
		// Somebody found a clean arrival — the live stage, or a neighbour a
		// millisecond ago. Two things arrive with it. The STATE becomes a
		// place this hall rehearses from, which costs it nothing: a start
		// position is not an opinion about what to do. The GRADIENT, when the
		// finder sent one, is stepped along right now at a fraction of the
		// learning rate — off-policy, so a fraction is the honest size of it,
		// but it means a discovery reaches every hall's weights in the same
		// frame it was made rather than seeping in over the next ten seconds.
		curriculum.guests.push(m.state);
		if (m.grad) {
			const step = lr * SHARED_GRAD;
			for (let i = 0; i < theta.length; i++) theta[i] += step * m.grad[i];
		}
	}
};

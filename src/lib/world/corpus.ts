import { renderSensorInto, SENSOR_SIZE } from './sensor';
import { ARM_DT, DEFAULT_DAMPING, stepArm, type Action, type ArmState } from './simulator';

export interface ArmEpisode {
	seed: number;
	steps: number;
	/** (steps + 1) × frameSize; frame t precedes action t. */
	frames: Float32Array;
	/** steps × 2; action t drives frame t to frame t+1. */
	actions: Float32Array;
	/** Privileged labels: diagnostic readout/evaluation only, never training inputs. */
	states: ArmState[];
}

export interface WorldCorpus {
	train: ArmEpisode[];
	validation: ArmEpisode[];
	size: number;
	frameSize: number;
	dt: number;
	damping: number;
}

export interface CorpusOptions {
	seed?: number;
	trainEpisodes?: number;
	validationEpisodes?: number;
	stepsPerEpisode?: number;
	size?: number;
	dt?: number;
	damping?: number;
}

export interface TransitionBatch {
	batchSize: number;
	/** Each frame array has shape [batchSize, frameSize]. */
	previous: Float32Array;
	current: Float32Array;
	next: Float32Array;
	/** [batchSize, 4]: a[t−1,0], a[t−1,1], a[t,0], a[t,1]. */
	actions: Float32Array;
}

/** Local deterministic randomness, independent of UI activity or training seeds. */
export function seededRandom(seed: number): () => number {
	let word = seed >>> 0;
	return () => {
		word = (word + 0x6d2b79f5) | 0;
		let mixed = Math.imul(word ^ (word >>> 15), word | 1);
		mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
		return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
	};
}

function episode(
	seed: number,
	steps: number,
	size: number,
	dt: number,
	damping: number
): ArmEpisode {
	const random = seededRandom(seed);
	const symmetric = () => 2 * random() - 1;
	const frameSize = size * size;
	const frames = new Float32Array((steps + 1) * frameSize);
	const actions = new Float32Array(steps * 2);
	const states: ArmState[] = [
		{
			q1: Math.PI * symmetric(),
			q2: Math.PI * symmetric(),
			v1: 0.7 * symmetric(),
			v2: 0.7 * symmetric()
		}
	];
	renderSensorInto(states[0], frames, 0, size);
	let action: Action = [symmetric(), symmetric()];
	let remaining = 0;
	for (let t = 0; t < steps; t++) {
		if (remaining === 0) {
			const mode = random();
			if (mode < 0.18) action = [0, 0];
			else if (mode < 0.4) action = [-action[0], -action[1]];
			else action = [symmetric(), symmetric()];
			remaining = 2 + Math.floor(random() * 9);
		}
		// Store before integrating, so the physical action and recorded Float32 agree exactly.
		actions[t * 2] = action[0];
		actions[t * 2 + 1] = action[1];
		const next = stepArm(states[t], [actions[t * 2], actions[t * 2 + 1]], { dt, damping });
		states.push(next);
		renderSensorInto(next, frames, (t + 1) * frameSize, size);
		remaining--;
	}
	return { seed, steps, frames, actions, states };
}

/** Whole episodes and their random streams stay on one side of the split. */
export function generateCorpus(options: CorpusOptions = {}): WorldCorpus {
	const seed = (options.seed ?? 2026) >>> 0;
	const trainEpisodes = options.trainEpisodes ?? 64;
	const validationEpisodes = options.validationEpisodes ?? 12;
	const steps = options.stepsPerEpisode ?? 64;
	const size = options.size ?? SENSOR_SIZE;
	const dt = options.dt ?? ARM_DT;
	const damping = options.damping ?? DEFAULT_DAMPING;
	if (
		![trainEpisodes, validationEpisodes, steps, size].every(Number.isInteger) ||
		trainEpisodes < 1 ||
		validationEpisodes < 1 ||
		steps < 2 ||
		size < 8 ||
		size > 256
	)
		throw new RangeError(
			'Corpus requires positive episode counts, at least two steps, and a valid sensor size.'
		);
	const makeEpisodes = (count: number, start: number) =>
		Array.from({ length: count }, (_, i) =>
			episode((seed + start + i) >>> 0, steps, size, dt, damping)
		);
	return {
		train: makeEpisodes(trainEpisodes, 0),
		validation: makeEpisodes(validationEpisodes, trainEpisodes),
		size,
		frameSize: size * size,
		dt,
		damping
	};
}

/** Samples aligned two-observation windows, without admitting diagnostic state labels. */
export function gatherTransitionBatch(
	episodes: readonly ArmEpisode[],
	batchSize: number,
	random: () => number,
	frameSize = SENSOR_SIZE * SENSOR_SIZE
): TransitionBatch {
	if (episodes.length === 0 || !Number.isInteger(batchSize) || batchSize < 1)
		throw new RangeError('A transition batch needs episodes and a positive integer size.');
	const previous = new Float32Array(batchSize * frameSize);
	const current = new Float32Array(batchSize * frameSize);
	const next = new Float32Array(batchSize * frameSize);
	const actions = new Float32Array(batchSize * 4);
	for (let b = 0; b < batchSize; b++) {
		const episode = episodes[Math.floor(random() * episodes.length)];
		if (episode.steps < 2 || episode.frames.length !== (episode.steps + 1) * frameSize)
			throw new RangeError('Episode shape does not match the transition batch.');
		const t = 1 + Math.floor(random() * (episode.steps - 1));
		previous.set(episode.frames.subarray((t - 1) * frameSize, t * frameSize), b * frameSize);
		current.set(episode.frames.subarray(t * frameSize, (t + 1) * frameSize), b * frameSize);
		next.set(episode.frames.subarray((t + 1) * frameSize, (t + 2) * frameSize), b * frameSize);
		actions.set(episode.actions.subarray((t - 1) * 2, (t + 1) * 2), b * 4);
	}
	return { batchSize, previous, current, next, actions };
}

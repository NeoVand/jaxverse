/** Owns one local experiment. Simulation labels are confined to diagnostics. */
import { numpy as np, jit, tree, blockUntilReady } from '@jax-js/jax';
import { generateCorpus, type ArmEpisode, type WorldCorpus } from './corpus';
import { type ArmState } from './simulator';
import { renderSensor } from './sensor';
import {
	WORLD_CONFIG,
	encode,
	initWorldParams,
	normalRandom,
	objective,
	parameterCount,
	projectionDirections,
	seededRandom,
	type Tensor,
	type WorldConfig
} from './model';
import { createWorldOptimizer, type WorldBatch, type WorldOptimizer } from './runtime';
import {
	createLatentRollout,
	planLatents,
	type LatentContext,
	type LatentRollout
} from './planner';

export interface WorldInit {
	resolution?: number;
	seed?: number;
	batch?: number;
	lr?: number;
	regularization?: number;
	latent?: number;
	hidden?: number;
	predictorHidden?: number;
	dt?: number;
	trainEpisodes?: number;
	stepsPerEpisode?: number;
}

/** Chosen after browser learning/control experiments; distinct from display framerate. */
export const WORLD_INTERVAL = 0.24;
export const WORLD_TRAINING_STEPS = 5000;
export const WORLD_PLANNING_EFFORT = 0.01;

export interface WorldInfo {
	seed: number;
	step: number;
	dt: number;
	transitions: number;
	validationTransitions: number;
	parameters: number;
	config: WorldConfig;
	batch: number;
	preview: { states: ArmState[]; actions: number[]; frames: Float32Array };
}

export interface WorldMetrics {
	step: number;
	loss: number;
	predictionLoss: number;
	regularizer: number;
	stepMs: number;
	trainingMs: number;
}

export interface ReadoutMetrics {
	/** RMS circular joint error, in radians, on unseen diagnostic frames. */
	poseError: number;
	jointErrors: [number, number];
	examples: number;
	step: number;
}

export interface WorldEvaluation {
	step: number;
	loss: number;
	predictionLoss: number;
	regularizer: number;
	persistenceLoss: number;
	shuffledActionLoss: number;
	noHistoryLoss: number;
	/** Mean per-coordinate standard deviation, with fixed original scale. */
	spread: number;
	minimumStd: number;
	effectiveRank: number;
	/** Fixed first coordinate; chart axes should remain fixed across runs. */
	projection: number[];
	readout: ReadoutMetrics | null;
}

export interface RolloutMetric {
	horizon: number;
	latentMse: number;
	persistenceMse: number;
	/** Includes error from the separate diagnostic readout. */
	poseError: number | null;
}

export interface WorldForecastRequest {
	observations: Float32Array;
	previousAction: readonly [number, number];
	actions: Float32Array;
}

export interface WorldForecast {
	poses: ArmState[];
	latents: Float32Array;
	ms: number;
	readoutError: number;
}

export interface WorldPlanRequest {
	observations: Float32Array;
	previousAction: readonly [number, number];
	goal: Float32Array;
	horizon?: number;
	hold?: boolean;
	seed?: number;
	effort?: number;
}

export interface WorldPlan {
	action: [number, number];
	actions: Float32Array;
	cost: number;
	ms: number;
	poses: ArmState[];
	candidates: { cost: number; poses: ArmState[] }[];
	readoutError: number;
}

type Readout = {
	mean: Float64Array;
	scale: Float64Array;
	weights: Float64Array;
	features: Float64Array;
	width: number;
	metrics: ReadoutMetrics;
};

function circular(a: number): number {
	return Math.atan2(Math.sin(a), Math.cos(a));
}

/** Ridge regression with random nonlinear features. Labels never affect E or P. */
function featureVector(
	z: ArrayLike<number>,
	offset: number,
	d: number,
	readout: Readout
): Float64Array {
	const out = new Float64Array(readout.width);
	out[0] = 1;
	for (let j = 0; j < d; j++) out[1 + j] = (z[offset + j] - readout.mean[j]) / readout.scale[j];
	for (let k = d + 1; k < out.length; k++) {
		let sum = readout.features[(k - d - 1) * (d + 1) + d];
		for (let j = 0; j < d; j++) sum += out[1 + j] * readout.features[(k - d - 1) * (d + 1) + j];
		out[k] = Math.tanh(sum);
	}
	return out;
}

function decodePose(z: ArrayLike<number>, offset: number, d: number, readout: Readout): ArmState {
	const x = featureVector(z, offset, d, readout);
	const target = new Float64Array(4);
	for (let j = 0; j < x.length; j++)
		for (let k = 0; k < 4; k++) target[k] += x[j] * readout.weights[j * 4 + k];
	return {
		q1: Math.atan2(target[1], target[0]),
		q2: Math.atan2(target[3], target[2]),
		v1: 0,
		v2: 0
	};
}

/** Small positive-definite solve in ordinary JS, only for the diagnostic fit. */
function choleskySolve(gram: Float64Array, rhs: Float64Array, width: number): Float64Array {
	const lower = new Float64Array(width * width);
	for (let i = 0; i < width; i++)
		for (let j = 0; j <= i; j++) {
			let sum = gram[i * width + j];
			for (let k = 0; k < j; k++) sum -= lower[i * width + k] * lower[j * width + k];
			lower[i * width + j] = i === j ? Math.sqrt(Math.max(1e-12, sum)) : sum / lower[j * width + j];
		}
	const solution = new Float64Array(rhs);
	for (let i = 0; i < width; i++)
		for (let k = 0; k < 4; k++) {
			for (let j = 0; j < i; j++) solution[i * 4 + k] -= lower[i * width + j] * solution[j * 4 + k];
			solution[i * 4 + k] /= lower[i * width + i];
		}
	for (let i = width - 1; i >= 0; i--)
		for (let k = 0; k < 4; k++) {
			for (let j = i + 1; j < width; j++)
				solution[i * 4 + k] -= lower[j * width + i] * solution[j * 4 + k];
			solution[i * 4 + k] /= lower[i * width + i];
		}
	return solution;
}

export class WorldCore {
	readonly options: WorldInit;
	readonly config: WorldConfig;
	readonly batchSize: number;
	private seed: number;
	private params: Tensor = null;
	private optimizer: WorldOptimizer | null = null;
	private corpus: WorldCorpus | null = null;
	private readout: Readout | null = null;
	private step = 0;
	private trainingMs = 0;
	private dataRandom: () => number;
	private projectionRandom: () => number;
	private encodeCompiled = jit(encode);
	private rollout: LatentRollout;
	private latest: WorldMetrics = {
		step: 0,
		loss: 0,
		predictionLoss: 0,
		regularizer: 0,
		stepMs: 0,
		trainingMs: 0
	};

	constructor(options: WorldInit = {}) {
		this.options = { ...options };
		this.seed = options.seed ?? 17;
		this.config = {
			...WORLD_CONFIG,
			resolution: options.resolution ?? WORLD_CONFIG.resolution,
			hidden: options.hidden ?? WORLD_CONFIG.hidden,
			predictorHidden: options.predictorHidden ?? WORLD_CONFIG.predictorHidden,
			latent: options.latent ?? WORLD_CONFIG.latent,
			regularization: options.regularization ?? WORLD_CONFIG.regularization
		};
		this.batchSize = options.batch ?? 64;
		this.dataRandom = seededRandom(this.seed + 101);
		this.projectionRandom = seededRandom(this.seed + 202);
		this.rollout = createLatentRollout(this.config);
	}

	async init(): Promise<WorldInfo> {
		if (!this.corpus)
			this.corpus = generateCorpus({
				seed: this.seed + 303,
				trainEpisodes: this.options.trainEpisodes ?? 256,
				validationEpisodes: 12,
				size: this.config.resolution,
				stepsPerEpisode: this.options.stepsPerEpisode ?? 64,
				dt: this.options.dt ?? WORLD_INTERVAL
			});
		return this.reset(this.seed);
	}

	/** Fresh parameters and optimizer; immutable observations can be shared safely. */
	async createUnregularizedBaseline(): Promise<WorldCore> {
		if (!this.corpus) throw new Error('Initialize the experiment first');
		const baseline = new WorldCore({ ...this.options, seed: this.seed, regularization: 0 });
		baseline.corpus = this.corpus;
		try {
			await baseline.init();
			return baseline;
		} catch (error) {
			baseline.dispose();
			throw error;
		}
	}

	private info(): WorldInfo {
		if (!this.corpus) throw new Error('Initialize the experiment first');
		const episode = this.corpus.train[0];
		const previewFrames = Math.min(24, episode.states.length);
		return {
			seed: this.seed,
			step: this.step,
			dt: this.corpus.dt,
			transitions: this.corpus.train.reduce((n, e) => n + e.steps, 0),
			validationTransitions: this.corpus.validation.reduce((n, e) => n + e.steps, 0),
			parameters: parameterCount(this.config),
			config: { ...this.config },
			batch: this.batchSize,
			preview: {
				states: episode.states.slice(0, previewFrames),
				actions: Array.from(episode.actions.slice(0, (previewFrames - 1) * 2)),
				frames: episode.frames.slice(0, previewFrames * this.corpus.frameSize)
			}
		};
	}

	async reset(seed = this.seed, regularization = this.config.regularization): Promise<WorldInfo> {
		if (seed !== this.seed)
			this.corpus = generateCorpus({
				seed: seed + 303,
				trainEpisodes: this.options.trainEpisodes ?? 256,
				validationEpisodes: 12,
				size: this.config.resolution,
				stepsPerEpisode: this.options.stepsPerEpisode ?? 64,
				dt: this.options.dt ?? WORLD_INTERVAL
			});
		this.optimizer?.dispose();
		if (this.params) tree.dispose(this.params);
		this.seed = seed;
		this.config.regularization = regularization;
		this.params = initWorldParams(this.config, seed);
		await blockUntilReady(this.params);
		this.optimizer = createWorldOptimizer(this.config, this.batchSize, this.params);
		this.dataRandom = seededRandom(seed + 101);
		this.projectionRandom = seededRandom(seed + 202);
		this.step = 0;
		this.trainingMs = 0;
		this.readout = null;
		this.latest = { step: 0, loss: 0, predictionLoss: 0, regularizer: 0, stepMs: 0, trainingMs: 0 };
		return this.info();
	}

	private batch(
		episodes: ArmEpisode[],
		rand: () => number,
		projectionRand: () => number
	): WorldBatch {
		const size = this.config.resolution ** 2;
		const pixels = new Float32Array(3 * this.batchSize * size);
		const actions = new Float32Array(this.batchSize * 4);
		for (let b = 0; b < this.batchSize; b++) {
			const e = episodes[Math.floor(rand() * episodes.length)];
			const t = 1 + Math.floor(rand() * (e.steps - 1));
			for (let k = 0; k < 3; k++)
				pixels.set(
					e.frames.subarray((t - 1 + k) * size, (t + k) * size),
					(k * this.batchSize + b) * size
				);
			actions.set(e.actions.subarray((t - 1) * 2, (t + 1) * 2), b * 4);
		}
		return { pixels, actions, directions: projectionDirections(this.config, projectionRand) };
	}

	async train(
		steps: number,
		onMetrics?: (metrics: WorldMetrics) => void,
		shouldStop?: () => boolean
	): Promise<WorldMetrics> {
		if (!this.optimizer || !this.corpus) throw new Error('Initialize the experiment first');
		for (let i = 0; i < steps && !shouldStop?.(); i++) {
			const start = performance.now();
			const batch = this.batch(this.corpus.train, this.dataRandom, this.projectionRandom);
			const [loss, next] = this.optimizer.step(this.params, batch, this.options.lr ?? 0.001);
			this.params = next;
			const values = await loss.data();
			if (!Number.isFinite(values[0]))
				throw new Error('Training became non-finite; reset before continuing');
			this.step++;
			const stepMs = performance.now() - start;
			this.trainingMs += stepMs;
			this.latest = {
				step: this.step,
				loss: values[0],
				predictionLoss: values[1],
				regularizer: values[2],
				stepMs,
				trainingMs: this.trainingMs
			};
			onMetrics?.({ ...this.latest });
			// GPU awaits usually yield; this also makes stop responsive on Wasm.
			if (i % 8 === 7) await new Promise((resolve) => setTimeout(resolve, 0));
		}
		this.readout = null;
		return { ...this.latest };
	}

	private async encodePixels(pixels: Float32Array): Promise<Float32Array> {
		const count = pixels.length / this.config.resolution ** 2;
		const result = this.encodeCompiled(
			tree.ref(this.params.encoder),
			np.array(new Float32Array(pixels)).reshape([count, this.config.resolution ** 2])
		);
		const values = new Float32Array(await result.data());
		return values;
	}

	async evaluate(): Promise<WorldEvaluation> {
		if (!this.corpus || !this.params) throw new Error('Initialize the experiment first');
		const b = this.batch(this.corpus.validation, seededRandom(989), seededRandom(121));
		const [total, terms] = objective(
			tree.ref(this.params),
			this.config,
			np.array(b.pixels).reshape([3, this.batchSize, this.config.resolution ** 2]),
			np.array(b.actions).reshape([this.batchSize, 4]),
			np.array(b.directions).reshape([this.config.latent, this.config.projections])
		);
		const [lossValues, values, z] = await Promise.all([
			total.data(),
			terms.data(),
			this.encodePixels(b.pixels)
		]);
		const shuffled = new Float32Array(b.actions);
		for (let i = 0; i < this.batchSize; i++) {
			const from = (i + 17) % this.batchSize;
			shuffled[4 * i + 2] = b.actions[4 * from + 2];
			shuffled[4 * i + 3] = b.actions[4 * from + 3];
		}
		const withoutHistory = new Float32Array(b.pixels);
		const frameBatch = this.batchSize * this.config.resolution ** 2;
		withoutHistory.set(b.pixels.subarray(frameBatch, 2 * frameBatch), 0);
		const diagnosticLoss = async (
			pixels: Float32Array<ArrayBuffer>,
			actions: Float32Array<ArrayBuffer>
		) => {
			const [loss, parts] = objective(
				tree.ref(this.params),
				this.config,
				np.array(pixels).reshape([3, this.batchSize, this.config.resolution ** 2]),
				np.array(actions).reshape([this.batchSize, 4]),
				np.array(b.directions).reshape([this.config.latent, this.config.projections])
			);
			loss.dispose();
			return (await parts.data())[0];
		};
		const shuffledActionLoss = await diagnosticLoss(b.pixels, shuffled);
		const noHistoryLoss = await diagnosticLoss(withoutHistory, b.actions);
		const d = this.config.latent,
			n = z.length / d;
		const mean = new Float64Array(d),
			variance = new Float64Array(d);
		for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += z[i * d + j] / n;
		for (let i = 0; i < n; i++)
			for (let j = 0; j < d; j++) variance[j] += (z[i * d + j] - mean[j]) ** 2 / n;
		// Participation ratio tr(C)^2/tr(C²): 1 for a line, d for equal axes.
		let covarianceSquare = 0;
		for (let j = 0; j < d; j++)
			for (let k = 0; k < d; k++) {
				let cov = 0;
				for (let i = 0; i < n; i++)
					cov += ((z[i * d + j] - mean[j]) * (z[i * d + k] - mean[k])) / n;
				covarianceSquare += cov * cov;
			}
		let persistenceLoss = 0;
		for (let i = 0; i < this.batchSize * d; i++)
			persistenceLoss +=
				(z[this.batchSize * d + i] - z[2 * this.batchSize * d + i]) ** 2 / (this.batchSize * d);
		const trace = variance.reduce((a, b) => a + b, 0);
		return {
			step: this.step,
			loss: lossValues[0],
			predictionLoss: values[0],
			regularizer: values[1],
			persistenceLoss,
			shuffledActionLoss,
			noHistoryLoss,
			spread: variance.reduce((a, b) => a + Math.sqrt(b), 0) / d,
			minimumStd: Math.sqrt(Math.min(...variance)),
			effectiveRank: covarianceSquare > 0 ? (trace * trace) / covarianceSquare : 0,
			projection: Array.from({ length: n }, (_, i) => z[i * d]),
			readout: this.readout?.metrics ?? null
		};
	}

	async fitReadout(): Promise<ReadoutMetrics> {
		if (this.readout?.metrics.step === this.step) return { ...this.readout.metrics };
		// Independent pose samples, not adjacent correlated frames, make the
		// diagnostic fit cover the full pose space. These labels train only it.
		const sample = (count: number, seed: number) => {
			const random = seededRandom(seed);
			const states: ArmState[] = [];
			const pixels = new Float32Array(count * this.config.resolution ** 2);
			for (let i = 0; i < count; i++) {
				const state = {
					q1: (random() * 2 - 1) * Math.PI,
					q2: (random() * 2 - 1) * Math.PI,
					v1: 0,
					v2: 0
				};
				states.push(state);
				pixels.set(renderSensor(state, this.config.resolution), i * this.config.resolution ** 2);
			}
			return { states, pixels };
		};
		const train = sample(1024, this.seed + 909),
			test = sample(256, this.seed + 1919);
		const z = await this.encodePixels(train.pixels),
			testZ = await this.encodePixels(test.pixels);
		const d = this.config.latent,
			width = d + 1 + 256;
		const readout: Readout = {
			mean: new Float64Array(d),
			scale: new Float64Array(d),
			weights: new Float64Array(width * 4),
			features: new Float64Array(256 * (d + 1)),
			width,
			metrics: { poseError: 0, jointErrors: [0, 0], examples: train.states.length, step: this.step }
		};
		const rand = seededRandom(1991);
		for (let i = 0; i < readout.features.length; i++)
			readout.features[i] = normalRandom(rand) * 0.5;
		for (let i = 0; i < train.states.length; i++)
			for (let j = 0; j < d; j++) readout.mean[j] += z[i * d + j] / train.states.length;
		for (let i = 0; i < train.states.length; i++)
			for (let j = 0; j < d; j++)
				readout.scale[j] += (z[i * d + j] - readout.mean[j]) ** 2 / train.states.length;
		for (let j = 0; j < d; j++) readout.scale[j] = Math.max(0.05, Math.sqrt(readout.scale[j]));
		const gram = new Float64Array(width * width),
			rhs = new Float64Array(width * 4);
		for (let i = 0; i < train.states.length; i++) {
			const x = featureVector(z, i * d, d, readout),
				s = train.states[i];
			const y = [Math.cos(s.q1), Math.sin(s.q1), Math.cos(s.q2), Math.sin(s.q2)];
			for (let j = 0; j < width; j++) {
				for (let k = 0; k <= j; k++) gram[j * width + k] += x[j] * x[k];
				for (let k = 0; k < 4; k++) rhs[j * 4 + k] += x[j] * y[k];
			}
		}
		for (let j = 0; j < width; j++) {
			for (let k = 0; k < j; k++) gram[k * width + j] = gram[j * width + k];
			gram[j * width + j] += j === 0 ? 1e-6 : 0.02;
		}
		readout.weights = choleskySolve(gram, rhs, width);
		let error = 0;
		let firstError = 0,
			secondError = 0;
		for (let i = 0; i < test.states.length; i++) {
			const p = decodePose(testZ, i * d, d, readout),
				actual = test.states[i];
			error +=
				(circular(p.q1 - actual.q1) ** 2 + circular(p.q2 - actual.q2) ** 2) /
				(2 * test.states.length);
			firstError += circular(p.q1 - actual.q1) ** 2 / test.states.length;
			secondError += circular(p.q2 - actual.q2) ** 2 / test.states.length;
		}
		readout.metrics.poseError = Math.sqrt(error);
		readout.metrics.jointErrors = [Math.sqrt(firstError), Math.sqrt(secondError)];
		this.readout = readout;
		return { ...readout.metrics };
	}

	/** Fixed held-out episodes, identical recorded actions, no feedback mid-rollout. */
	async evaluateRollouts(): Promise<RolloutMetric[]> {
		if (!this.corpus || !this.params) throw new Error('Initialize the experiment first');
		const maxHorizon = Math.min(12, ...this.corpus.validation.map((episode) => episode.steps - 1));
		const results: RolloutMetric[] = [1, 3, 6, 12]
			.filter((horizon) => horizon <= maxHorizon)
			.map((horizon) => ({
				horizon,
				latentMse: 0,
				persistenceMse: 0,
				poseError: this.readout ? 0 : null
			}));
		const episodes = this.corpus.validation.slice(0, 8);
		const pixels = this.config.resolution ** 2;
		for (const episode of episodes) {
			const context = await this.context(episode.frames.slice(0, 2 * pixels), [
				episode.actions[0],
				episode.actions[1]
			]);
			const predicted = await this.rollout(
				tree.ref(this.params.predictor),
				context,
				episode.actions.slice(2, 2 * (maxHorizon + 1)),
				1,
				maxHorizon
			);
			const actual = await this.encodePixels(
				episode.frames.slice(2 * pixels, (maxHorizon + 2) * pixels)
			);
			const poses = this.poses(predicted);
			for (const result of results) {
				const offset = (result.horizon - 1) * this.config.latent;
				for (let j = 0; j < this.config.latent; j++) {
					result.latentMse +=
						(predicted[offset + j] - actual[offset + j]) ** 2 /
						(episodes.length * this.config.latent);
					result.persistenceMse +=
						(context.current[j] - actual[offset + j]) ** 2 / (episodes.length * this.config.latent);
				}
				if (result.poseError !== null) {
					const estimate = poses[result.horizon - 1],
						state = episode.states[result.horizon + 1];
					result.poseError +=
						(circular(estimate.q1 - state.q1) ** 2 + circular(estimate.q2 - state.q2) ** 2) /
						(2 * episodes.length);
				}
			}
		}
		for (const result of results)
			if (result.poseError !== null) result.poseError = Math.sqrt(result.poseError);
		return results;
	}

	private async context(
		observations: Float32Array,
		previousAction: readonly [number, number]
	): Promise<LatentContext> {
		if (observations.length !== 2 * this.config.resolution ** 2)
			throw new Error('Two complete observation frames are required');
		const z = await this.encodePixels(observations);
		return {
			previous: z.slice(0, this.config.latent),
			current: z.slice(this.config.latent),
			previousAction
		};
	}

	private poses(latents: Float32Array): ArmState[] {
		if (!this.readout) return [];
		return Array.from({ length: latents.length / this.config.latent }, (_, i) =>
			decodePose(latents, i * this.config.latent, this.config.latent, this.readout!)
		);
	}

	async forecast(req: WorldForecastRequest): Promise<WorldForecast> {
		await this.fitReadout();
		const start = performance.now();
		const context = await this.context(req.observations, req.previousAction);
		const horizon = req.actions.length / 2;
		if (!Number.isInteger(horizon) || horizon < 1 || horizon > 32)
			throw new Error('Forecast requires 1–32 torque pairs');
		const latents = await this.rollout(
			tree.ref(this.params.predictor),
			context,
			req.actions,
			1,
			horizon
		);
		return {
			poses: this.poses(latents),
			latents,
			ms: performance.now() - start,
			readoutError: this.readout!.metrics.poseError
		};
	}

	async plan(req: WorldPlanRequest): Promise<WorldPlan> {
		await this.fitReadout();
		const start = performance.now();
		const context = await this.context(req.observations, req.previousAction);
		if (req.goal.length !== this.config.resolution ** 2)
			throw new Error('A complete goal image is required');
		const goal = await this.encodePixels(req.goal);
		const horizon = req.horizon ?? 6;
		if (!Number.isInteger(horizon) || horizon < 2 || horizon > 24)
			throw new Error('Planning horizon must be 2–24');
		const result = await planLatents(
			tree.ref(this.params.predictor),
			context,
			goal,
			this.config,
			{
				horizon,
				hold: req.hold,
				seed: req.seed,
				effort: req.effort ?? WORLD_PLANNING_EFFORT
			},
			this.rollout
		);
		return {
			action: [result.actions[0], result.actions[1]],
			actions: result.actions,
			cost: result.cost,
			ms: performance.now() - start,
			poses: this.poses(result.latents),
			candidates: result.candidates.map((candidate) => ({
				cost: candidate.cost,
				poses: this.poses(candidate.latents)
			})),
			readoutError: this.readout!.metrics.poseError
		};
	}

	dispose(): void {
		this.optimizer?.dispose();
		this.optimizer = null;
		if (this.params) tree.dispose(this.params);
		this.params = null;
		this.corpus = null;
		this.readout = null;
	}
}

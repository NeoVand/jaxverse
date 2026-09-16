/** Latent-only planning. This module deliberately cannot step the simulator. */
import { numpy as np, jit, tree } from '@jax-js/jax';
import { latentGoalCost } from './goal-cost';
export { latentGoalCost } from './goal-cost';
import { normalRandom, predict, seededRandom, type Tensor, type WorldConfig } from './model';

export interface LatentContext {
	previous: Float32Array;
	current: Float32Array;
	previousAction: readonly [number, number];
}

export interface LatentPlanOptions {
	horizon: number;
	hold?: boolean;
	candidates?: number;
	iterations?: number;
	seed?: number;
	/** Optional mean squared torque preference, separate from model training. */
	effort?: number;
}

export interface LatentCandidate {
	cost: number;
	actions: Float32Array;
	latents: Float32Array;
}

export interface LatentPlan extends LatentCandidate {
	candidates: LatentCandidate[];
}

/** Keep a small set of fixed horizon/batch shapes; JIT caches each signature. */
export function createLatentRollout(config: WorldConfig) {
	const compiled = new Map<string, Tensor>();
	return async (
		params: Tensor,
		context: LatentContext,
		actions: Float32Array,
		candidates: number,
		horizon: number
	): Promise<Float32Array> => {
		if (context.current.length !== config.latent)
			throw new Error('Latent context has the wrong width');
		const key = `${candidates}:${horizon}`;
		let run = compiled.get(key);
		if (!run) {
			run = jit(
				(p: Tensor, previous: Tensor, current: Tensor, previousAction: Tensor, a: Tensor) => {
					const outputs: Tensor[] = [];
					for (let t = 0; t < horizon; t++) {
						const action = a.ref.slice([], [2 * t, 2 * t + 2]);
						const next = predict(
							tree.ref(p),
							previous,
							current.ref,
							np.concatenate([previousAction, action.ref], 1)
						);
						previous = current;
						current = next;
						previousAction = action;
						outputs.push(next.ref);
					}
					tree.dispose(p);
					previous.dispose();
					current.dispose();
					previousAction.dispose();
					a.dispose();
					return np.stack(outputs, 1);
				}
			);
			compiled.set(key, run);
		}
		const repeat = (x: Float32Array | readonly number[]) => {
			const out = new Float32Array(candidates * x.length);
			for (let i = 0; i < candidates; i++) out.set(x, i * x.length);
			return np.array(out).reshape([candidates, x.length]);
		};
		const result = run(
			params,
			repeat(context.previous),
			repeat(context.current),
			repeat(context.previousAction),
			np.array(new Float32Array(actions)).reshape([candidates, 2 * horizon])
		);
		const data = new Float32Array(await result.data());
		return data;
	};
}

export type LatentRollout = ReturnType<typeof createLatentRollout>;

/**
 * Bounded CEM. Samples have two-frame torque knots to reduce search dimension.
 * Returns the best sequence actually evaluated, not an unevaluated elite mean.
 */
export async function planLatents(
	params: Tensor,
	context: LatentContext,
	goal: Float32Array,
	config: WorldConfig,
	options: LatentPlanOptions,
	rollout: LatentRollout
): Promise<LatentPlan> {
	const { horizon } = options;
	const count = options.candidates ?? 64;
	const rounds = options.iterations ?? 3;
	if (
		!Number.isInteger(horizon) ||
		horizon < 1 ||
		!Number.isInteger(count) ||
		count < 4 ||
		!Number.isInteger(rounds) ||
		rounds < 1
	)
		throw new Error(
			'CEM requires a positive horizon, at least four candidates, and refinement rounds'
		);
	const knots = Math.ceil(horizon / 2);
	const width = knots * 2;
	const eliteCount = Math.max(4, Math.floor(count / 8));
	const rand = seededRandom(options.seed ?? 37);
	const mean = new Float32Array(width);
	const deviation = new Float32Array(width).fill(0.8);
	let best: LatentCandidate | null = null;
	let shown: LatentCandidate[] = [];
	try {
		for (let round = 0; round < rounds; round++) {
			const samples = new Float32Array(count * width);
			const actions = new Float32Array(count * horizon * 2);
			for (let b = 0; b < count; b++) {
				for (let k = 0; k < width; k++)
					samples[b * width + k] =
						b === 0
							? mean[k]
							: Math.max(-1, Math.min(1, mean[k] + deviation[k] * normalRandom(rand)));
				for (let t = 0; t < horizon; t++)
					for (let j = 0; j < 2; j++)
						actions[(b * horizon + t) * 2 + j] = samples[b * width + Math.floor(t / 2) * 2 + j];
			}
			const predictions = await rollout(tree.ref(params), context, actions, count, horizon);
			const ranked: LatentCandidate[] = [];
			for (let b = 0; b < count; b++) {
				const latents = predictions.slice(
					b * horizon * config.latent,
					(b + 1) * horizon * config.latent
				);
				const candidateActions = actions.slice(b * horizon * 2, (b + 1) * horizon * 2);
				const effort =
					candidateActions.reduce((sum, action) => sum + action * action, 0) /
					candidateActions.length;
				ranked.push({
					cost:
						latentGoalCost(latents, goal, horizon, options.hold ?? false) +
						(options.effort ?? 0) * effort,
					actions: candidateActions,
					latents
				});
			}
			const order = Array.from({ length: count }, (_, i) => i).sort(
				(a, b) => ranked[a].cost - ranked[b].cost
			);
			if (!best || ranked[order[0]].cost < best.cost) best = ranked[order[0]];
			shown = [
				ranked[order[0]],
				ranked[order[Math.floor(count / 3)]],
				ranked[order[Math.floor((2 * count) / 3)]]
			];
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
		if (!best) throw new Error('Planner did not evaluate any candidates');
		return { ...best, candidates: shown };
	} finally {
		tree.dispose(params);
	}
}

// An 8×6 field with a treasure, three pits, a step tax, and a few walls.
// The policy is a table of logits θ[s][4]; learning is REINFORCE with
// return-to-go and a per-state running-mean baseline. Plain loops on plain
// arrays — small enough that the main thread runs hundreds of episodes a
// second with no GPU anywhere in sight.

import type { Rand } from './rng';
import { sampleFrom, softmax } from './softmax';

export const GRID_W = 8;
export const GRID_H = 6;
export const N_ACTIONS = 4;

/** Actions 0..3: up, down, left, right — y grows upward (row 0 is the bottom). */
export const ACTION_DX = [0, 0, -1, 1] as const;
export const ACTION_DY = [1, -1, 0, 0] as const;

export interface World {
	w: number;
	h: number;
	/** Mutable on purpose — the reader relocates it in the map editor. */
	start: number;
	/** Mutable on purpose — the reader drags the treasure mid-training. */
	goal: number;
	/** Mutable sets: the map editor cycles cells empty → wall → pit live. */
	pits: Set<number>;
	walls: Set<number>;
	stepCost: number;
	goalReward: number;
	pitReward: number;
	/** Discount for return-to-go (the chapter's γ — not the learning rate). */
	gamma: number;
	maxSteps: number;
}

export function cellIndex(x: number, y: number, w = GRID_W): number {
	return y * w + x;
}

/** The chapter's world: two wall fences, pits punishing the careless shortcuts. */
export function makeWorld(overrides: Partial<World> = {}): World {
	return {
		w: GRID_W,
		h: GRID_H,
		start: cellIndex(0, 0),
		goal: cellIndex(7, 5),
		walls: new Set([
			cellIndex(2, 1),
			cellIndex(2, 2),
			cellIndex(2, 3),
			cellIndex(5, 2),
			cellIndex(5, 3),
			cellIndex(5, 4)
		]),
		pits: new Set([cellIndex(1, 1), cellIndex(4, 1), cellIndex(6, 4)]),
		stepCost: -0.15,
		goalReward: 10,
		pitReward: -8,
		gamma: 0.96,
		maxSteps: 200,
		...overrides
	};
}

export interface StepResult {
	s: number;
	r: number;
	done: boolean;
}

/** One transition. Walking into a wall or off the field = stay put; the step still costs. */
export function stepEnv(world: World, s: number, a: number): StepResult {
	const x = s % world.w;
	const y = (s / world.w) | 0;
	let nx = x + ACTION_DX[a];
	let ny = y + ACTION_DY[a];
	if (
		nx < 0 ||
		nx >= world.w ||
		ny < 0 ||
		ny >= world.h ||
		world.walls.has(cellIndex(nx, ny, world.w))
	) {
		nx = x;
		ny = y;
	}
	const s2 = cellIndex(nx, ny, world.w);
	if (s2 === world.goal) return { s: s2, r: world.stepCost + world.goalReward, done: true };
	if (world.pits.has(s2)) return { s: s2, r: world.stepCost + world.pitReward, done: true };
	return { s: s2, r: world.stepCost, done: false };
}

/** Fresh logits, all zero — a perfectly uniform (maximally curious) policy. */
export function createTheta(world: World): Float64Array {
	return new Float64Array(world.w * world.h * N_ACTIONS);
}

/** Fresh per-state baseline (running mean of observed returns), all zero. */
export function createBaseline(world: World): Float64Array {
	return new Float64Array(world.w * world.h);
}

/** π(·|s) = softmax over the four logits of state s. */
export function policyAt(theta: Float64Array, s: number, out?: Float64Array): Float64Array {
	return softmax(theta.subarray(s * N_ACTIONS, s * N_ACTIONS + N_ACTIONS), out);
}

export type EpisodeEnd = 'goal' | 'pit' | 'timeout';

export interface Episode {
	/** State each action was taken from: s_0 … s_{T−1}. */
	states: number[];
	actions: number[];
	rewards: number[];
	/** Every cell visited, including the terminal one — for drawing trails. */
	path: number[];
	end: EpisodeEnd;
	/** Undiscounted reward sum — what the sparkline reports. */
	totalReward: number;
	steps: number;
}

/** Roll out one episode from the start cell under the current policy. */
export function runEpisode(world: World, theta: Float64Array, rand: Rand): Episode {
	const states: number[] = [];
	const actions: number[] = [];
	const rewards: number[] = [];
	let s = world.start;
	const path: number[] = [s];
	let end: EpisodeEnd = 'timeout';
	let total = 0;
	const probs = new Float64Array(N_ACTIONS);
	for (let t = 0; t < world.maxSteps; t++) {
		policyAt(theta, s, probs);
		const a = sampleFrom(probs, rand);
		const step = stepEnv(world, s, a);
		states.push(s);
		actions.push(a);
		rewards.push(step.r);
		path.push(step.s);
		total += step.r;
		s = step.s;
		if (step.done) {
			end = s === world.goal ? 'goal' : 'pit';
			break;
		}
	}
	return { states, actions, rewards, path, end, totalReward: total, steps: states.length };
}

/**
 * Discounted return-to-go: G_t = r_t + γ·G_{t+1}, computed backwards.
 * `tail` seeds the recursion past the last reward — 0 for a true terminal
 * end, V(s_final) when the episode was merely cut off by the step limit.
 */
export function returnsToGo(rewards: number[], gamma: number, tail = 0): Float64Array {
	const g = new Float64Array(rewards.length);
	let acc = tail;
	for (let t = rewards.length - 1; t >= 0; t--) {
		acc = rewards[t] + gamma * acc;
		g[t] = acc;
	}
	return g;
}

/** How fast the per-state baseline chases fresh returns. */
export const BASELINE_LR = 0.15;

/**
 * REINFORCE with baseline, one whole episode:
 *
 *   θ[s_t, i] += lr · (G_t − V(s_t)) · (1[i = a_t] − π_i(s_t))
 *
 * V is a running mean of the returns observed from each state; each G_t is
 * judged against the estimate that existed *before* it arrived, then folded
 * in. Return-to-go means an early step is credited with everything that
 * followed it — that is the whole answer to "which move earned the treasure?".
 *
 * Episodes cut off by the step limit bootstrap their tail with V(s_final).
 * Without this, truncation teaches a lie — late steps of a wasted episode
 * have little future left to pay for, so idling near the deadline looks
 * cheap, and the policy can lock into a wall-bumping loop and stay there.
 */
export function reinforceUpdate(
	world: World,
	theta: Float64Array,
	baseline: Float64Array,
	ep: Pick<Episode, 'states' | 'actions' | 'rewards' | 'path' | 'end'>,
	lr: number
): void {
	const tail = ep.end === 'timeout' ? baseline[ep.path[ep.path.length - 1]] : 0;
	const g = returnsToGo(ep.rewards, world.gamma, tail);
	const probs = new Float64Array(N_ACTIONS);
	for (let t = 0; t < ep.states.length; t++) {
		const s = ep.states[t];
		const a = ep.actions[t];
		policyAt(theta, s, probs);
		const adv = g[t] - baseline[s];
		for (let i = 0; i < N_ACTIONS; i++) {
			theta[s * N_ACTIONS + i] += lr * adv * ((i === a ? 1 : 0) - probs[i]);
		}
		baseline[s] += BASELINE_LR * (g[t] - baseline[s]);
	}
}

/**
 * Fewest steps from `from` to `to`, walking around walls and pits (a pit
 * ends the episode, so no sane route enters one). −1 when unreachable.
 */
export function bfsSteps(world: World, from: number, to: number): number {
	if (from === to) return 0;
	const dist = new Int32Array(world.w * world.h).fill(-1);
	dist[from] = 0;
	const queue = [from];
	for (let head = 0; head < queue.length; head++) {
		const s = queue[head];
		const x = s % world.w;
		const y = (s / world.w) | 0;
		for (let a = 0; a < N_ACTIONS; a++) {
			const nx = x + ACTION_DX[a];
			const ny = y + ACTION_DY[a];
			if (nx < 0 || nx >= world.w || ny < 0 || ny >= world.h) continue;
			const s2 = cellIndex(nx, ny, world.w);
			if (dist[s2] !== -1 || world.walls.has(s2)) continue;
			if (world.pits.has(s2) && s2 !== to) continue;
			dist[s2] = dist[s] + 1;
			if (s2 === to) return dist[s2];
			queue.push(s2);
		}
	}
	return -1;
}

/** Best achievable undiscounted episode return: goal bonus minus the step tax. */
export function optimalReturn(world: World): number {
	const steps = bfsSteps(world, world.start, world.goal);
	return steps < 0 ? NaN : world.goalReward + world.stepCost * steps;
}

import { describe, expect, it } from 'vitest';
import type { WorldPlan } from './engine';
import { rescoreFutures } from './future-choices';

describe('asking a new question of fixed predicted futures', () => {
	it('changes the chosen action with the goal or scoring window while preserving all forecasts', () => {
		const pose = { q1: 0, q2: 0, v1: 0, v2: 0 };
		const candidates = [
			{
				cost: 0,
				poses: [pose, pose],
				latents: new Float32Array([0, 1]),
				actions: new Float32Array([1, 0, 1, 0])
			},
			{
				cost: 1,
				poses: [pose, pose],
				latents: new Float32Array([0.8, 0.8]),
				actions: new Float32Array([0, 1, 0, 1])
			},
			{
				cost: 2,
				poses: [pose, pose],
				latents: new Float32Array([0, 0]),
				actions: new Float32Array([-1, 0, -1, 0])
			}
		];
		const source: WorldPlan = {
			action: [1, 0],
			actions: candidates[0].actions,
			poses: candidates[0].poses,
			cost: 0,
			ms: 5,
			candidates,
			comparisonGoals: [new Float32Array([1]), new Float32Array([0])],
			readoutError: 0.1
		};
		const reach = rescoreFutures(source, 0, false);
		const stay = rescoreFutures(source, 0, true);
		const newGoal = rescoreFutures(source, 1, false);
		expect(reach.selectedIndex).toBe(0);
		expect(stay.selectedIndex).toBe(1);
		expect(newGoal.selectedIndex).toBe(2);
		expect(stay.plan.action).toEqual([0, 1]);
		expect(newGoal.plan.action).toEqual([-1, 0]);
		for (const result of [reach, stay, newGoal])
			result.plan.candidates.forEach((candidate, i) => {
				expect(candidate.latents).toBe(source.candidates[i].latents);
				expect(candidate.poses).toBe(source.candidates[i].poses);
				expect(candidate.actions).toBe(source.candidates[i].actions);
			});
		expect(source.candidates.map((candidate) => candidate.cost)).toEqual([0, 1, 2]);
	});
});

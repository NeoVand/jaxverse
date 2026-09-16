import { describe, expect, it } from 'vitest';
import { latentGoalCost, planLatents } from './planner';
import { WORLD_CONFIG } from './model';

describe('latent planning', () => {
	it('distinguishes touching the pose at the last moment from remaining near it', () => {
		const path = new Float32Array([3, 3, 2, 2, 0, 0]);
		const goal = new Float32Array([0, 0]);
		expect(latentGoalCost(path, goal, 3, false)).toBe(0);
		expect(latentGoalCost(path, goal, 3, true)).toBeCloseTo(13 / 3);
	});

	it('returns bounded, evaluated actions and improves a simple latent goal', async () => {
		const config = { ...WORLD_CONFIG, latent: 2 };
		const seen: number[][] = [];
		const result = await planLatents(
			{},
			{ previous: new Float32Array(2), current: new Float32Array(2), previousAction: [0, 0] },
			new Float32Array([0.5, -0.5]),
			config,
			{ horizon: 4, candidates: 32, iterations: 3, seed: 12 },
			async (_p, _context, actions) => {
				seen.push(Array.from(actions));
				return new Float32Array(actions);
			}
		);
		expect(Array.from(result.actions).every((a) => a >= -1 && a <= 1)).toBe(true);
		expect(result.cost).toBeLessThan(0.05);
		expect(
			seen.some((batch) => {
				for (let i = 0; i < batch.length; i += 8)
					if (result.actions.every((a, j) => a === batch[i + j])) return true;
				return false;
			})
		).toBe(true);
	});
});

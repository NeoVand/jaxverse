import { describe, expect, it } from 'vitest';
import {
	bfsSteps,
	cellIndex,
	createBaseline,
	createTheta,
	makeWorld,
	optimalReturn,
	reinforceUpdate,
	returnsToGo,
	runEpisode,
	stepEnv
} from './gridworld';
import { mulberry32 } from './rng';

describe('gridworld environment', () => {
	it('walls block, borders block, pits and goal terminate with their rewards', () => {
		const w = makeWorld();
		// down from the start leaves the field → stay put, still pay the tax
		const stay = stepEnv(w, w.start, 1);
		expect(stay).toEqual({ s: w.start, r: w.stepCost, done: false });
		// right from (1,1)? — (1,1) is a pit; instead test walking INTO a pit
		const intoPit = stepEnv(w, cellIndex(1, 0), 0); // up into the pit at (1,1)
		expect(intoPit.done).toBe(true);
		expect(intoPit.r).toBeCloseTo(w.stepCost + w.pitReward, 12);
		// right from (1,1)'s neighbor into the wall at (2,1) → stay
		const intoWall = stepEnv(w, cellIndex(2, 0), 0); // up from (2,0) is open…
		expect(intoWall.done).toBe(false);
		const blocked = stepEnv(w, cellIndex(1, 2), 3); // right into wall (2,2)
		expect(blocked.s).toBe(cellIndex(1, 2));
		// stepping onto the goal pays the bonus and ends the episode
		const win = stepEnv(w, cellIndex(7, 4), 0);
		expect(win).toEqual({ s: w.goal, r: w.stepCost + w.goalReward, done: true });
	});

	it('returns-to-go discounts correctly and BFS finds the 12-step optimum', () => {
		const g = returnsToGo([1, 0, 2], 0.5);
		expect(g[2]).toBeCloseTo(2, 12);
		expect(g[1]).toBeCloseTo(1, 12);
		expect(g[0]).toBeCloseTo(1.5, 12);

		const w = makeWorld();
		expect(bfsSteps(w, w.start, w.goal)).toBe(12);
		expect(optimalReturn(w)).toBeCloseTo(10 - 0.15 * 12, 12);
		// dragging the treasure changes the optimum the same way the demo does
		const moved = makeWorld({ goal: cellIndex(0, 5) });
		expect(bfsSteps(moved, moved.start, moved.goal)).toBe(5);
		expect(optimalReturn(moved)).toBeCloseTo(10 - 0.15 * 5, 12);
	});

	it('REINFORCE converges at the demo default (lr 0.1): ≥90% success, near-optimal returns', () => {
		// Two seeds so a single lucky run can't pass for convergence.
		for (const seed of [7, 3]) {
			const world = makeWorld();
			const theta = createTheta(world);
			const baseline = createBaseline(world);
			const rand = mulberry32(seed);
			const wins: number[] = [];
			const rets: number[] = [];
			let solvedAt = -1;
			for (let e = 0; e < 800; e++) {
				const ep = runEpisode(world, theta, rand);
				reinforceUpdate(world, theta, baseline, ep, 0.1);
				wins.push(ep.end === 'goal' ? 1 : 0);
				rets.push(ep.totalReward);
				if (solvedAt === -1 && e >= 50) {
					const w50 = wins.slice(-50).reduce((a, b) => a + b, 0);
					if (w50 >= 45) solvedAt = e;
				}
			}
			const last50 = wins.slice(-50).reduce((a, b) => a + b, 0) / 50;
			const meanRet = rets.slice(-50).reduce((a, b) => a + b, 0) / 50;
			expect(last50).toBeGreaterThanOrEqual(0.9);
			// solves quickly — the demo runs ~40 episodes/s, so this is a few seconds
			expect(solvedAt).toBeGreaterThan(-1);
			expect(solvedAt).toBeLessThan(300);
			// settles within reach of the BFS optimum (8.2), not on some long detour
			expect(meanRet).toBeGreaterThan(optimalReturn(world) - 2.5);
		}
	});
});

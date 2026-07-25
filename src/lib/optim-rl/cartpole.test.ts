import { describe, expect, it } from 'vitest';
import {
	cartReinforceUpdate,
	createCartBaseline,
	createCartTheta,
	isFailed,
	physicsStep,
	runCartEpisode,
	TH_LIMIT
} from './cartpole';
import { mulberry32 } from './rng';

describe('cartpole', () => {
	it('physics: an unforced pole tips over, and an untrained policy falls fast', () => {
		// deterministic: slight lean, no force → the lean grows until failure
		const s = { x: 0, xd: 0, th: 0.05, thd: 0 };
		let steps = 0;
		while (!isFailed(s) && steps < 500) {
			physicsStep(s, 0);
			steps++;
		}
		expect(Math.abs(s.th)).toBeGreaterThan(TH_LIMIT);
		expect(steps).toBeLessThan(150);

		// a uniform-random policy survives only briefly — this is the "before"
		const theta = createCartTheta();
		const rand = mulberry32(9);
		let total = 0;
		let worst = 0;
		for (let e = 0; e < 100; e++) {
			const ep = runCartEpisode(theta, rand);
			total += ep.steps;
			worst = Math.max(worst, ep.steps);
		}
		expect(total / 100).toBeLessThan(80);
		expect(worst).toBeLessThan(200);
	});

	it('REINFORCE improves from ~30 steps to near the 500 ceiling in 300 episodes', () => {
		const theta = createCartTheta();
		const baseline = createCartBaseline();
		const rand = mulberry32(42);
		const rets: number[] = [];
		for (let e = 0; e < 300; e++) {
			const ep = runCartEpisode(theta, rand);
			cartReinforceUpdate(theta, baseline, ep, 0.05);
			rets.push(ep.steps);
		}
		const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
		const first = mean(rets.slice(0, 50));
		const last = mean(rets.slice(-50));
		expect(last).toBeGreaterThan(400);
		expect(last).toBeGreaterThan(first * 2);
	});
});

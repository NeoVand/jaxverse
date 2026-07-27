import { describe, expect, it } from 'vitest';
import {
	createDpoleBaseline,
	createDpoleTheta,
	dpoleReinforceUpdate,
	GRAVITY,
	HINGE_MASS,
	L1,
	L2,
	M1,
	M2,
	physicsStep,
	resetDpole,
	runDpoleEpisode,
	tipHeight,
	type DpoleEpisode,
	type DpoleState
} from './dpole';
import { mulberry32 } from './rng';

function energy(s: DpoleState): number {
	const v1x = s.xd + L1 * Math.cos(s.th1) * s.th1d;
	const v1y = -L1 * Math.sin(s.th1) * s.th1d;
	const v2x = v1x + L2 * Math.cos(s.th2) * s.th2d;
	const v2y = v1y - L2 * Math.sin(s.th2) * s.th2d;
	const T =
		0.5 * HINGE_MASS * s.xd * s.xd +
		0.5 * M1 * (v1x * v1x + v1y * v1y) +
		0.5 * M2 * (v2x * v2x + v2y * v2y);
	const V =
		M1 * GRAVITY * L1 * Math.cos(s.th1) +
		M2 * GRAVITY * (L1 * Math.cos(s.th1) + L2 * Math.cos(s.th2));
	return T + V;
}

const meanOf = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;

describe('double pendulum swing-up', () => {
	it('physics: RK4 conserves energy through 10 s of unforced chaos', () => {
		// a big lean, no force — full-swing chaotic regime, the integrator's
		// worst case (and exactly what the stage shows while it swings)
		const s: DpoleState = { x: 0, xd: 0, th1: 2.6, th1d: 0, th2: 1.2, th2d: 0 };
		const e0 = energy(s);
		for (let i = 0; i < 500; i++) physicsStep(s, 0);
		expect(Math.abs((energy(s) - e0) / e0)).toBeLessThan(1e-4);
	});

	it('an untrained (uniform) policy keeps the tip low', () => {
		const theta = createDpoleTheta();
		const rand = mulberry32(9);
		let h = 0;
		let caught = 0;
		for (let e = 0; e < 30; e++) {
			const ep = runDpoleEpisode(theta, rand, 0);
			caught += ep.caught;
			// undiscounted return ≈ mean height; use it as the height proxy
			h += ep.ret / ep.steps;
		}
		// hanging start + random flailing: tip mostly below the rail
		expect(h / 30).toBeLessThan(0.6);
		expect(caught / 30).toBeLessThan(10);
	});

	it('REINFORCE learns the hold from balanced starts', () => {
		const theta = createDpoleTheta();
		const baseline = createDpoleBaseline();
		const rand = mulberry32(512);
		for (let e = 0; e < 1400; e++) {
			const batch: DpoleEpisode[] = [];
			for (let i = 0; i < 8; i++) batch.push(runDpoleEpisode(theta, rand, 1));
			dpoleReinforceUpdate(theta, baseline, batch, 0.15, 0.995);
		}
		// an untrained policy falls out of the cones in ~25 ticks; a trained
		// one should survive most of the episode
		const hold: number[] = [];
		for (let e = 0; e < 20; e++) hold.push(runDpoleEpisode(theta, rand, 1).steps);
		expect(meanOf(hold)).toBeGreaterThan(150);
	});

	it('the mixed curriculum makes the swing-up pay', () => {
		const theta = createDpoleTheta();
		const baseline = createDpoleBaseline();
		const rand = mulberry32(512);
		let firstHang = 0;
		let firstN = 0;
		for (let e = 0; e < 1500; e++) {
			const batch: DpoleEpisode[] = [];
			for (let i = 0; i < 8; i++) batch.push(runDpoleEpisode(theta, rand));
			dpoleReinforceUpdate(theta, baseline, batch, 0.15, 0.995);
			if (e < 63)
				for (const ep of batch)
					if (ep.kind === 0) {
						firstHang += ep.ret;
						firstN++;
					}
		}
		// hanging-start returns should clearly beat untrained ones
		const hang: number[] = [];
		for (let e = 0; e < 20; e++) hang.push(runDpoleEpisode(theta, rand, 0).ret);
		expect(meanOf(hang)).toBeGreaterThan((firstHang / Math.max(firstN, 1)) * 1.5);
	});

	it('reward: the summit pays far more than the swing', () => {
		const up = resetDpole(() => 0.5, 1); // exactly upright, still
		const hang = resetDpole(() => 0.5, 0); // exactly hanging, still
		expect(tipHeight(up)).toBeCloseTo(1, 5);
		expect(tipHeight(hang)).toBeCloseTo(-1, 5);
	});
});

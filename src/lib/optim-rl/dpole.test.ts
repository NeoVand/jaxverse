import { describe, expect, it } from 'vitest';
import {
	createDpoleBaseline,
	createDpoleTheta,
	deliveryBonus,
	DpoleCurriculum,
	dpoleReinforceUpdate,
	E_TOP,
	GRAVITY,
	HANDOFF_H01,
	HINGE_MASS,
	L1,
	L2,
	M1,
	M2,
	mechanicalEnergy,
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
	it('physics: energy decays slowly through 10 s of unforced chaos', () => {
		// a big lean, no force — full-swing chaotic regime, the integrator's
		// worst case (and exactly what the stage shows while it swings). The
		// pins carry a whisper of friction, so energy must fall — but only
		// by a few percent over ten full seconds.
		const s: DpoleState = { x: 0, xd: 0, th1: 2.6, th1d: 0, th2: 1.2, th2d: 0 };
		const e0 = energy(s);
		let prev = e0;
		for (let i = 0; i < 500; i++) {
			physicsStep(s, 0);
			const e = energy(s);
			expect(e).toBeLessThanOrEqual(prev + 1e-9); // monotone: friction only takes
			prev = e;
		}
		const drop = (e0 - energy(s)) / Math.abs(e0);
		expect(drop).toBeGreaterThan(0.001);
		expect(drop).toBeLessThan(0.25);
	});

	it('physics: a runaway propeller bleeds out, not whirls forever', () => {
		// one missed catch used to leave the stack over-energized FOREVER;
		// with spin² pin drag even a violent whirl (30× the top's energy)
		// must drain to near-swingable within fifteen seconds
		const s: DpoleState = { x: 0, xd: 0, th1: 0, th1d: 30, th2: Math.PI, th2d: -35 };
		for (let i = 0; i < 750; i++) physicsStep(s, 0);
		expect(mechanicalEnergy(s)).toBeLessThan(2 * E_TOP + 1);
		expect(Math.abs(s.th1d) + Math.abs(s.th2d)).toBeLessThan(25);
	});

	it('an untrained (uniform) policy neither delivers nor gets caught', () => {
		const theta = createDpoleTheta();
		const rand = mulberry32(9);
		let ret = 0;
		let caught = 0;
		let delivered = 0;
		for (let e = 0; e < 30; e++) {
			const ep = runDpoleEpisode(theta, rand, 0);
			caught += ep.caught;
			ret += ep.ret;
			if (ep.delivered) delivered++;
		}
		// hanging start + random flailing: the clock tax dominates
		expect(ret / 30).toBeLessThan(0);
		expect(caught / 30).toBeLessThan(10);
		expect(delivered / 30).toBeLessThan(0.5);
	});

	it('delivery bonus prefers a slow arrival', () => {
		const calm: DpoleState = { x: 0, xd: 0, th1: 0.4, th1d: 1, th2: 0.4, th2d: 1 };
		const wild: DpoleState = { ...calm, th1d: 6, th2d: 6 };
		expect(deliveryBonus(calm)).toBeGreaterThan(5 * deliveryBonus(wild));
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
		// an untrained policy drops the stack in a few dozen ticks; a
		// trained one should survive most of the episode
		const hold: number[] = [];
		for (let e = 0; e < 20; e++) hold.push(runDpoleEpisode(theta, rand, 1).steps);
		expect(meanOf(hold)).toBeGreaterThan(150);
	});

	it('the curriculum teaches the swing to deliver', () => {
		const theta = createDpoleTheta();
		const baseline = createDpoleBaseline();
		const curriculum = new DpoleCurriculum();
		const rand = mulberry32(512);
		let firstHang = 0;
		let firstN = 0;
		for (let e = 0; e < 1500; e++) {
			const batch: DpoleEpisode[] = [];
			for (let i = 0; i < 8; i++) batch.push(curriculum.next(theta, rand));
			dpoleReinforceUpdate(theta, baseline, batch, 0.15, 0.995);
			if (e < 63)
				for (const ep of batch)
					if (ep.kind === 0) {
						firstHang += ep.ret;
						firstN++;
					}
		}
		// hanging-start returns are dominated by the delivery bonus: an
		// improvement here means real, calmer arrivals at the hand-off
		const hang: number[] = [];
		let delivered = 0;
		for (let e = 0; e < 20; e++) {
			const ep = runDpoleEpisode(theta, rand, 0);
			hang.push(ep.ret);
			if (ep.delivered && (tipHeight(ep.delivered) + 1) / 2 > HANDOFF_H01) delivered++;
		}
		expect(meanOf(hang)).toBeGreaterThan(firstHang / Math.max(firstN, 1) + 25);
		expect(delivered).toBeGreaterThan(14);
	});

	it('reward: the summit pays far more than the swing', () => {
		const up = resetDpole(() => 0.5, 1); // exactly upright, still
		const hang = resetDpole(() => 0.5, 0); // exactly hanging, still
		expect(tipHeight(up)).toBeCloseTo(1, 5);
		expect(tipHeight(hang)).toBeCloseTo(-1, 5);
	});
});

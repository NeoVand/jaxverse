import { describe, expect, it } from 'vitest';
import {
	armPoints,
	energy,
	initialArm,
	LINK_LENGTHS,
	poseError,
	stepArm,
	type ArmState
} from './simulator';

function independentKineticEnergy(s: ArmState): number {
	const [l1, l2] = LINK_LENGTHS;
	const elbowVx = -l1 * Math.sin(s.q1) * s.v1;
	const elbowVy = l1 * Math.cos(s.q1) * s.v1;
	const com2Vx = elbowVx - (l2 / 2) * Math.sin(s.q1 + s.q2) * (s.v1 + s.v2);
	const com2Vy = elbowVy + (l2 / 2) * Math.cos(s.q1 + s.q2) * (s.v1 + s.v2);
	return (
		0.5 * (l1 / 2) ** 2 * s.v1 ** 2 +
		0.5 * ((l1 * l1) / 12) * s.v1 ** 2 +
		0.5 * 0.8 * (com2Vx * com2Vx + com2Vy * com2Vy) +
		0.5 * ((0.8 * l2 * l2) / 12) * (s.v1 + s.v2) ** 2
	);
}

function difference(a: ArmState, b: ArmState): number {
	return Math.hypot(poseError(a, b), a.v1 - b.v1, a.v2 - b.v2);
}

describe('horizontal two-link dynamics', () => {
	it('uses relative elbow angles and preserves both rigid lengths', () => {
		const points = armPoints({ q1: 0, q2: Math.PI / 2 });
		expect(points.elbow.x).toBeCloseTo(LINK_LENGTHS[0], 12);
		expect(points.elbow.y).toBeCloseTo(0, 12);
		expect(points.tip.x).toBeCloseTo(LINK_LENGTHS[0], 12);
		expect(points.tip.y).toBeCloseTo(LINK_LENGTHS[1], 12);
	});

	it('has the same energy as an independent center-of-mass calculation', () => {
		const s = { q1: 0.8, q2: -1.3, v1: 1.2, v2: -2.6 };
		expect(energy(s)).toBeCloseTo(independentKineticEnergy(s), 12);
	});

	it('conserves energy in an undriven, undamped mechanism', () => {
		let s = { q1: 0.7, q2: 1.2, v1: 1.1, v2: -2.2 };
		const e0 = energy(s);
		let largestRelativeDrift = 0;
		for (let t = 0; t < 400; t++) {
			s = stepArm(s, [0, 0], { damping: 0, substeps: 8 });
			largestRelativeDrift = Math.max(largestRelativeDrift, Math.abs(energy(s) - e0) / e0);
		}
		expect(largestRelativeDrift).toBeLessThan(1e-6);
	});

	it('converges under timestep refinement with the same held motor commands', () => {
		function run(substeps: number): ArmState {
			let s = { q1: 0.4, q2: 1.3, v1: 1.5, v2: -2 };
			for (let t = 0; t < 60; t++)
				s = stepArm(s, [Math.sin(t * 0.23), Math.cos(t * 0.17)], { substeps });
			return s;
		}
		const reference = run(32);
		const coarse = difference(run(1), reference);
		const medium = difference(run(2), reference);
		const fine = difference(run(4), reference);
		expect(medium).toBeLessThan(coarse / 8);
		expect(fine).toBeLessThan(medium / 8);
		expect(fine).toBeLessThan(1e-5);
	});

	it('damping dissipates energy and a resting arm has no hidden restoring force', () => {
		let s = { q1: -0.4, q2: 0.7, v1: 1.4, v2: -0.8 };
		const e0 = energy(s);
		for (let t = 0; t < 120; t++) {
			const before = energy(s);
			s = stepArm(s, [0, 0]);
			expect(energy(s)).toBeLessThanOrEqual(before + 1e-10);
		}
		expect(energy(s)).toBeLessThan(e0 * 0.05);
		const rest = initialArm();
		expect(difference(stepArm(rest, [0, 0]), rest)).toBeLessThan(1e-12);
	});

	it('the second motor couples into the first joint, without mutating inputs', () => {
		const s = initialArm();
		const original = { ...s };
		const driven = stepArm(s, [0, 1]);
		expect(Math.abs(driven.v1)).toBeGreaterThan(1e-3);
		expect(driven.v2).toBeGreaterThan(0);
		expect(s).toEqual(original);
		expect(stepArm(s, [0, 20])).toEqual(driven);
	});

	it('measures pose across the angular wrap without an artificial discontinuity', () => {
		expect(poseError({ q1: Math.PI - 0.01, q2: 0 }, { q1: -Math.PI + 0.01, q2: 0 })).toBeCloseTo(
			0.02 / Math.SQRT2,
			12
		);
	});
});

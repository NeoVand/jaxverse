import { describe, expect, it } from 'vitest';
import { createForecastStart, forecastCommands } from './forecast-scenario';
import { energy, initialArm, poseError, stepArm } from './simulator';

describe('the shared moving start for forecast comparisons', () => {
	it('supplies two genuinely successive observations and their connecting action', () => {
		const start = createForecastStart(0.24);
		expect(stepArm(start.previous, start.previousAction, { dt: 0.24 })).toEqual(start.current);
		expect(poseError(start.previous, start.current)).toBeGreaterThan(0.1);
		expect(energy(start.current)).toBeGreaterThan(0);
		expect(createForecastStart(0.24)).toEqual(start);
	});

	it('coasts visibly after release while damping dissipates energy', () => {
		const start = createForecastStart(0.24);
		const released = forecastCommands('release', 20);
		expect([...released].every((value) => value === 0)).toBe(true);
		let actual = start.current;
		for (let i = 0; i < 20; i++) {
			const before = energy(actual);
			actual = stepArm(actual, [released[2 * i], released[2 * i + 1]], { dt: 0.24 });
			expect(energy(actual)).toBeLessThan(before);
			if (i === 2) expect(poseError(start.current, actual)).toBeGreaterThan(0.15);
		}
		expect(poseError(start.current, actual)).toBeGreaterThan(0.3);
		expect(poseError(initialArm(), stepArm(initialArm(), [0, 0], { dt: 0.24 }))).toBeLessThan(
			1e-12
		);
	});

	it('keeps pushing and reversing opposite for the entire selected horizon', () => {
		const forward = forecastCommands('push', 20);
		const reverse = forecastCommands('reverse', 20);
		const start = createForecastStart(0.24);
		for (let i = 0; i < forward.length; i++) {
			expect(forward[i]).toBe(start.previousAction[i % 2]);
			expect(reverse[i]).toBe(-forward[i]);
		}
	});
});

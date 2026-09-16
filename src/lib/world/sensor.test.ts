import { describe, expect, it } from 'vitest';
import { renderSensor, renderSensorInto, SENSOR_PIXELS } from './sensor';
import { armPoints, initialArm, LINK_LENGTHS, stepArm } from './simulator';

function squaredDifference(a: Float32Array, b: Float32Array): number {
	return a.reduce((sum, value, i) => sum + (value - b[i]) ** 2, 0);
}

describe('world-model visual sensor', () => {
	it('maps positive world y toward the top row of the camera', () => {
		const up = renderSensor({ q1: Math.PI / 2, q2: 0 });
		const down = renderSensor({ q1: -Math.PI / 2, q2: 0 });
		let upperInk = 0;
		let lowerInk = 0;
		for (let row = 0; row < 32; row++) {
			for (let column = 0; column < 32; column++) {
				const index = row * 32 + column;
				if (row < 16) upperInk += 1 - up[index];
				else lowerInk += 1 - up[index];
				expect(up[index]).toBeCloseTo(down[(31 - row) * 32 + column], 6);
			}
		}
		expect(upperInk).toBeGreaterThan(5 * lowerInk);
	});

	it('is a fixed grayscale raster with smooth edge coverage', () => {
		const frame = renderSensor(initialArm());
		expect(frame).toHaveLength(SENSOR_PIXELS);
		expect(frame.every((value) => value >= 0 && value <= 1)).toBe(true);
		expect(frame.filter((value) => value < 0.9).length).toBeGreaterThan(20);
		expect(frame.filter((value) => value < 0.9).length).toBeLessThan(150);
		expect(new Set(frame).size).toBeGreaterThan(20);
	});

	it('keeps velocity out of a still picture but reveals it in successive observations', () => {
		const forward = { ...initialArm(), v1: 0.8, v2: -0.6 };
		const backward = { ...forward, v1: -forward.v1, v2: -forward.v2 };
		expect(renderSensor(forward)).toEqual(renderSensor(backward));
		const a = renderSensor(stepArm(forward, [0, 0]));
		const b = renderSensor(stepArm(backward, [0, 0]));
		expect(squaredDifference(a, b)).toBeGreaterThan(1);
	});

	it('distinguishes elbow configurations with exactly the same tip location', () => {
		const [l1, l2] = LINK_LENGTHS;
		const q2 = Math.acos((0.5 ** 2 - l1 * l1 - l2 * l2) / (2 * l1 * l2));
		const q1 = -Math.atan2(l2 * Math.sin(q2), l1 + l2 * Math.cos(q2));
		const a = { q1, q2 };
		const b = { q1: -q1, q2: -q2 };
		expect(armPoints(a).tip.x).toBeCloseTo(armPoints(b).tip.x, 12);
		expect(armPoints(a).tip.y).toBeCloseTo(armPoints(b).tip.y, 12);
		expect(squaredDifference(renderSensor(a), renderSensor(b))).toBeGreaterThan(5);
	});

	it('resolves nearby poses and angular wrapping in the actual low-resolution input', () => {
		const s = initialArm();
		expect(
			squaredDifference(renderSensor(s), renderSensor({ ...s, q1: s.q1 + 0.08 }))
		).toBeGreaterThan(0.1);
		expect(
			squaredDifference(renderSensor(s), renderSensor({ ...s, q1: s.q1 + 2 * Math.PI }))
		).toBeLessThan(1e-12);
	});

	it('writes a frame bank without touching adjacent frames', () => {
		const bank = new Float32Array(3 * SENSOR_PIXELS).fill(-1);
		renderSensorInto(initialArm(), bank, SENSOR_PIXELS);
		expect(bank.subarray(0, SENSOR_PIXELS).every((value) => value === -1)).toBe(true);
		expect(bank.subarray(2 * SENSOR_PIXELS).every((value) => value === -1)).toBe(true);
		expect(bank.subarray(SENSOR_PIXELS, 2 * SENSOR_PIXELS)).toEqual(renderSensor(initialArm()));
	});
});

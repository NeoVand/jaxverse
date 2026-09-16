import { armPoints, type ArmState, type Point } from './simulator';

export const SENSOR_SIZE = 32;
export const SENSOR_PIXELS = SENSOR_SIZE * SENSOR_SIZE;
export const SENSOR_EXTENT = 1;

function coverage(distance: number, radius: number, pixelWidth: number): number {
	return Math.max(0, Math.min(1, 0.5 + (radius - distance) / pixelWidth));
}

function segmentDistance(x: number, y: number, a: Point, b: Point): number {
	const dx = b.x - a.x;
	const dy = b.y - a.y;
	const t = Math.max(0, Math.min(1, ((x - a.x) * dx + (y - a.y) * dy) / (dx * dx + dy * dy)));
	return Math.hypot(x - (a.x + t * dx), y - (a.y + t * dy));
}

/**
 * Fixed-camera grayscale observation, background 1 and ink near 0.
 * No velocity, goal, trace, predictions, or UI reaches this raster.
 */
export function renderSensor(state: Pick<ArmState, 'q1' | 'q2'>, size = SENSOR_SIZE): Float32Array {
	const image = new Float32Array(size * size);
	renderSensorInto(state, image, 0, size);
	return image;
}

/** Writes directly into a frame bank to avoid per-frame allocations during collection. */
export function renderSensorInto(
	state: Pick<ArmState, 'q1' | 'q2'>,
	output: Float32Array,
	offset = 0,
	size = SENSOR_SIZE
): void {
	if (!Number.isInteger(size) || size < 8 || size > 256)
		throw new RangeError('Sensor size must be an integer in [8, 256].');
	if (!Number.isInteger(offset) || offset < 0 || offset + size * size > output.length)
		throw new RangeError('Sensor output does not contain a complete frame.');
	if (!Number.isFinite(state.q1) || !Number.isFinite(state.q2))
		throw new RangeError('Sensor angles must be finite.');
	const { base, elbow, tip } = armPoints(state);
	const pixelWidth = (2 * SENSOR_EXTENT) / size;
	for (let py = 0; py < size; py++) {
		const y = SENSOR_EXTENT - (py + 0.5) * pixelWidth;
		for (let px = 0; px < size; px++) {
			const x = (px + 0.5) * pixelWidth - SENSOR_EXTENT;
			let value = 1;
			const blend = (amount: number, ink: number) => {
				value += amount * (ink - value);
			};
			blend(coverage(segmentDistance(x, y, base, elbow), 0.034, pixelWidth), 0.3);
			blend(coverage(segmentDistance(x, y, elbow, tip), 0.028, pixelWidth), 0.12);
			blend(coverage(Math.hypot(x - tip.x, y - tip.y), 0.038, pixelWidth), 0.08);
			// Distinct bearings keep the elbow observable when the links overlap.
			const elbowDistance = Math.hypot(x - elbow.x, y - elbow.y);
			blend(coverage(elbowDistance, 0.054, pixelWidth), 0.08);
			blend(coverage(elbowDistance, 0.017, pixelWidth), 0.85);
			const baseDistance = Math.hypot(x, y);
			blend(coverage(baseDistance, 0.066, pixelWidth), 0.12);
			blend(coverage(baseDistance, 0.027, pixelWidth), 0.85);
			output[offset + py * size + px] = value;
		}
	}
}

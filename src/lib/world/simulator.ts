/** Two uniform rods moving on a horizontal plane. q2 is relative to the first rod. */
export interface ArmState {
	q1: number;
	q2: number;
	v1: number;
	v2: number;
}

/** Normalized motor commands; each component is clipped to [-1, 1]. */
export type Action = readonly [number, number];

export interface ArmStepOptions {
	dt?: number;
	substeps?: number;
	damping?: number;
}

export interface Point {
	x: number;
	y: number;
}

export interface ArmPoints {
	base: Point;
	elbow: Point;
	tip: Point;
}

export const LINK_LENGTHS = [0.46, 0.39] as const;
export const ARM_DT = 0.12;
export const DEFAULT_DAMPING = 0.055;
export const DEFAULT_SUBSTEPS = 4;
export const TORQUE_LIMIT = 0.08;

const [L1, L2] = LINK_LENGTHS;
const M1 = 1;
const M2 = 0.8;
const C1 = L1 / 2;
const C2 = L2 / 2;
const I1 = (M1 * L1 * L1) / 12;
const I2 = (M2 * L2 * L2) / 12;
const M22 = I2 + M2 * C2 * C2;
const COUPLING = M2 * L1 * C2;
const TWO_PI = 2 * Math.PI;

export function initialArm(): ArmState {
	return { q1: -1.35, q2: 1.65, v1: 0, v2: 0 };
}

export function wrapAngle(angle: number): number {
	return ((((angle + Math.PI) % TWO_PI) + TWO_PI) % TWO_PI) - Math.PI;
}

export function armPoints(state: Pick<ArmState, 'q1' | 'q2'>): ArmPoints {
	const elbow = { x: L1 * Math.cos(state.q1), y: L1 * Math.sin(state.q1) };
	return {
		base: { x: 0, y: 0 },
		elbow,
		tip: {
			x: elbow.x + L2 * Math.cos(state.q1 + state.q2),
			y: elbow.y + L2 * Math.sin(state.q1 + state.q2)
		}
	};
}

/** Root mean square of the two circular joint errors, in radians. */
export function poseError(
	state: Pick<ArmState, 'q1' | 'q2'>,
	goal: Pick<ArmState, 'q1' | 'q2'>
): number {
	return Math.hypot(wrapAngle(state.q1 - goal.q1), wrapAngle(state.q2 - goal.q2)) / Math.SQRT2;
}

/** Kinetic energy; there is no gravitational potential on the horizontal plane. */
export function energy(state: ArmState): number {
	const c = COUPLING * Math.cos(state.q2);
	const m11 = I1 + M1 * C1 * C1 + M2 * L1 * L1 + M22 + 2 * c;
	const m12 = M22 + c;
	return (
		0.5 * m11 * state.v1 * state.v1 + m12 * state.v1 * state.v2 + 0.5 * M22 * state.v2 * state.v2
	);
}

function derivative(state: ArmState, torque1: number, torque2: number, damping: number): ArmState {
	const c = COUPLING * Math.cos(state.q2);
	const h = COUPLING * Math.sin(state.q2);
	const m11 = I1 + M1 * C1 * C1 + M2 * L1 * L1 + M22 + 2 * c;
	const m12 = M22 + c;
	// M(q) q̈ + C(q,q̇) + b q̇ = τ, including both Coriolis terms.
	const f1 = torque1 - damping * state.v1 + h * (2 * state.v1 * state.v2 + state.v2 * state.v2);
	const f2 = torque2 - damping * state.v2 - h * state.v1 * state.v1;
	const det = m11 * M22 - m12 * m12;
	return {
		q1: state.v1,
		q2: state.v2,
		v1: (M22 * f1 - m12 * f2) / det,
		v2: (m11 * f2 - m12 * f1) / det
	};
}

function advance(state: ArmState, rate: ArmState, dt: number): ArmState {
	return {
		q1: state.q1 + dt * rate.q1,
		q2: state.q2 + dt * rate.q2,
		v1: state.v1 + dt * rate.v1,
		v2: state.v2 + dt * rate.v2
	};
}

/** Integrates one observation interval; the caller retains its unmodified input. */
export function stepArm(state: ArmState, action: Action, options: ArmStepOptions = {}): ArmState {
	const dt = options.dt ?? ARM_DT;
	const damping = options.damping ?? DEFAULT_DAMPING;
	const substeps = options.substeps ?? DEFAULT_SUBSTEPS;
	if (!Number.isFinite(dt) || dt <= 0 || dt > 1) throw new RangeError('Arm dt must be in (0, 1].');
	if (!Number.isFinite(damping) || damping < 0)
		throw new RangeError('Arm damping must be finite and nonnegative.');
	if (!Number.isInteger(substeps) || substeps < 1 || substeps > 256)
		throw new RangeError('Arm substeps must be an integer in [1, 256].');
	if (![state.q1, state.q2, state.v1, state.v2, ...action].every(Number.isFinite))
		throw new RangeError('Arm states and actions must be finite.');
	const torque1 = Math.max(-1, Math.min(1, action[0])) * TORQUE_LIMIT;
	const torque2 = Math.max(-1, Math.min(1, action[1])) * TORQUE_LIMIT;
	const h = dt / substeps;
	let next = { ...state };
	for (let i = 0; i < substeps; i++) {
		const k1 = derivative(next, torque1, torque2, damping);
		const k2 = derivative(advance(next, k1, h / 2), torque1, torque2, damping);
		const k3 = derivative(advance(next, k2, h / 2), torque1, torque2, damping);
		const k4 = derivative(advance(next, k3, h), torque1, torque2, damping);
		next = {
			q1: next.q1 + (h / 6) * (k1.q1 + 2 * k2.q1 + 2 * k3.q1 + k4.q1),
			q2: next.q2 + (h / 6) * (k1.q2 + 2 * k2.q2 + 2 * k3.q2 + k4.q2),
			v1: next.v1 + (h / 6) * (k1.v1 + 2 * k2.v1 + 2 * k3.v1 + k4.v1),
			v2: next.v2 + (h / 6) * (k1.v2 + 2 * k2.v2 + 2 * k3.v2 + k4.v2)
		};
	}
	return { ...next, q1: wrapAngle(next.q1), q2: wrapAngle(next.q2) };
}

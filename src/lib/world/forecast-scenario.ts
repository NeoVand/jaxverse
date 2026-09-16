import { initialArm, stepArm, type Action } from './simulator';

export type ForecastProgram = 'push' | 'reverse' | 'release';

// Match the Float32 motor commands supplied to the network and stored in the corpus.
const push: Action = [Math.fround(0.55), Math.fround(-0.45)];

/** A real shared history, ending with momentum before any of the three choices. */
export function createForecastStart(dt: number) {
	let previous = initialArm();
	let current = previous;
	for (let i = 0; i < 4; i++) {
		previous = current;
		current = stepArm(previous, push, { dt });
	}
	return { previous, current, previousAction: push };
}

/** Hold the selected command throughout the forecast; no hidden direction changes. */
export function forecastCommands(program: ForecastProgram, horizon: number): Float32Array {
	const direction = program === 'release' ? 0 : program === 'reverse' ? -1 : 1;
	const actions = new Float32Array(2 * horizon);
	if (direction !== 0)
		for (let i = 0; i < horizon; i++) {
			actions[2 * i] = direction * push[0];
			actions[2 * i + 1] = direction * push[1];
		}
	return actions;
}

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { WorldMetrics } from './engine';
import { WorldLab } from './lab.svelte';

const fake = vi.hoisted(() => ({
	step: 0,
	stopped: false,
	evaluations: [] as number[],
	onStep: null as ((step: number) => void) | null
}));

vi.mock('./worker-engine', () => ({
	WorldEngine: class {
		async init() {
			return { step: 0, backend: 'cpu' };
		}
		async train(steps: number, emit: (metrics: WorldMetrics) => void) {
			fake.stopped = false;
			for (let i = 0; i < steps && !fake.stopped; i++) {
				const step = ++fake.step;
				emit({ step, loss: 1, predictionLoss: 1, regularizer: 0, stepMs: 1, trainingMs: step });
				fake.onStep?.(step);
			}
		}
		async evaluate() {
			fake.evaluations.push(fake.step);
			return { step: fake.step };
		}
		async stop() {
			fake.stopped = true;
		}
		async dispose() {}
	}
}));

beforeEach(() => {
	fake.step = 0;
	fake.stopped = false;
	fake.evaluations = [];
	fake.onStep = null;
});

describe('world-model training diagnostics', () => {
	it('refreshes held-out measurements after an off-boundary pause and at the resumed session end', async () => {
		const lab = new WorldLab();
		try {
			fake.onStep = (step) => {
				if (step === 43) void lab.stop();
			};
			await lab.toggleTrain();
			expect(lab.initialEvaluation?.step).toBe(0);
			expect(lab.step).toBe(43);
			expect(lab.evaluation?.step).toBe(43);
			expect(lab.phase).toBe('ready');

			fake.onStep = null;
			await lab.toggleTrain();
			expect(lab.step).toBe(5043);
			expect(fake.evaluations.slice(0, 3)).toEqual([0, 43, 143]);
			expect(fake.evaluations.at(-1)).toBe(5043);
			expect(fake.evaluations.slice(1).every((step, i) => step - fake.evaluations[i] <= 100)).toBe(
				true
			);
			expect(lab.evaluation?.step).toBe(lab.step);
			expect(lab.initialEvaluation?.step).toBe(0);
			expect(lab.training).toBe(false);
			expect(lab.phase).toBe('ready');
		} finally {
			await lab.dispose();
		}
	});
});

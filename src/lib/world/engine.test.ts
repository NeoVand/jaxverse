import { describe, expect, it } from 'vitest';
import { defaultDevice, numpy as np, type Array as JaxArray } from '@jax-js/jax';
import { WorldCore } from './engine';
import { renderSensor } from './sensor';
import { generateCorpus, gatherTransitionBatch, seededRandom } from './corpus';

defaultDevice('cpu');

describe('local world experiment lifecycle', () => {
	it('keeps future matching invariant to latent units while treating exact collapse as a tie', async () => {
		const core = new WorldCore({
			seed: 19,
			resolution: 8,
			trainEpisodes: 2,
			stepsPerEpisode: 5,
			hidden: 4,
			predictorHidden: 4,
			latent: 2,
			batch: 8
		});
		try {
			await core.init();
			const before = await core.evaluate();
			expect(before.futureMatching.ties).toBe(0);
			// Change only the latent coordinate units in the actual model. Inverting
			// the predictor's latent-input weights preserves its hidden computation.
			const params = (
				core as unknown as {
					params: {
						encoder: { w: JaxArray; b: JaxArray }[];
						predictor: { w: JaxArray; b: JaxArray }[];
					};
				}
			).params;
			const scale = 1e-6;
			const encoderOutput = params.encoder.at(-1)!;
			const predictorOutput = params.predictor.at(-1)!;
			encoderOutput.w = encoderOutput.w.mul(scale);
			encoderOutput.b = encoderOutput.b.mul(scale);
			predictorOutput.w = predictorOutput.w.mul(scale);
			predictorOutput.b = predictorOutput.b.mul(scale);
			const inputScale = new Float32Array(2 * core.config.latent + 4).fill(1);
			inputScale.fill(1 / scale, 0, 2 * core.config.latent);
			params.predictor[0].w = params.predictor[0].w.mul(
				np.array(inputScale).reshape([inputScale.length, 1])
			);
			const rescaled = await core.evaluate();
			expect(rescaled.spread / before.spread).toBeCloseTo(scale, 10);
			expect(rescaled.futureMatching.ties).toBe(before.futureMatching.ties);
			expect(rescaled.futureMatching.correct).toBe(before.futureMatching.correct);
			expect(rescaled.futureMatching.persistenceCorrect).toBe(
				before.futureMatching.persistenceCorrect
			);
			for (let i = 0; i < before.futureMatching.examples.length; i++) {
				const a = before.futureMatching.examples[i],
					b = rescaled.futureMatching.examples[i];
				expect(b.selectedIndex).toBe(a.selectedIndex);
				expect(b.persistenceIndex).toBe(a.persistenceIndex);
				for (let j = 0; j < a.distances.length; j++)
					expect(b.distances[j] / (a.distances[j] * scale ** 2)).toBeCloseTo(1, 4);
			}
			// A genuinely constant encoder/predictor has no unique nearest picture.
			encoderOutput.w = encoderOutput.w.mul(0);
			encoderOutput.b = encoderOutput.b.mul(0);
			predictorOutput.w = predictorOutput.w.mul(0);
			predictorOutput.b = predictorOutput.b.mul(0);
			const collapsed = (await core.evaluate()).futureMatching;
			expect(collapsed.ties).toBe(collapsed.total);
			expect(collapsed.persistenceTies).toBe(collapsed.total);
			expect(collapsed.correct).toBe(0);
			expect(collapsed.examples.every((example) => example.selectedIndex === null)).toBe(true);
		} finally {
			core.dispose();
		}
	});

	it('matches fixed held-out photographs without a readout or changing training randomness', async () => {
		const options = {
			seed: 19,
			resolution: 8,
			trainEpisodes: 2,
			stepsPerEpisode: 5,
			hidden: 4,
			predictorHidden: 4,
			latent: 2,
			batch: 8
		};
		const inspected = new WorldCore(options);
		const untouched = new WorldCore(options);
		try {
			await inspected.init();
			await untouched.init();
			const before = await inspected.evaluate();
			expect(before.readout).toBeNull();
			expect(before.futureMatching.total).toBe(8);
			expect(before.futureMatching.candidatesPerExample).toBe(6);
			expect(before.futureMatching.examples.map((example) => example.id)).toEqual([
				0, 1, 2, 4, 5, 6
			]);
			const corpus = generateCorpus({
				...options,
				seed: options.seed + 303,
				validationEpisodes: 12,
				size: 8,
				dt: 0.24
			});
			const heldOut = gatherTransitionBatch(corpus.validation, 8, seededRandom(989), 64);
			for (const example of before.futureMatching.examples) {
				const { id } = example;
				expect(example.previous).toEqual(heldOut.previous.slice(id * 64, (id + 1) * 64));
				expect(example.current).toEqual(heldOut.current.slice(id * 64, (id + 1) * 64));
				expect(example.candidates[example.correctIndex]).toEqual(
					heldOut.next.slice(id * 64, (id + 1) * 64)
				);
				expect([...example.previousAction, ...example.action]).toEqual(
					Array.from(heldOut.actions.slice(id * 4, (id + 1) * 4))
				);
				for (let i = 0; i < example.candidates.length; i++) {
					expect(example.candidates[i]).toHaveLength(64);
					expect(
						Array.from({ length: 8 }, (_, j) => heldOut.next.slice(j * 64, (j + 1) * 64))
					).toContainEqual(example.candidates[i]);
					for (let j = i + 1; j < example.candidates.length; j++)
						expect(example.candidates[i]).not.toEqual(example.candidates[j]);
				}
				expect(example.distances.every(Number.isFinite)).toBe(true);
				expect(example.persistenceDistances.every(Number.isFinite)).toBe(true);
				if (example.selectedIndex !== null)
					expect(example.distances[example.selectedIndex]).toBe(Math.min(...example.distances));
				if (example.persistenceIndex !== null)
					expect(example.persistenceDistances[example.persistenceIndex]).toBe(
						Math.min(...example.persistenceDistances)
					);
			}
			expect(await inspected.evaluate()).toEqual(before);
			await inspected.train(3);
			await untouched.train(3);
			const after = await inspected.evaluate();
			expect(after).toEqual(await untouched.evaluate());
			expect(after.step).toBe(3);
			for (let i = 0; i < before.futureMatching.examples.length; i++) {
				const a = before.futureMatching.examples[i],
					b = after.futureMatching.examples[i];
				expect(b.candidates).toEqual(a.candidates);
				expect(b.correctIndex).toBe(a.correctIndex);
				expect(b.previous).toEqual(a.previous);
				expect(b.current).toEqual(a.current);
			}
		} finally {
			inspected.dispose();
			untouched.dispose();
		}
	});

	it('keeps 64-pixel observations aligned with model inputs through training, reset, and comparison', async () => {
		const main = new WorldCore({
			resolution: 64,
			seed: 19,
			trainEpisodes: 1,
			stepsPerEpisode: 3,
			hidden: 4,
			predictorHidden: 4,
			latent: 2,
			batch: 4
		});
		let baseline: WorldCore | null = null;
		try {
			const initial = await main.init();
			expect(initial.config.resolution).toBe(64);
			expect(initial.preview.frames).toHaveLength(initial.preview.states.length * 64 * 64);
			for (let i = 0; i < initial.preview.states.length; i++)
				expect(initial.preview.frames.slice(i * 4096, (i + 1) * 4096)).toEqual(
					renderSensor(initial.preview.states[i], 64)
				);
			const trained = await main.train(2);
			expect(trained.step).toBe(2);
			expect(Number.isFinite(trained.loss)).toBe(true);
			const evaluation = await main.evaluate();
			expect(Number.isFinite(evaluation.predictionLoss)).toBe(true);
			expect(evaluation.futureMatching.examples[0].current).toHaveLength(4096);
			expect(
				evaluation.futureMatching.examples[0].candidates.every((frame) => frame.length === 4096)
			).toBe(true);

			// Changing the seed rebuilds the corpus; it must retain the configured camera size.
			const reset = await main.reset(23);
			expect(reset.step).toBe(0);
			expect(reset.preview.frames).toHaveLength(reset.preview.states.length * 4096);
			expect(reset.preview.frames.slice(0, 4096)).toEqual(
				renderSensor(reset.preview.states[0], 64)
			);
			expect(reset.preview.frames).not.toEqual(initial.preview.frames);
			const resetEvaluation = await main.evaluate();
			baseline = await main.createUnregularizedBaseline();
			expect(baseline.config.resolution).toBe(64);
			expect((await baseline.evaluate()).projection).toEqual(resetEvaluation.projection);
			const baselineMetrics = await baseline.train(1);
			expect(baselineMetrics.step).toBe(1);
			expect(Number.isFinite(baselineMetrics.loss)).toBe(true);
		} finally {
			baseline?.dispose();
			main.dispose();
		}
	});

	it('initializes a fresh matched baseline without changing the trained model', async () => {
		const main = new WorldCore({
			seed: 19,
			trainEpisodes: 1,
			stepsPerEpisode: 3,
			hidden: 4,
			predictorHidden: 4,
			latent: 2,
			batch: 4
		});
		let baseline: WorldCore | null = null;
		try {
			await main.init();
			const initial = await main.evaluate();
			await main.train(2);
			const trained = await main.evaluate();
			baseline = await main.createUnregularizedBaseline();
			const fresh = await baseline.evaluate();
			expect(fresh.step).toBe(0);
			expect(fresh.projection).toEqual(initial.projection);
			expect(fresh.predictionLoss).toBe(initial.predictionLoss);
			expect(baseline.config.regularization).toBe(0);
			await baseline.train(1);
			expect(await main.evaluate()).toEqual(trained);
			baseline.dispose();
			baseline = null;
			expect(await main.evaluate()).toEqual(trained);
		} finally {
			baseline?.dispose();
			main.dispose();
		}
	});

	it('uses identical experience for matched regularizer runs and changes it consistently with seed', async () => {
		const options = { seed: 17, trainEpisodes: 1, stepsPerEpisode: 3, dt: 0.24 };
		const main = new WorldCore(options);
		const baseline = new WorldCore({ ...options, regularization: 0 });
		try {
			const a = await main.init(),
				b = await baseline.init();
			expect(a.dt).toBe(0.24);
			expect(a.preview.frames).toEqual(b.preview.frames);
			expect(a.preview.actions).toEqual(b.preview.actions);
			const reset = await main.reset(19);
			const fresh = new WorldCore({ ...options, seed: 19 });
			try {
				const next = await fresh.init();
				expect(reset.preview.frames).toEqual(next.preview.frames);
				expect(reset.preview.frames).not.toEqual(a.preview.frames);
			} finally {
				fresh.dispose();
			}
		} finally {
			main.dispose();
			baseline.dispose();
		}
	});
});

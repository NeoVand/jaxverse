import { describe, expect, it } from 'vitest';
import { defaultDevice } from '@jax-js/jax';
import { WorldCore } from './engine';

defaultDevice('cpu');

describe('local world experiment lifecycle', () => {
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

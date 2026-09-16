import { describe, expect, it } from 'vitest';
import { gatherTransitionBatch, generateCorpus, seededRandom } from './corpus';
import { renderSensor } from './sensor';
import { stepArm } from './simulator';

const options = { trainEpisodes: 3, validationEpisodes: 2, stepsPerEpisode: 8, size: 16, seed: 81 };

describe('world-model experience corpus', () => {
	it('splits complete, reproducible episodes by independent random streams', () => {
		const corpus = generateCorpus(options);
		const repeat = generateCorpus(options);
		expect(corpus).toEqual(repeat);
		const trainingSeeds = new Set(corpus.train.map((episode) => episode.seed));
		expect(corpus.validation.every((episode) => !trainingSeeds.has(episode.seed))).toBe(true);
		expect(corpus.validation[0].frames).not.toEqual(corpus.train[0].frames);
	});

	it('records precisely the actions between each pair of rendered frames', () => {
		const corpus = generateCorpus(options);
		for (const episode of [...corpus.train, ...corpus.validation]) {
			expect(episode.states).toHaveLength(episode.steps + 1);
			for (let t = 0; t < episode.steps; t++) {
				const expected = stepArm(
					episode.states[t],
					[episode.actions[t * 2], episode.actions[t * 2 + 1]],
					{
						dt: corpus.dt,
						damping: corpus.damping
					}
				);
				expect(episode.states[t + 1]).toEqual(expected);
				expect(
					episode.frames.subarray((t + 1) * corpus.frameSize, (t + 2) * corpus.frameSize)
				).toEqual(renderSensor(expected, corpus.size));
			}
		}
	});

	it('gathers both past and candidate actions in the correct two-frame history order', () => {
		const corpus = generateCorpus(options);
		const random = () => 0.4;
		const batch = gatherTransitionBatch(corpus.train, 2, random, corpus.frameSize);
		const episode = corpus.train[1];
		const t = 1 + Math.floor(0.4 * (episode.steps - 1));
		expect(batch.previous.subarray(0, corpus.frameSize)).toEqual(
			episode.frames.subarray((t - 1) * corpus.frameSize, t * corpus.frameSize)
		);
		expect(batch.current.subarray(0, corpus.frameSize)).toEqual(
			episode.frames.subarray(t * corpus.frameSize, (t + 1) * corpus.frameSize)
		);
		expect(batch.next.subarray(0, corpus.frameSize)).toEqual(
			episode.frames.subarray((t + 1) * corpus.frameSize, (t + 2) * corpus.frameSize)
		);
		expect(batch.actions.subarray(0, 4)).toEqual(
			episode.actions.subarray((t - 1) * 2, (t + 1) * 2)
		);
		expect(Object.keys(batch).sort()).toEqual([
			'actions',
			'batchSize',
			'current',
			'next',
			'previous'
		]);
	});

	it('uses bounded, held, released, and reversed commands rather than only white noise', () => {
		const corpus = generateCorpus({ ...options, stepsPerEpisode: 64 });
		const episodes = corpus.train;
		expect(
			episodes.every((episode) => episode.actions.every((value) => value >= -1 && value <= 1))
		).toBe(true);
		let held = 0;
		let release = 0;
		let reversed = 0;
		for (const { actions, steps } of episodes) {
			for (let t = 1; t < steps; t++) {
				if (
					actions[t * 2] === actions[(t - 1) * 2] &&
					actions[t * 2 + 1] === actions[(t - 1) * 2 + 1]
				)
					held++;
				if (actions[t * 2] === 0 && actions[t * 2 + 1] === 0) release++;
				if (actions[t * 2] !== 0 && actions[t * 2] === -actions[(t - 1) * 2]) reversed++;
			}
		}
		expect(held).toBeGreaterThan(80);
		expect(release).toBeGreaterThan(5);
		expect(reversed).toBeGreaterThan(0);
	});

	it('keeps sampling deterministic without modifying the corpus', () => {
		const corpus = generateCorpus(options);
		const a = gatherTransitionBatch(corpus.train, 5, seededRandom(9), corpus.frameSize);
		const b = gatherTransitionBatch(corpus.train, 5, seededRandom(9), corpus.frameSize);
		expect(a).toEqual(b);
		expect(a.previous.buffer).not.toBe(corpus.train[0].frames.buffer);
	});
});

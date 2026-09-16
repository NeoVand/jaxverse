import { describe, expect, it } from 'vitest';
import { defaultDevice, numpy as np, tree, valueAndGrad } from '@jax-js/jax';
import {
	initWorldParams,
	objective,
	parameterCount,
	sigreg,
	WORLD_CONFIG,
	type WorldConfig
} from './model';

defaultDevice('cpu');

function referenceSigreg(
	z: number[],
	times: number,
	batch: number,
	dim: number,
	directions: number[],
	projections: number
) {
	let result = 0;
	for (let t = 0; t < times; t++)
		for (let m = 0; m < projections; m++) {
			let statistic = 0;
			for (let k = 0; k < 17; k++) {
				const x = (3 * k) / 16,
					phi = Math.exp((-x * x) / 2);
				let real = 0,
					imag = 0;
				for (let b = 0; b < batch; b++) {
					let projection = 0;
					for (let d = 0; d < dim; d++)
						projection += z[(t * batch + b) * dim + d] * directions[d * projections + m];
					real += Math.cos(projection * x) / batch;
					imag += Math.sin(projection * x) / batch;
				}
				statistic +=
					((real - phi) ** 2 + imag ** 2) * (k === 0 || k === 16 ? 1 : 2) * (3 / 16) * phi;
			}
			result += (statistic * batch) / (times * projections);
		}
	return result;
}

describe('world objective', () => {
	it('matches independent characteristic-function quadrature at each time position', () => {
		const z = [-1, 0.3, 0.2, -0.5, 1.2, 0.7, 0.4, 1.4, -0.8, 0.1, 1.1, -1.4];
		const a = [1, 0, 0, 1];
		const value = sigreg(np.array(z).reshape([2, 3, 2]), np.array(a).reshape([2, 2]));
		expect(value.item()).toBeCloseTo(referenceSigreg(z, 2, 3, 2, a, 2), 5);
	});

	it('differentiates the shared target encoder as well as its context uses', () => {
		const config: WorldConfig = {
			resolution: 2,
			hidden: 4,
			latent: 2,
			predictorHidden: 4,
			projections: 2,
			regularization: 0.04
		};
		const pixels = Float32Array.from(
			{ length: 3 * 3 * 4 },
			(_, i) => 0.3 + (0.6 * ((i * 7) % 17)) / 17
		);
		const actions = Float32Array.from({ length: 12 }, (_, i) => Math.sin(i));
		const input = () => [
			np.array(pixels).reshape([3, 3, 4]),
			np.array(actions).reshape([3, 4]),
			np.array([1, 0, 0, 1]).reshape([2, 2])
		];
		const params = initWorldParams(config, 7);
		const [value, gradient] = valueAndGrad((p) => {
			const [x, a, directions] = input();
			const [loss, parts] = objective(p, config, x, a, directions);
			parts.dispose();
			return loss;
		})(params);
		const actual = gradient.encoder[0].w.ref.dataSync()[2];
		value.dispose();
		tree.dispose(gradient);
		const finiteDifference = (delta: number) => {
			const p = initWorldParams(config, 7);
			const w = new Float32Array(p.encoder[0].w.dataSync());
			w[2] += delta;
			p.encoder[0].w = np.array(w).reshape([4, 4]);
			const [x, a, directions] = input();
			const [loss, parts] = objective(p, config, x, a, directions);
			const v = loss.item();
			parts.dispose();
			return v;
		};
		const numerical = (finiteDifference(0.001) - finiteDifference(-0.001)) / 0.002;
		expect(Math.abs(actual)).toBeGreaterThan(1e-5);
		expect(actual).toBeCloseTo(numerical, 3);
	});

	it('reports the actual default trainable size', () => {
		const p = initWorldParams(WORLD_CONFIG, 11);
		const count = (tree.leaves(p) as { size: number }[]).reduce((n, x) => n + x.size, 0);
		expect(parameterCount(WORLD_CONFIG)).toBe(count);
		expect(count).toBe(152464);
		tree.dispose(p);
	});
});

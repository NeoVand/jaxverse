import { numpy as np, jit, valueAndGrad, tree } from '@jax-js/jax';
import { objective, type Tensor, type WorldConfig } from './model';

export interface WorldBatch {
	/** Time-major [3,B,R²]. */
	pixels: Float32Array<ArrayBuffer>;
	/** [B,4]: a[t−1] then a[t]. */
	actions: Float32Array<ArrayBuffer>;
	directions: Float32Array<ArrayBuffer>;
}

export interface WorldOptimizer {
	step(params: Tensor, batch: WorldBatch, learningRate: number): [Tensor, Tensor];
	dispose(): void;
}

/** Adam lives inside JIT, including a global gradient cap. */
export function createWorldOptimizer(
	config: WorldConfig,
	batchSize: number,
	params: Tensor
): WorldOptimizer {
	let first = tree.map((x: Tensor) => np.zerosLike(x), tree.ref(params));
	let second = tree.map((x: Tensor) => np.zerosLike(x), tree.ref(params));
	let step = 0;
	const update = jit(
		(
			p: Tensor,
			m: Tensor,
			v: Tensor,
			constants: Tensor,
			x: Tensor,
			a: Tensor,
			directions: Tensor
		) => {
			const [[loss, parts], grads] = valueAndGrad(
				(pp: Tensor) => objective(pp, config, x, a, directions),
				{ hasAux: true }
			)(tree.ref(p));
			const [leaves, def] = tree.flatten(p);
			const gs = tree.leaves(grads) as Tensor[];
			const ms = tree.leaves(m) as Tensor[];
			const vs = tree.leaves(v) as Tensor[];
			let sum = np.sum(np.square(gs[0].ref));
			for (let i = 1; i < gs.length; i++) sum = sum.add(np.sum(np.square(gs[i].ref)));
			const scale = np.minimum(np.array(5).div(np.sqrt(sum).add(1e-8)), 1);
			const correction1 = constants.ref.slice([0, 1]);
			const correction2 = constants.ref.slice([1, 2]);
			const lr = constants.slice([2, 3]);
			const nextP: Tensor[] = [],
				nextM: Tensor[] = [],
				nextV: Tensor[] = [];
			for (let i = 0; i < gs.length; i++) {
				const g = gs[i].mul(scale.ref);
				const mi = ms[i].mul(0.9).add(g.ref.mul(0.1));
				const vi = vs[i].mul(0.99).add(np.square(g).mul(0.01));
				nextP.push(
					leaves[i].sub(
						mi.ref
							.mul(correction1.ref)
							.mul(lr.ref)
							.div(np.sqrt(vi.ref.mul(correction2.ref)).add(1e-8))
					)
				);
				nextM.push(mi);
				nextV.push(vi);
			}
			scale.dispose();
			correction1.dispose();
			correction2.dispose();
			lr.dispose();
			return [
				np.concatenate([loss.reshape([1]), parts]),
				tree.unflatten(def, nextP),
				tree.unflatten(def, nextM),
				tree.unflatten(def, nextV)
			];
		}
	);
	return {
		step(p, b, learningRate) {
			step++;
			const k = np.array(
				new Float32Array([1 / (1 - 0.9 ** step), 1 / (1 - 0.99 ** step), learningRate])
			);
			const [metrics, next, nextFirst, nextSecond] = update(
				p,
				first,
				second,
				k,
				np.array(b.pixels).reshape([3, batchSize, config.resolution ** 2]),
				np.array(b.actions).reshape([batchSize, 4]),
				np.array(b.directions).reshape([config.latent, config.projections])
			);
			first = nextFirst;
			second = nextSecond;
			return [metrics, next];
		},
		dispose() {
			tree.dispose(first);
			tree.dispose(second);
		}
	};
}

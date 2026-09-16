/** CPU checks for the research-only temporal-transformer predictor. */
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import ts from 'typescript';
import { defaultDevice, numpy as np, tree, valueAndGrad, jit } from '@jax-js/jax';
defaultDevice('cpu');
const load = async (path) => {
	const source = await readFile(path, 'utf8');
	const js = ts
		.transpileModule(source, {
			compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 }
		})
		.outputText.replaceAll('@jax-js/jax', import.meta.resolve('@jax-js/jax'));
	return import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'));
};
const experimental = await load(new URL('./world-transformer-model.ts', import.meta.url));
const original = await load(new URL('../src/lib/world/model.ts', import.meta.url));
const count = (p) => tree.leaves(p).reduce((n, x) => n + x.size, 0);
const arrays = (p) => tree.leaves(p).map((x) => Array.from(x.ref.dataSync()));
const params = experimental.initWorldParams(experimental.WORLD_CONFIG, 17);
const reference = original.initWorldParams(original.WORLD_CONFIG, 17);
assert.equal(count(params), experimental.parameterCount(experimental.WORLD_CONFIG));
assert.deepEqual(arrays(params.encoder), arrays(reference.encoder));
console.log(
	JSON.stringify({
		check: 'parameter count and identical encoder',
		total: count(params),
		encoder: count(params.encoder),
		predictor: count(params.predictor)
	})
);
tree.dispose(params);
tree.dispose(reference);
const c = { ...experimental.WORLD_CONFIG, resolution: 4, hidden: 8, latent: 8, projections: 3 };
let p = experimental.initWorldParams(c, 17);
const pixels = Float32Array.from(
	{ length: 3 * 3 * 16 },
	(_, i) => 0.3 + (0.6 * ((i * 7) % 17)) / 17
);
const actions = new Float32Array([
	0.5, -0.2, -0.4, 0.7, 0.2, 0.3, -0.2, -0.1, 0.8, 0.4, -0.5, -0.7
]);
const directions = experimental.projectionDirections(c, experimental.seededRandom(31));
const loss = (pp) =>
	experimental.objective(
		pp,
		c,
		np.array(pixels).reshape([3, 3, 16]),
		np.array(actions).reshape([3, 4]),
		np.array(directions).reshape([8, 3])
	);
const initialMetrics = [];
for (let step = 0; step < 3; step++) {
	const [[total, parts], grads] = valueAndGrad(loss, { hasAux: true })(tree.ref(p));
	const g = arrays(grads);
	assert(g.flat().every(Number.isFinite));
	const metrics = {
		step,
		loss: total.item(),
		parts: Array.from(parts.dataSync()),
		nonzeroLeaves: g.filter((a) => a.some((v) => Math.abs(v) > 1e-12)).length,
		leaves: g.length
	};
	assert(Number.isFinite(metrics.loss));
	console.log(JSON.stringify(metrics));
	initialMetrics.push(metrics);
	p = tree.map((pp, gg) => pp.sub(gg.mul(0.001)), p, grads);
}
assert(initialMetrics[2].nonzeroLeaves > initialMetrics[0].nonzeroLeaves);
const prev = Float32Array.from({ length: 16 }, (_, i) => Math.sin(i * 0.2));
const curr = Float32Array.from({ length: 16 }, (_, i) => Math.cos(i * 0.2));
const prediction = (a) =>
	Array.from(
		experimental
			.predict(
				tree.ref(p.predictor),
				np.array(prev).reshape([2, 8]),
				np.array(curr).reshape([2, 8]),
				np.array(a).reshape([2, 4])
			)
			.dataSync()
	);
const one = prediction([0.1, 0.2, 0.3, 0.4, -0.2, -0.4, 0.2, 0.1]);
const two = prediction([0.1, 0.2, -0.7, 0.8, -0.2, -0.4, -0.5, 0.9]);
assert(one.every(Number.isFinite));
assert(one.some((v, i) => Math.abs(v - two[i]) > 1e-9));
console.log(
	JSON.stringify({
		check: 'outgoing action changes next prediction after gate learning',
		maxDifference: Math.max(...one.map((v, i) => Math.abs(v - two[i])))
	})
);
const compiledPredict = jit(experimental.predict);
const jitOutput = Array.from(
	compiledPredict(
		tree.ref(p.predictor),
		np.array(prev).reshape([2, 8]),
		np.array(curr).reshape([2, 8]),
		np.array([0.1, 0.2, 0.3, 0.4, -0.2, -0.4, 0.2, 0.1]).reshape([2, 4])
	).dataSync()
);
assert(jitOutput.every(Number.isFinite));
assert(jitOutput.every((v, i) => Math.abs(v - one[i]) < 1e-5));
console.log(
	JSON.stringify({
		check: 'JIT predictor matches eager CPU',
		maxDifference: Math.max(...jitOutput.map((v, i) => Math.abs(v - one[i])))
	})
);
const compiledObjective = jit(valueAndGrad(loss, { hasAux: true }));
const [[jitLoss, jitParts], jitGrads] = compiledObjective(tree.ref(p));
assert(arrays(jitGrads).flat().every(Number.isFinite));
console.log(
	JSON.stringify({
		check: 'JIT full objective and gradients finite',
		loss: jitLoss.item(),
		parts: Array.from(jitParts.dataSync())
	})
);
tree.dispose(jitGrads);
tree.dispose(p);

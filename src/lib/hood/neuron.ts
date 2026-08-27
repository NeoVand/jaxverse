import type { HoodChapter } from './types';

export const neuron: HoodChapter = {
	slug: 'neuron',
	blocks: [
		{
			id: 'bare',
			lesson: 'when three knobs need no library',
			ml: [
				{
					title: 'One honest footnote',
					body: `This plate — the single neuron with three sliders — uses no engine at all: it is <code>v·σ(wx + b)</code> redrawn per frame, because a demo about <em>feeling three knobs</em> should not spend a GPU. Knowing when not to reach for the machinery is part of the lesson, and the reward chapter will make the same choice at larger scale. The real engine arrives two plates down, where a hundred parameters need training at once.`
				}
			],
			ui: []
		},
		{
			id: 'engine',
			lesson: 'params as a tree, valueAndGrad, jit',
			lede: `This plate is where jax-js starts doing the training for real. The curve workshop runs on the book's MLP engine — a thousand lines in <code>src/lib/nn/</code> that also carry the next three chapters unchanged. Three ideas do all the work: parameters live in a plain object (a <em>pytree</em>), <code>valueAndGrad</code> differentiates the loss with respect to that whole object at once, and <code>jit</code> compiles the step so it stops being JavaScript.`,
			ml: [
				{
					title: 'Parameters are just an object',
					body: `No <code>Model</code> class, no framework. The network is a record with two lists of arrays — weights and biases — initialized Glorot-style from a seeded generator so a reset reproduces exactly. JAX calls any such nested structure a <em>pytree</em>, and every transformation (<code>grad</code>, <code>jit</code>, the optimizer) works over the whole tree as if it were one number.`,
					code: {
						file: 'src/lib/nn/worker.ts',
						code: `function initParams(c: MlpConfig, seed: number): any {
	const rand = mulberry32(seed);
	const w: any[] = [];
	const b: any[] = [];
	for (let i = 0; i < c.layers.length - 1; i++) {
		const [fin, fout] = [c.layers[i], c.layers[i + 1]];
		const limit = Math.sqrt(6 / (fin + fout));
		const buf = new Float32Array(fin * fout);
		for (let j = 0; j < buf.length; j++) buf[j] = (rand() * 2 - 1) * limit;
		w.push(np.array(buf).reshape([fin, fout]));
		b.push(np.zeros([fout]));
	}
	return { w, b };
}`
					}
				},
				{
					title: 'The loss, and its gradient over the whole tree',
					body: `The forward pass is a fold — matmul, add bias, bend, repeat — and the loss for this chapter is the mean squared error you met on the plate. The magic line is the last one: <code>valueAndGrad</code> takes a function of the parameter tree and returns the loss <em>and</em> a gradient tree with the same shape — one gradient array per weight matrix, one per bias, all from a single backward pass.`,
					code: {
						file: 'src/lib/nn/worker.ts',
						code: `// the forward fold: matmul, bias, bend, repeat — the last layer stays linear
let h = x;
for (let k = 0; k < L; k++) {
	h = np.dot(h, p.w[k]).add(p.b[k]);
	if (k < L - 1) h = ACT[c.activation](h); // tanh / relu / gelu / silu
}
const loss = np.mean(np.square(h.sub(y))); // this chapter: plain MSE

// loss and ∂loss/∂(every array in p), from one backward pass
const [lossVal, grads] = valueAndGrad((pp: any) => lossFn(pp, c, x, y))(params);`
					}
				},
				{
					title: 'Adam from optax, and a step that compiles',
					body: `The Adam you hand-traced in the prologue returns as three lines of <code>@jax-js/optax</code>, applied tree-wise. And the whole step is wrapped in <code>jit</code>: the first call traces the JavaScript into a compute graph and compiles it — one WebGPU kernel launch per step afterwards, instead of hundreds of interpreted array calls. That first sluggish step you feel on every plate is the compiler working; every step after is why the loss curve moves in real time.`,
					code: {
						file: 'src/lib/nn/worker.ts',
						code: `solver = adam(cfg.lr ?? 3e-3, { b1: 0.9, b2: 0.99 });
optState = solver.init(tree.ref(params));
jitStep = jit((p: any, x: any, y: any) =>
	valueAndGrad((pp: any) => lossFn(pp, c, x, y))(p)
);

// one training step, every frame the plate is running
const [lossVal, grads] = jitStep(tree.ref(params), x.ref, y.ref);
const [updates, newOptState] = solver.update(grads, optState, tree.ref(params));
params = applyUpdates(params, updates);
optState = newOptState;`
					}
				}
			],
			ui: [
				{
					title: 'Training off the main thread',
					body: `The engine lives in a Web Worker. The page talks to it through a tiny RPC — <code>{ id, op }</code> messages in, <code>{ id, ok, result }</code> or streaming <code>{ event: 'metrics' }</code> frames back — so a thousand training steps never block a scroll or a slider. The main-thread half is <code>src/lib/nn/mlp-engine.ts</code>; every training chapter reuses it.`
				},
				{
					title: 'The palette and the wiring diagram',
					body: `The curve palette asks the worker for hidden activations over a dense sweep of x, then draws each unit's output as its own curve — colored by the sign of its outgoing weight, thickened by its magnitude. The architecture diagram (<code>ArchDiagram.svelte</code>) renders the same live weights as edges, and the two views share hover state through plain Svelte props, not a store: the coupling is one component deep.`
				}
			],
			lab: {
				file: 'neuron.zip',
				note: 'A 140-line MLP fitting a sine wave in your browser: pytree params, valueAndGrad, optax Adam, live canvas plot'
			}
		}
	]
};

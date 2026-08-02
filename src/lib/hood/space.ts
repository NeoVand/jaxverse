import type { HoodChapter } from './types';

export const space: HoodChapter = {
	slug: 'space',
	lesson: 'Lesson 2 — devices, and reading the middle of a network',
	lede: `Same engine as the last chapter, two new ideas. First: where the arithmetic actually runs. jax-js initializes against whatever the machine offers — WebGPU if the browser grants it, WASM if not, plain CPU as the floor — and the plate's telemetry strip tells you which one you got. Second: a trained network is not a sealed pipe. The middle view of both plates is nothing more than asking the engine for its hidden activations and plotting them.`,
	ml: [
		{
			title: 'Pick a device once, forget it forever',
			body: `<code>init()</code> probes the machine and returns the devices it could bring up; <code>defaultDevice</code> pins one, and from then on every array op runs there without another word. The same training code executes on a gaming GPU and on a locked-down tablet — slower, but identically. That portability is most of the reason this book can exist as a website.`,
			code: {
				file: 'src/lib/nn/worker.ts',
				code: `const devices = await init();
device = devices.includes('webgpu')
	? 'webgpu'
	: devices.includes('wasm')
		? 'wasm'
		: 'cpu';
defaultDevice(device as 'webgpu' | 'wasm' | 'cpu');`
			}
		},
		{
			title: 'Same network, new head: cross-entropy',
			body: `The only change from the curve workshop is the loss. Two outputs instead of one, <code>logSoftmax</code> to turn scores into log-probabilities, and the negative log-likelihood of the true class — the cross-entropy the digits chapter will dissect in full. Swapping a loss is a one-line edit precisely because the loss is just a function the gradient flows through.`,
			code: {
				file: 'src/lib/nn/worker.ts',
				code: `if (c.loss === 'mse') {
	loss = np.mean(np.square(out.sub(y)));
} else {
	// y is one-hot: pick out the log-probability of the true class
	const logp = nn.logSoftmax(out, -1);
	loss = np.mean(np.sum(logp.mul(y), -1).neg());
}`
			}
		},
		{
			title: 'The bent grid is just a forward pass',
			body: `Everything you watched deform — the grid, the points, their ghosts in hidden space — comes from one worker op: run the network and hand back <em>every layer's</em> activations, not just the verdict. The middle view feeds the data and a lattice of synthetic grid points through the same trained weights in one batch and draws the last hidden layer; the "unfold" replay is those two point sets interpolated on the CPU. No second model, no tricks: the visualization <em>is</em> the network, evaluated where you can see it.`,
			code: {
				file: 'src/lib/components/demos/space/SpaceLab.svelte',
				code: `// data rows and grid lattice ride one batch through the trained weights
const [probLogits, acts, w] = await Promise.all([
	engine.predict(probeGrid, PROBE_RES * PROBE_RES, 1024), // the verdict wash
	engine.activations(combined, n + g, 1024), // every layer, data + lattice
	engine.weights() // the live wiring diagram
]);
const hAll = acts.layers[acts.layers.length - 2]; // the last hidden layer`
			}
		}
	],
	ui: [
		{
			title: 'Three canvases, one clock',
			body: `Input space, hidden space, and the wiring diagram redraw from a single <code>requestAnimationFrame</code> loop that polls the worker's latest weights snapshot. The hidden view keeps its own camera — drag to rotate when the layer is 3-D, PCA shadow beyond that — and eases its frame between training steps so the cloud never teleports.`
		},
		{
			title: 'The fold button',
			body: `Unfold interpolates every grid vertex between its input position and its hidden position with a cubic ease, drawing the in-between lattices — a linear homotopy, which is honest enough for intuition even though the network itself bends space nonlinearly along the way. The dataset thumbnails are the real datasets rendered tiny into offscreen canvases at boot, not images.`
		}
	],
	lab: {
		file: 'space.zip',
		note: 'A two-moons classifier that prints its own hidden space as ASCII art while it trains — devices, xent loss, activations'
	}
};

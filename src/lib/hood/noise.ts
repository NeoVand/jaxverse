import type { HoodChapter } from './types';

export const noise: HoodChapter = {
	slug: 'noise',
	blocks: [
		{
			id: 'denoise',
			lesson: 'one training example, from scratch',
			lede: `There is no dataset of noisy pictures anywhere in this chapter. Every training example is manufactured on the spot out of a clean picture, a fresh Gaussian and a number drawn uniformly from zero to one — which is why the corpus is eight thousand images and the model never sees the same example twice.`,
			ml: [
				{
					title: 'Corrupt, and record what would undo it',
					body: `The whole objective is in the inner loop. Pick a picture, pick how far to ruin it, mix, and store the noise you mixed in as the target. The <code>flow</code> branch is the next chapter's version of the same three lines and is worth reading beside this one: identical machinery, different target. The tenth of rows that drop the prompt or the style are what make guidance possible later — one set of weights has to know both what a cat looks like and what a picture looks like.`,
					code: {
						file: 'src/lib/diffusion/runtime.ts',
						code: `const tau = rand();
const o = b * dim;
if (c.objective === 'flow') {
	for (let i = 0; i < dim; i++) {
		const x0 = toUnit(corpus.images[src + i]);
		out.x[o + i] = (1 - tau) * x0 + tau * noise[i];
		out.target[o + i] = noise[i] - x0;
	}
} else {
	const ab = alphaBar(tau);
	const sa = Math.sqrt(ab);
	const sn = Math.sqrt(1 - ab);
	for (let i = 0; i < dim; i++) {
		out.x[o + i] = sa * toUnit(corpus.images[src + i]) + sn * noise[i];
		out.target[o + i] = noise[i];
	}
}

writeCondition(out.cond, b, c, tau, {
	tags: rand() < 0.1 ? [] : corpus.tags[idx],
	style: rand() < 0.1 ? null : style
});`
					}
				},
				{
					title: 'The schedule, as a function rather than a table',
					body: `Published diffusion code usually precomputes a thousand-entry array of α̅ values because it thinks in discrete timesteps. Nothing here needs to: τ is continuous, the network is told its value directly, and the schedule is just a function you can call at any real number between 0 and 1. That is also what lets the sampler choose its own step count after training — five rungs or a hundred, from the same weights.`,
					code: {
						file: 'src/lib/diffusion/model.ts',
						code: `export function alphaBar(tau: number): number {
	const s = 0.008;
	const f = (u: number) => Math.cos(((u + s) / (1 + s)) * (Math.PI / 2)) ** 2;
	return Math.min(Math.max(f(tau) / f(0), 1e-6), 1);
}`
					}
				},
				{
					title: 'Adam, moved inside jit',
					body: `This one is a performance finding rather than a lesson in diffusion, but it is the difference between a plate you watch and a plate you wait for. <code>optax</code>'s Adam reads its own step counter with <code>.item()</code>, which a tracer cannot answer, so its update has to run outside the jit boundary — and out there it dispatches several tiny kernels for each of forty-odd parameter tensors. Measured on this model: 151 ms a step with the update outside, 88 ms with it inside. The two bias corrections and the learning rate go in as one traced triple so the shape signature never changes and nothing recompiles. The same harness measured the U-Net the chapter mentions; run it yourself with <code>node scripts/bench-denoiser.mjs</code>.`,
					code: {
						file: 'src/lib/diffusion/runtime.ts',
						code: `const fused = jit((p: Arr, mm: Arr, vv: Arr, k: Arr, x: Arr, cond: Arr, target: Arr) => {
	const [loss, grads] = valueAndGrad((pp: Arr) => lossFn(pp, c, batch, x, cond, target))(
		tree.ref(p)
	);
	const [leaves, def] = tree.flatten(p) as [Arr[], Arr];
	const gl = tree.leaves(grads) as Arr[];
	const c1 = k.ref.slice([0, 1]);
	const c2 = k.ref.slice([1, 2]);
	const lr = k.slice([2, 3]);
	for (let i = 0; i < gl.length; i++) {
		const mi = ml[i].mul(0.9).add(gl[i].ref.mul(0.1));
		const vi = vl[i].mul(0.99).add(np.square(gl[i]).mul(0.01));
		const mhat = mi.ref.mul(c1.ref);
		const vhat = vi.ref.mul(c2.ref);
		nextP.push(leaves[i].sub(mhat.mul(lr.ref).div(np.sqrt(vhat).add(1e-8))));
	}
	return [loss, tree.unflatten(def, nextP), ...];
});`
					}
				}
			],
			ui: [
				{
					title: 'Two batch buffers, ping-ponged',
					body: `Corrupting sixty-four pictures means a few hundred thousand calls to <code>Math.log</code>, <code>sin</code> and <code>cos</code> — Box–Muller does not come free. Unoverlapped that was a third of the wall clock, because the CPU was building batch <em>n</em> + 1 while the GPU sat idle. The fix is to submit the step first (which copies the batch to the device), then fill the spare buffer, and only then read the loss back. The read is the synchronization point, and the batch is built underneath it.`,
					code: {
						file: 'src/lib/diffusion/worker.ts',
						code: `const [lossArr, next] = opt.step(params, batchBuf!, lr);
params = next;
makeBatch(corpus, cfg, batchSize, rand, spareBuf!);
[batchBuf, spareBuf] = [spareBuf!, batchBuf!];
const loss = lossArr.item();`
					}
				},
				{
					title: 'Why the pictures are premultiplied',
					body: `Emoji have transparent backgrounds, and an RGBA image stores colour in the transparent regions that no one ever looks at — whatever the artist happened to leave there. To a model trained on squared error that garbage is a real target with real gradients. Multiplying colour by alpha at build time collapses all of it to a single value the network can actually hit. (The PNG encoder then writes its own constant into the fully transparent pixels, and the canvas hands back <code>(0, 0, 0, 0)</code> for them on the way in, so the loader sees exact zeros either way — but the premultiply is what makes the <em>partly</em> transparent edges consistent, and those are most of an emoji's outline.) It also makes drawing the result one line: premultiplied compositing is <code>src + background · (1 − α)</code>, no division and no halo.`
				}
			]
		},
		{
			id: 'walk',
			lesson: 'sampling as a loop you write yourself',
			lede: `Training never simulated a chain, so the chain has to be written by hand at sampling time — and because it lives outside the network, the step count, the step rule and the amount of noise put back are all decisions you can change without retraining anything.`,
			ml: [
				{
					title: 'The reverse step, both flavours',
					body: `<code>eta</code> interpolates between the two published samplers. At 0 the update is deterministic: work out the clean picture the model's answer implies, then re-noise it to the next rung down. At 1 a fresh Gaussian is stirred in and the walk becomes the random one the original paper described. The clamp is not decoration — with few steps the implied clean image routinely lands outside the valid range, and letting it stay there is exactly how a short schedule produces saturated garbage.`,
					code: {
						file: 'src/lib/diffusion/runtime.ts',
						code: `const ab = alphaBar(tau);
const abNext = tauNext <= 0 ? 1 : alphaBar(tauNext);
const sa = Math.sqrt(ab);
const sn = Math.sqrt(1 - ab);
const eta = tauNext <= 0 ? 0 : (req.eta ?? 0);
const sigma = eta * Math.sqrt(((1 - abNext) / (1 - ab)) * Math.max(1 - ab / abNext, 0));
const keep = Math.sqrt(Math.max(1 - abNext - sigma * sigma, 0));
if (sigma > 0) gaussianFill(noise, rand);
for (let i = 0; i < state.length; i++) {
	const x0 = Math.min(Math.max((state[i] - sn * field[i]) / sa, -1), 1);
	state[i] = Math.sqrt(abNext) * x0 + keep * field[i];
	if (sigma > 0) state[i] += sigma * noise[i];
}`
					}
				}
			],
			ui: [
				{
					title: 'The checkpoint that ships, and why it is int8',
					body: `Two and a half million float32 weights are ten megabytes, which is not a download to put in front of a reader. Quantizing each tensor to signed bytes with its own scale factor takes that to 2.5 MB with no visible cost to the pictures — per-tensor rather than global, so a small tensor's range is not crushed by a large one's. The same file format carries the float32 version the offline trainer resumes from; only the header changes.`,
					code: {
						file: 'src/lib/diffusion/checkpoint.ts',
						code: `for (const l of leaves) {
	const data = l.dataSync() as Float32Array;
	let peak = 0;
	for (const v of data) peak = Math.max(peak, Math.abs(v));
	const scale = peak / 127 || 1;
	scales.push(scale);
	for (let i = 0; i < data.length; i++) {
		payload[off + i] = Math.max(-127, Math.min(127, Math.round(data[i] / scale)));
	}
	off += data.length;
}`
					}
				}
			],
			lab: {
				file: 'noise.zip',
				note: 'A denoising diffusion model on the emoji corpus in one file: corrupt, predict the noise, and walk back out of static, drawing eight fresh pictures every five hundred steps'
			}
		}
	]
};

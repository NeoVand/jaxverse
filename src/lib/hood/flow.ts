import type { HoodChapter } from './types';

export const flow: HoodChapter = {
	slug: 'flow',
	blocks: [
		{
			id: 'steer',
			lesson: 'guidance as arithmetic on a batch',
			lede: `Guidance and composition look like two features and are one loop. Every prompt you want to weigh is a branch; every branch is a row of the same batch; and the combination is a weighted sum of their outputs. Writing it that way means one forward pass per step no matter how many prompts are in play — and, more usefully here, one compiled kernel.`,
			ml: [
				{
					title: 'Three branches, always',
					body: `jit compiles per shape, so a sampler that ran two branches for one prompt and three for two would recompile the moment a reader typed in the second box — a visible stall in the middle of a demo. The sampler therefore always runs three: unconditional, prompt A, prompt B. A single-prompt request pays for one branch it does not need, which is far cheaper than a recompile, and prompt B simply carries a weight of zero.`,
					code: {
						file: 'src/lib/diffusion/runtime.ts',
						code: `for (let br = 0; br < BRANCHES; br++) {
	xBuf.set(state, br * count * dim);
}
for (let n = 0; n < count; n++) {
	writeCondition(condBuf, n, c, tau, NOTHING);
	writeCondition(condBuf, count + n, c, tau, req.a);
	writeCondition(condBuf, 2 * count + n, c, tau, condB);
}

const out = await net(
	tree.ref(req.params),
	np.array(xBuf).reshape([B, c.channels, c.res, c.res]),
	np.array(condBuf).reshape([B, condWidth(c)])
).data();`
					}
				},
				{
					title: 'The combination, in four lines',
					body: `This is the equation from the chapter, and it is worth seeing how little it is. <code>u</code> is what the model would do with no prompt; each conditional branch is read as a difference from it; the differences are scaled and added. Composition is not a separate code path from guidance — it is the same expression with a second term that is usually zero.`,
					code: {
						file: 'src/lib/diffusion/runtime.ts',
						code: `const aOff = count * dim;
const bOff = 2 * count * dim;
for (let i = 0; i < field.length; i++) {
	const u = out[i];
	field[i] = u + req.guidanceA * (out[aOff + i] - u) + wB * (out[bOff + i] - u);
}`
					}
				},
				{
					title: 'What a prompt actually is',
					body: `The conditioning vector is time features, then a tag block, then a style block, then two flags. The flags are the part worth explaining: an all-zero tag block could mean either "no prompt" or "a prompt whose words all missed", and those should not look the same to the model. Scaling by the inverse square root of the tag count is the other deliberate choice — without it a six-word prompt arrives six times louder than a one-word prompt and guidance behaves differently for long and short requests.`,
					code: {
						file: 'src/lib/diffusion/runtime.ts',
						code: `const tagBase = off + TIME_FEATURES;
const flagBase = tagBase + c.tags + c.styles;
if (cond.tags.length > 0) {
	const v = 1 / Math.sqrt(cond.tags.length);
	for (const t of cond.tags) if (t >= 0 && t < c.tags) dst[tagBase + t] = v;
	dst[flagBase] = 1;
}
if (cond.style !== null && cond.style >= 0 && cond.style < c.styles) {
	dst[tagBase + c.tags + cond.style] = 1;
	dst[flagBase + 1] = 1;
}`
					}
				},
				{
					title: 'Euler, and nothing else',
					body: `Set against the diffusion sampler in the previous chapter — the square roots, the α̅ lookups, the variance bookkeeping, the clamp that stops a short schedule exploding — the flow sampler is one subtraction. That is the practical dividend of choosing the interpolation rather than inheriting it.`,
					code: {
						file: 'src/lib/diffusion/runtime.ts',
						code: `if (c.objective === 'flow') {
	// Euler, along a path training made as straight as it could
	const dt = tau - tauNext;
	for (let i = 0; i < state.length; i++) state[i] -= dt * field[i];
}`
					}
				}
			],
			ui: [
				{
					title: 'Two workers, because the chapter is an argument',
					body: `Two plates here put the chapter-9 model beside the chapter-10 one, which means two sets of weights resident at once. jax-js takes one GPU device per worker, so the comparison plates boot a second lab rather than swapping checkpoints in and out of the first — swapping would mean a reload and a jit warm-up between every row, and the plate would spend more time reloading than drawing. The second lab is only booted by the plates that need it, so a reader who never scrolls to them never pays for it.`,
					code: {
						file: 'src/lib/components/demos/flow/lab.svelte.ts',
						code: `export const lab = new DiffusionLab({
	objective: 'flow',
	checkpoint: 'emoji-flow.bin',
	batch: 32,
	lr: 3e-4
});

/** The chapter-9 model, for the two comparison plates. */
export const rival = new DiffusionLab({
	objective: 'eps',
	checkpoint: 'emoji-eps.bin',
	batch: 32,
	lr: 3e-4
});`
					}
				},
				{
					title: 'One queue per lab',
					body: `Several plates share one worker, and each of them wants to sample whenever a slider moves. Left alone they interleave: two sample loops walking the same weights, each reading back frames the other has already stepped past. A promise chain per lab serializes the GPU work, so a plate's request waits its turn instead of corrupting someone else's walk.`,
					code: {
						file: 'src/lib/diffusion/lab.svelte.ts',
						code: `private run<T>(job: () => Promise<T>): Promise<T> {
	const next = this.queue.then(job, job);
	this.queue = next.catch(() => {});
	return next;
}`
					}
				}
			],
			lab: {
				file: 'flow.zip',
				note: 'Rectified flow on the emoji corpus in one file: interpolate, regress the velocity, integrate back with plain Euler, and compose two prompts you can change at the top of the file'
			}
		}
	]
};

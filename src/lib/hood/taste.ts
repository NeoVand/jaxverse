import type { HoodChapter } from './types';

export const taste: HoodChapter = {
	slug: 'taste',
	blocks: [
		{
			id: 'judge',
			lesson: 'a reward model is two forward passes and a subtraction',
			lede: `Another chapter with no jax-js in it, and again the absence is the lesson. The judge here is 401 parameters and the whole training set is a few dozen clicks; dispatching that to a GPU would cost more in bookkeeping than the arithmetic is worth. What the smallness lets you see is that <em>nothing about the mechanism changes with scale</em>. A frontier reward model is this file with more zeros.`,
			ml: [
				{
					title: 'Bradley–Terry, in the loss',
					body: `The entire fit. For each pair, score the winner and the loser, take the gap, push it through a sigmoid, and pay <code>−log</code> of the result — near zero when the model was confidently right, unboundedly large when it was confidently wrong. That asymmetry is what stops a judge from bluffing. The gradient at the bottom is one line: <code>σ(margin) − 1</code>, which is the same shape every classifier in this book descends.`,
					code: {
						file: 'src/lib/optim-rl/preference.ts',
						code: `for (const pair of pairs) {
	const rw = forwardInto(j, pair.winner, aw);
	const rl = forwardInto(j, pair.loser, al);

	const p = sigmoid(rw - rl);
	loss += -Math.log(Math.max(1e-12, p));
	const dm = p - 1;              // d/d(margin) of −log σ(margin)

	accumulate(j, grads, xw, hw1, hw2, dm);   // winner: push up
	accumulate(j, grads, al.x, al.h1, al.h2, -dm); // loser: push down
}`
					}
				},
				{
					title: 'Scoring a judge you have no test set for',
					body: `Thirty comparisons is not enough data to split, so the plate does not split it. Before each click, the judge — fitted only to everything earlier — is asked to call the pair, and whether it was right is recorded before the answer arrives. Every judgment is a test case exactly once, on its way to becoming training data. Statisticians call it <em>prequential</em> validation, and it is the right instrument any time the data arrives in a stream and there is too little of it to waste.`,
					code: {
						file: 'src/lib/components/demos/taste/taste-context.svelte.ts',
						code: `record(winner: Gene, loser: Gene): void {
	if (this.count >= MIN_PAIRS) {
		// ask before telling — the only honest order
		this.calls = [...this.calls, preferProb(this.judge, winner, loser) > 0.5];
	}
	this.pairs = [...this.pairs, { winner, loser }];
	this.count = this.pairs.length;
	this.refit();
}`
					}
				}
			],
			ui: [
				{
					title: 'Sixty ornaments without melting the page',
					body: `Every colour in a rosette is a <code>color-mix</code> over CSS variables, so the browser resolves it during style recalculation. Writing those expressions onto the forty-odd paths of an ornament costs forty resolutions each, and the taste map carries sixty ornaments — enough to spend most of a second on colour alone. So the mixes are resolved once into six custom properties on the <code>&lt;svg&gt;</code>, and every path just names one.`,
					code: {
						file: 'src/lib/components/demos/taste/Rosette.svelte',
						code: `<svg style="--rz:{ink}; --rz-fill:{fill}; --rz-frame:{frame}; …">
	{#each shape.outer.petals as d}
		<path {d} fill="var(--rz-fill)" stroke="var(--rz)" />
	{/each}
</svg>`
					}
				},
				{
					title: 'A field of 12,288 opinions',
					body: `The wash behind the specimens is the judge's score evaluated on a 128 × 96 grid — twelve thousand forward passes, about fifteen milliseconds — written straight into an <code>ImageData</code> and scaled up by the browser. The colour scale is deliberately symmetric about zero, so "better than the average ornament" is the same blue whatever your taste turned out to be.`
				}
			]
		},
		{
			id: 'goodhart',
			lesson: 'differentiating through a die roll, the easy way',
			ml: [
				{
					title: 'Reparameterization, because the judge is differentiable',
					body: `The policy is a Gaussian over ornament space, and the objective is an average over what it draws — so the knobs are buried inside the sampling and there is nothing obvious to differentiate. The fix is to roll the dice <em>first</em>: write <code>u = μ + σ·ε</code> with <code>ε</code> drawn before anyone consults the policy, and the sample becomes an ordinary differentiable function of the knobs. This works here only because the judge is a network we can differentiate. A language model, whose actions are discrete tokens, never gets this luxury — which is why every method in the next chapter is stuck with the noisier estimator.`,
					code: {
						file: 'src/lib/optim-rl/preference.ts',
						code: `for (let s = 0; s < n; s++) {
	for (let i = 0; i < N_GENES; i++) {
		eps[i] = gauss(rand);                       // dice, rolled first
		u[i] = p.mu[i] + Math.exp(p.logSigma[i]) * eps[i];
	}
	geneFromU(u, gene);
	scoreGrad(j, gene, dr);                       // ∂r/∂gene, by hand
	for (let i = 0; i < N_GENES; i++) {
		const d = dr[i] * gene[i] * (1 - gene[i]); // chain through the squash
		gMu[i] += d;
		gLs[i] += d * eps[i] * Math.exp(p.logSigma[i]);
	}
}`
					}
				},
				{
					title: 'The x-axis nobody plots',
					body: `Both curves on the plate are drawn against KL travelled from the reference rather than against training step, and that choice is the figure. Steps are an artefact of your learning rate; KL is how far the behaviour has actually moved, and it is the axis along which the proxy and the gold reward provably separate. Two diagonal Gaussians make it a closed form, so it costs nothing to report — and any run that does not report it is measuring optimization pressure in units nobody can compare.`,
					code: {
						file: 'src/lib/optim-rl/preference.ts',
						code: `export function klToRef(p: Policy): number {
	let kl = 0;
	const s0 = REF_SIGMA;
	for (let i = 0; i < N_GENES; i++) {
		const s = Math.exp(p.logSigma[i]);
		kl += Math.log(s0 / s) + (s * s + p.mu[i] * p.mu[i]) / (2 * s0 * s0) - 0.5;
	}
	return kl;
}`
					}
				}
			],
			ui: [
				{
					title: 'Measuring a thing that costs clicks',
					body: `A win rate from two rounds is worth ±35 points, so the plate draws a Wilson interval on every gold point rather than a bare dot — Wilson because the textbook formula collapses to a zero-width bar at 0 and 1 and lies about it. The button underneath buys one more round at every checkpoint. Watching the bars narrow as you pay for them is the most honest thing on the page.`
				},
				{
					title: 'Shuffled sides',
					body: `Every head-to-head randomizes which side the optimized ornament lands on. Without that, the plate would be measuring which way the reader looks first, and it would find a strong, entirely spurious effect.`
				}
			]
		},
		{
			id: 'leash',
			lesson: 'a penalty with a closed-form answer',
			ml: [
				{
					title: 'One step, two forces',
					body: `The update is the reward gradient minus β times the KL gradient, both in closed form. The only subtlety is the step size: a short leash is a stiff spring, and one fixed stride across a stiff spring is the Prologue's divergence — the step outruns the valley it is measuring. Scaling the whole objective by <code>1/(1+β)</code> moves the optimum nowhere and keeps the reader's slider from exploding, which is a step-size policy rather than a change of problem, and is worth labelling as one.`,
					code: {
						file: 'src/lib/optim-rl/preference.ts',
						code: `const k = lr / (1 + beta);            // step-size policy, not a new objective
for (let i = 0; i < N_GENES; i++) {
	const sig = Math.exp(p.logSigma[i]);
	const dklMu = p.mu[i] / s0sq;
	const dklLs = (sig * sig) / s0sq - 1;
	p.mu[i]       += k * (gMu[i] / n - beta * dklMu);
	p.logSigma[i] += k * (gLs[i] / n - beta * dklLs);
}`
					}
				},
				{
					title: 'Drawing the exact answer',
					body: `The left panel is not a sketch of <code>π* ∝ π_ref·exp(r/β)</code>; it is that formula evaluated on a 200-point grid. The reference density along one gene is a Gaussian in the unconstrained coordinate pushed through a logistic squash, which is why it has that lopsided shape near the edges — the same change-of-variables correction that a squashed policy needs anywhere else, and that is silently wrong in a great deal of published code.`,
					code: {
						file: 'src/lib/components/demos/taste/Leash.svelte',
						code: `function refDensity(g: number): number {
	const p = Math.min(1 - 1e-9, Math.max(1e-9, g));
	const u = Math.log(p / (1 - p));                      // the logit
	// N(0, σ²) in u, divided by |dg/du| = g(1−g)
	return Math.exp(-(u * u) / (2 * REF_SIGMA ** 2)) / (REF_SIGMA * p * (1 - p));
}`
					}
				}
			],
			ui: [],
			lab: {
				file: 'taste.zip',
				note: 'Fit a Bradley–Terry judge to synthetic preferences, over-optimize against it, and watch the gold reward turn over — a terminal script, no GPU required'
			}
		}
	]
};

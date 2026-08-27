import type { HoodChapter } from './types';
import { plateLabel } from '$lib/data/plates';

export const digits: HoodChapter = {
	slug: 'digits',
	blocks: [
		{
			id: 'training',
			lesson: 'batches, and held-out truth',
			lede: `Ten thousand real images raise the questions toy data never asks: how do you feed a GPU efficiently, and how do you keep yourself honest about memorization? The chapter's plates share one engine and one set of weights — train here and every other plate feels it — which is also this book's first taste of treating a model as <em>infrastructure</em>.`,
			ml: [
				{
					title: 'Minibatches, and a validation tail the gradients never touch',
					body: `Each step samples 128 of the 8,000 training rows into one <code>[128, 784]</code> tensor — enough parallel work to keep the GPU busy, small enough to step fifty times a second. The engine reserves a <em>validation tail</em> it never trains on; this chapter sizes that tail to be exactly the 2,000-row test set, so the header's test accuracy is measured on rows the gradient has never seen. The discipline costs three lines and is the difference between a claim and a hope.`,
					code: {
						file: 'src/lib/nn/worker.ts',
						code: `// B random rows, drawn only from below valStart — the row where the
// held-out tail begins, and the reason the test number means anything
function makeBatch(B: number, lo: number, hi: number, rand: () => number) {
	const idx: number[] = [];
	for (let b = 0; b < B; b++) idx.push(lo + Math.floor(rand() * Math.max(1, hi - lo)));
	for (let b = 0; b < B; b++) xBuf.set(dataX!.subarray(idx[b] * din, (idx[b] + 1) * din), b * din);
	const x = np.array(xBuf).reshape([B, din]);
	for (let b = 0; b < B; b++) lBuf[b] = (dataY as Int32Array)[idx[b]];
	return { x, y: nn.oneHot(np.array(lBuf, { dtype: np.int32 }), dout) };
}

const { x, y } = makeBatch(B, 0, valStart, rng);
const [lossVal, grads] = jitStep(tree.ref(params), x.ref, y.ref);`
					}
				},
				{
					title: 'Cross-entropy, exactly as the prose dissected it',
					body: `The formula from the chapter — <code>−log p<sub>y</sub></code> — is these two lines. Targets arrive one-hot, so multiplying the log-probabilities by <code>y</code> and summing simply <em>selects</em> the true class's entry; the mean over the batch is the loss in the header. Nothing about ten classes required new machinery: softmax and its log live in <code>jax.nn</code>, and the gradient of the whole thing comes from the same <code>valueAndGrad</code> as before.`,
					code: {
						file: 'src/lib/nn/worker.ts',
						code: `const logp = nn.logSoftmax(out, -1);
loss = np.mean(np.sum(logp.mul(y), -1).neg()); // −log p_y, averaged`
					}
				}
			],
			ui: [
				{
					title: 'MNIST as two PNGs',
					body: `The dataset ships as spritesheets — <code>mnist-train.png</code> (1.3 MB) and <code>mnist-test.png</code> — decoded at boot by drawing them into an offscreen canvas and reading pixels back, plus a 10 KB binary of labels. A PNG of digits compresses far better than raw floats, the browser's image decoder does the work, and the gallery can blit any digit with one <code>drawImage</code>.`
				},
				{
					title: 'One engine, four plates',
					body: `<code>digits-context.svelte.ts</code> holds the single <code>MlpEngine</code> at module scope, with Svelte 5 <code>$state</code> fields for step, loss, and accuracy. The classifier owns training; DrawPad and Inside just call <code>predict</code>, <code>activations</code>, and <code>inputgrad</code> against whatever the weights currently are. That is why your drawn 4 gets smarter while ${plateLabel('digits', 'classifier')} trains above it.`
				}
			]
		},
		{
			id: 'saliency',
			lesson: 'gradients of the input',
			ml: [
				{
					title: 'The evidence map is a gradient with respect to pixels',
					body: `Everything so far differentiated the loss with respect to <em>weights</em>. This plate's evidence square flips the question: hold the weights still and ask how the winning score moves as each <em>pixel</em> brightens — <code>∂score/∂x</code>, a 784-number gradient reshaped back into an image. Warm pixels argued for the verdict, cool pixels against. In the literature this is a saliency map; in jax it is the same <code>grad</code>, pointed at argument 0 instead of the parameters.`,
					code: {
						file: 'src/lib/nn/worker.ts',
						code: `// d/dx of the target-class score Σ logits·target — per-pixel evidence.
// grad differentiates argument 0 by default: here that is x, not the params.
f = jit((x: any, p: any, t: any) =>
	grad((xx: any, pp: any, tt: any) => np.sum(forward(pp, c, xx).mul(tt)))(x, p, t)
);`
					}
				}
			],
			ui: [],
			lab: {
				file: 'digits.zip',
				note: 'A real MNIST classifier — the spritesheet decoder, minibatch cross-entropy training, and test accuracy, ~150 lines'
			}
		}
	]
};

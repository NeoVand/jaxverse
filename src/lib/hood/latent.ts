import type { HoodChapter } from './types';

export const latent: HoodChapter = {
	slug: 'latent',
	blocks: [
		{
			id: 'waist',
			lesson: 'the reparameterization trick',
			lede: `The autoencoder is the same MLP engine again — the hourglass is just a layer list that narrows and widens — but the variational waist earns its own lesson. Sampling is not differentiable; the <em>reparameterization trick</em> makes it so, and it is three lines here.`,
			ml: [
				{
					title: 'An hourglass is a layer list',
					body: `Depth 2 with a two-number waist is <code>[784, 256, 64, 2, 64, 256, 784]</code>, loss <code>mse</code>, and the targets are the inputs themselves — <code>y === x</code> is the entire specification of self-supervision. The one structural rule the prose insisted on is enforced by an index: the waist layer carries no bend, because its job is to be a plain coordinate system.`,
					code: {
						file: 'src/lib/components/demos/latent/latent-context.svelte.ts',
						code: `private config() {
	return {
		layers: layersFor(this.latentDim, this.depth), // 784 … waist … 784
		activation: this.activation,
		loss: 'mse' as const,
		vae: { at: latentLayerFor(this.depth), beta: (this.latentDim <= 3 ? 1 : 0.3) / DIM },
		lr: this.depth === 3 ? 1.2e-3 : 3e-3,
		batchSize: 128,
		seed: 7,
		valFraction: 0.2
	};
}`
					}
				},
				{
					title: 'The trick that lets a gradient cross a coin flip',
					body: `The encoder proposes a mean and a spread; the decoder needs a <em>sample</em>. Sampling has no derivative — but <code>z = μ + σ·ε</code> with the noise ε drawn outside the traced function does, because the randomness is now a constant input and the gradient flows through μ and σ untouched. That re-plumbing is the reparameterization trick, the one line that made VAEs trainable, and the KL rent from the prose is charged right beside it.`,
					code: {
						file: 'src/lib/nn/worker.ts',
						code: `/** The variational waist: read the layer's output as a mean and a
 * log-variance. With noise it passes a sample of that Gaussian and reports
 * how far the Gaussian sits from the standard normal. */
function waist(h: any, eps: any | null): { z: any; kl: any | null } {
	const [mu, lv] = np.split(h, 2, 1); // h holds [μ | log σ²] side by side
	if (!eps) {
		lv.dispose();
		return { z: mu, kl: null }; // deterministic pass: just the mean
	}
	// KL(N(μ, σ²) ‖ N(0, 1)) = ½ Σ (μ² + σ² − 1 − log σ²)
	const kl = np
		.mean(np.sum(np.square(mu.ref).add(np.exp(lv.ref)).sub(1).sub(lv.ref), 1))
		.mul(0.5);
	return { z: mu.add(np.exp(lv.mul(0.5)).mul(eps)), kl }; // z = μ + σ·ε
}`
					}
				}
			],
			ui: []
		},
		{
			id: 'halves',
			lesson: 'running half a network',
			lede: `The chapter breaks the network in half on purpose: the map plots <code>encode(x)</code>, the manifold plots <code>decode(z)</code>, and each half is its own worker op.`,
			ml: [
				{
					title: 'Encode and decode as separate calls',
					body: `Once trained, the halves work alone. The map runs the front half over two thousand held-out digits and plots the 2-D result; the manifold runs the back half over a lattice of addresses <em>no digit chose</em>; the search plate encodes your stroke and measures distances in the waist's own units. In jax terms each is a partial forward pass — start the fold at a different layer, or stop it early — which the engine exposes as <code>activations</code> (read any layer, including the waist) and <code>forwardFrom</code> (enter at any layer, such as just past the waist).`,
					code: {
						file: 'src/lib/components/demos/latent/latent-context.svelte.ts',
						code: `/** Latent coordinates (n × d): the bottleneck layer of activations(). */
const k = latentLayerFor(this.depth);
const acts = await this.engine.activations(x, n, 1024);
return { z: acts.layers[k], d: acts.widths[k] };

/** The manifold: enter the network just past the waist, decode addresses. */
async decode(z: Float32Array, n: number, chunk = 16): Promise<Float32Array> {
	return clamp01(await this.engine.forwardFrom(decoderFromFor(this.depth), z, n, chunk));
}`
					}
				}
			],
			ui: [
				{
					title: 'A map that refuses to teleport',
					body: `Training moves the whole latent cloud every few hundred milliseconds. The map keeps its footing by easing its viewport — center and span chase the cloud's bounding box with an exponential lag — and by lerping each point toward its new address rather than jumping. The same easing runs the manifold's 441 tiles, decoded in slices so no single frame stalls.`
				},
				{
					title: 'The drawing pad, shared across chapters',
					body: `The brush that draws a query here is the same module the digits chapter uses (<code>src/lib/data/brush.ts</code>): a Gaussian stamp accumulated into a 28 × 28 float grid, recentered by ink mass before encoding — because the encoder never saw a digit hiding in a corner. Sharing the module is why your handwriting feels identical on both pads.`
				}
			],
			lab: {
				file: 'latent.zip',
				note: 'A VAE on MNIST with the reparameterized waist, printing its latent map as ASCII every few hundred steps — 300 test digits, each character a real digit at its own address'
			}
		}
	]
};

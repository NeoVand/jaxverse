import type { HoodChapter } from './types';
import { plateLabel } from '$lib/data/plates';

export const language: HoodChapter = {
	slug: 'language',
	blocks: [
		{
			id: 'transformer',
			lesson: 'Lesson 5a — a transformer, whole, in jax-js',
			lede: `The scribe is a complete GPT-style transformer — embeddings, positional table, causal self-attention, MLP blocks, a language-model head — in one readable file, <code>src/lib/llm/model.ts</code>, trained by the same <code>valueAndGrad</code>-plus-Adam loop as every MLP before it. It runs in its own worker and asks for WebGPU by name: a transformer is the first model in this book big enough that nothing less will do.`,
			ml: [
				{
					title: 'A transformer is a pytree too',
					body: `The whole model is still just a nested record of arrays — a token embedding, a positional table, per-layer attention and MLP matrices, a head. Initialization is small Gaussians via <code>jax.random</code>, with the head started near zero so the untrained scribe begins at an honest uniform guess.`,
					code: {
						file: 'src/lib/llm/model.ts',
						code: `export function initParams(cfg: ModelConfig, seed: number): ModelParams {
	return {
		wte: random.normal(nk(), [cfg.vocab, cfg.nEmbd]).mul(0.02), // tokens
		wpe: random.normal(nk(), [cfg.blockSize, cfg.nEmbd]).mul(0.02), // positions
		layers: /* per layer: wq, wk, wv, wo, mlpFc1, mlpFc2, norms */
		lmHead: random.normal(nk(), [cfg.nEmbd, cfg.vocab]).mul(0.001)
	};
}`
					}
				},
				{
					title: 'The loss is cross-entropy over the next token',
					body: `Chapter 3's loss returns with a bigger alphabet: at every position, the target is one-hot over the vocabulary and the loss selects the log-probability the model gave the true next token. One implementation note worth stealing: tokens go in as one-hot matrices multiplied against the embedding table, rather than indexed lookups — a matmul differentiates cleanly everywhere, which keeps the whole model on the fast path.`,
					code: {
						file: 'src/lib/llm/model.ts',
						code: `export function lossFn(params, cfg, tokenOH, posOH, targetOH) {
	const logprobs = forwardLogprobs(params, cfg, cfg.blockSize, tokenOH, posOH);
	return np.mean(np.sum(logprobs.mul(targetOH.reshape([-1, cfg.vocab])), -1).neg());
}`
					}
				},
				{
					title: 'Training and sampling without a stutter',
					body: `The worker alternates: a few gradient steps, then — when the page asks — a sampling pass with the same weights, temperature applied to the logits before the draw. Loss curve and specimen text come back over the RPC as separate streams, which is why sampling a paragraph no longer freezes the curve. The tokenizer the scribe reads through is the same BPE the plate grows live, word-boundary merges and all (<code>src/lib/data/bpe.ts</code>); train it yourself in ${plateLabel('language', 'tokenizer')} or fall back to the 300-merge snapshot the book ships.`
				}
			],
			ui: [
				{
					title: 'Stable text under a firehose',
					body: `Sampled text changes length every few seconds, which used to bounce the page. The sample cards clamp to a fixed number of lines with an ellipsis, ghost cards keep the archive grid rectangular, and the loss chart owns its height outright — layout writes once, then only pixels change.`
				}
			]
		},
		{
			id: 'attention',
			lesson: 'Lesson 5b — the attention block, undissolved',
			ml: [
				{
					title: 'The attention block, undissolved',
					body: `Here is the equation the chapter's SVG dissected, as it actually executes. Queries, keys, and values are three matmuls; <code>nn.dotProductAttention</code> does the scaled dot-product, the softmax, and the causal mask in one fused call; a projection folds the heads back together, and the MLP does its two-matmul think. Residual adds thread the whole thing. Twelve lines is genuinely all of it.`,
					code: {
						file: 'src/lib/llm/model.ts',
						code: `for (let li = 0; li < cfg.nLayer; li++) {
	const layer = params.layers[li];
	// q, k, v: three matmuls, reshaped into heads
	const attnOut = nn.dotProductAttention(qH, kH, vH, { isCausal: true });
	x = np.dot(attnOut.reshape([-1, cfg.nEmbd]), layer.wo).add(xRes);
	x = nn.relu(np.dot(x, layer.mlpFc1));
	x = np.dot(x, layer.mlpFc2).add(mlpRes);
}
const logits = np.dot(x, params.lmHead);
return nn.logSoftmax(logits, -1);`
					}
				}
			],
			ui: [
				{
					title: 'Attention maps straight from the forward pass',
					body: `The inspector re-runs one forward pass with a flag that also returns each head's attention matrix, then paints them as heat ramps. Nothing is instrumented or approximated: the map you hover <em>is</em> the tensor the model used, fetched once per reading, not per frame.`
				}
			],
			lab: {
				file: 'language.zip',
				note: "A tiny GPT trained on this book's story corpus in your browser — attention, sampling, and the loss curve in the console"
			}
		}
	]
};

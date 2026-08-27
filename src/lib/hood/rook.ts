import type { HoodChapter } from './types';

export const rook: HoodChapter = {
	slug: 'rook',
	blocks: [
		{
			id: 'engine',
			lesson: 'one engine, three diets',
			lede: `Rook is the course's final exam: everything from the previous seven lessons, running as one system. The transformer is the scribe's architecture at 1.3 million parameters, and the three training stages are <em>one engine with three diets</em>, not three models.`,
			ml: [
				{
					title: 'Three stages, one engine, three diets',
					body: `Fine-tuning is not new machinery. The SFT plate swaps the token stream under the same worker — weights and Adam state untouched — and lowers the learning rate; the loop that runs afterwards is byte-for-byte the pretraining loop. That is the honest shape of the industry's pipeline, and here it is small enough to see whole: η = 1.2·10⁻³ to pretrain, 3·10⁻⁴ to fine-tune, 10⁻⁴ to reinforce.`,
					code: {
						file: 'src/lib/components/demos/rook/rook-context.svelte.ts',
						code: `async useCorpus(which: LabCorpus): Promise<void> {
	if (!this.engine || !this.data || this.corpus === which) return;
	const tokens = which === 'sft' ? await this.ensureSftTokens() : this.data.tokens;
	await this.engine.setTokens(tokens); // new diet; weights + Adam state survive
	this.corpus = which;
}`
					}
				},
				{
					title: 'Checkpoints: a pytree flattened, then quantized',
					body: `Exporting a model is walking its tree: <code>tree.leaves</code> gives the arrays in a stable order, and concatenating them into one <code>Float32Array</code> is the whole runtime format — it is how the arena snapshots stages and how the play plate always plays the current weights. The shipped time machine goes one step further: symmetric per-tensor int8, so each five-megabyte snapshot travels as 1.3 MB and dequantizes with one multiply per leaf.`,
					code: {
						file: 'src/lib/data/rook.ts',
						code: `/** Fetch one weight snapshot and dequantize: symmetric per-tensor int8,
 * so f32[i] = i8[i] · scales[leaf], with leaf boundaries from the manifest. */
export async function loadRookWaypoint(step: number): Promise<Float32Array> {
	const raw = new Int8Array(await fetchBin(wp.file));
	const out = new Float32Array(raw.length);
	let o = 0;
	for (let leaf = 0; leaf < manifest.leafSizes.length; leaf++) {
		const scale = wp.scales[leaf];
		for (let i = 0; i < manifest.leafSizes[leaf]; i++, o++) out[o] = raw[o] * scale;
	}
	return out;
}`
					}
				}
			],
			ui: [
				{
					title: 'chess.js is the judge, never the player',
					body: `The rules engine (<code>chess.js</code>) supplies legality, board state, and the ✕ marks where a sampled game went off the rails — but it never chooses a move. Rook's replies are sampled from its own distribution, masked to legal moves only at the last moment, and the legality gauge reports how much belief the mask had to rescue.`
				}
			]
		},
		{
			id: 'rlvr',
			lesson: 'RLVR: chapter 6, compiled by chapter 5',
			ml: [
				{
					title: 'RLVR: chapter 6, compiled by chapter 5',
					body: `The reinforcement step is REINFORCE wearing transformer clothes. Rollouts are sampled in a batch, a verifier scores them, and the group's own mean and spread turn scores into advantages — no learned critic, exactly as in GRPO. The gradient step is then an advantage-weighted negative log-likelihood on the <em>generated</em> tokens only, jitted like any other loss. Compare it with the pretraining loss: one extra multiply by the advantage weights. That proximity is the deepest fact in the chapter.`,
					code: {
						file: 'src/lib/llm/worker.ts',
						code: `// advantage-weighted NLL over generated tokens only: ww carries adv · mask
jitRl = jit((p: any, tok: any, pos: any, tgt: any, ww: any) =>
	valueAndGrad((pp: any) => {
		const logp = forwardLogprobs(pp, c, S, tok, pos);
		return np.sum(logp.mul(tgt.reshape([-1, c.vocab]).mul(ww))).neg();
	})(p)
);
const [lossVal, grads] = jitRl(tree.ref(params), tokenOH.ref, posOH.ref, targetOH.ref, w.ref);
const [updates, newOptState] = solver!.update(grads, optState, tree.ref(params));
params = applyUpdates(params, updates);`
					}
				}
			],
			ui: []
		},
		{
			id: 'arena',
			lesson: 'weights as hot-swappable state',
			ml: [
				{
					title: 'The arena: weights as hot-swappable state',
					body: `The comparison plate is only possible because a model is just its flat checkpoint. To ask three stages about one position, the lab saves the current weights, loads each stage's snapshot into the same engine, reads one next-move distribution, and restores — marking itself busy so the play plate cannot query mid-swap. No second GPU allocation, no second model: weights in, question, weights out.`
				}
			],
			ui: [
				{
					title: 'Arrows that fan',
					body: `The arena's board draws each stage's chosen move as an arrow in the stage's color, computed in <code>Board.svelte</code> as plain SVG lines with polygon heads. When two stages pick the same move, the arrows fan apart by a few degrees so both stay legible — a two-line rotation, and the difference between a comparison you can read and one you cannot.`
				}
			],
			lab: {
				file: 'rook.zip',
				note: 'Pretrain a small Rook on real games in your browser, watch the legal-move rate emerge, then fine-tune on the curated corpus'
			}
		}
	]
};

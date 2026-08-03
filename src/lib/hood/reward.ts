import type { HoodChapter } from './types';

export const reward: HoodChapter = {
	slug: 'reward',
	blocks: [
		{
			id: 'pendulum',
			lesson: 'Lesson 6a — knowing when not to use the library',
			lede: `An honest interlude in the jax course: <em>neither plate in this chapter uses jax-js</em>, and the reason is worth more than another code sample. Both policies are tiny — a table of logits, a 5 × 33 matrix — and both environments must step physics at 50 Hz in lockstep with learning. When each "tensor op" is thirty multiplications, the cost of dispatching to a GPU dwarfs the arithmetic itself. jax earns its keep when the math outweighs the bookkeeping; here it would not, so the chapter is plain TypeScript, and REINFORCE is laid bare because of it.`,
			ml: [
				{
					title: 'The pendulum: same rule, engineered signal',
					body: `The double pendulum's policy is a softmax over 33 hand-crafted features — and every hard-won lesson lives in the <em>reward and curriculum</em>, not the optimizer: whitened advantages so one lucky episode cannot yank the weights, an entropy bonus so the policy keeps exploring, a spin tax so it cannot cheat by becoming a propeller, and a replay buffer of its own successful deliveries so the catch gets practiced from states it can actually reach. RL's dirty secret, on display: the update rule is the easy part.`,
					code: {
						file: 'src/lib/optim-rl/dpole.ts',
						code: `// whiten advantages across the batch: mean 0, unit spread, clamped
const advs = eps.flatMap((e) => e.advs);
const sd = Math.sqrt(advs.reduce((a, v) => a + v * v, 0) / advs.length) || 1;
for (const e of eps)
	for (let t = 0; t < e.advs.length; t++) {
		const a = Math.max(-3, Math.min(3, e.advs[t] / sd));
		// θ[f, k] += lr · a · (1[k=aₜ] − πₖ) · featₜ[f]   (+ entropy, − decay)
	}`
					}
				},
				{
					title: 'Six learners racing on worker threads',
					body: `REINFORCE's discovery is luck-of-the-stream, so the demo manufactures luck: up to six workers each train an independent policy from an independent seed, the stage performs whichever is fittest, and stragglers adopt the champion's weights when they fall far behind. It is population-based training in miniature — and because each learner is plain TS, six of them fit in the browser without a GPU at all.`,
					code: {
						file: 'src/lib/optim-rl/dpole.worker.ts',
						code: `// each hall: own θ, own baseline, own curriculum, own rng
async function round() {
	const eps = collectEpisodes(theta, curriculum, rand, BATCH);
	dpoleReinforceUpdate(theta, baseline, eps, lr);
	fitness = 0.9 * fitness + 0.1 * scoreOf(eps);
	postMessage({ kind: 'report', fitness, theta: maybeShare() });
	setTimeout(round, 0); // yield, then keep practicing
}`
					}
				}
			],
			ui: [
				{
					title: 'Physics you can shove',
					body: `The cart-and-two-links dynamics come from the Lagrangian, integrated with RK4 at 50 Hz, with quadratic drag on the pins so a botched policy cannot wind the system up to numerical infinity. Your click-drag is injected as an external force on the hinge for a few frames — the red arrow — through exactly the same step function training uses.`
				},
				{
					title: 'One big stage, six small ones',
					body: `The race strip renders each worker's live policy into its own small canvas from the same draw routine as the main stage, scaled down. The champion's outline and score are just state from the latest worker reports; adopting a laggard is one <code>postMessage</code> carrying the champion's θ.`
				}
			]
		},
		{
			id: 'reinforce',
			lesson: 'Lesson 6b — REINFORCE in nine lines',
			ml: [
				{
					title: 'REINFORCE in nine lines',
					body: `The gridworld's whole learning rule. For every step of an episode: compute the return-to-go <code>G<sub>t</sub></code>, subtract a learned per-state baseline so the signal means "better or worse <em>than usual</em>", and nudge the chosen action's logit by exactly the policy-gradient formula the prose derived — <code>(𝟙[i=a] − π<sub>i</sub>)</code> is the derivative of log-softmax, written out.`,
					code: {
						file: 'src/lib/optim-rl/gridworld.ts',
						code: `export function reinforceUpdate(world, theta, baseline, ep, lr): void {
	const g = returnsToGo(ep.rewards, world.gamma, tail);
	for (let t = 0; t < ep.states.length; t++) {
		const s = ep.states[t];
		const adv = g[t] - baseline[s]; // better or worse than usual?
		baseline[s] += 0.1 * (g[t] - baseline[s]);
		const pi = softmax4(theta, s);
		for (let i = 0; i < 4; i++) {
			// θ[s,i] += lr · advantage · (1[i=a] − π_i)
			theta[s * 4 + i] += lr * adv * ((i === ep.actions[t] ? 1 : 0) - pi[i]);
		}
	}
}`
					}
				}
			],
			ui: [],
			lab: {
				file: 'reward.zip',
				note: 'The gridworld REINFORCE agent as a terminal script — watch the policy sharpen episode by episode, no GPU required'
			}
		}
	]
};

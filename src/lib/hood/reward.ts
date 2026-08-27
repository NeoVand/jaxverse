import type { HoodChapter } from './types';

export const reward: HoodChapter = {
	slug: 'reward',
	blocks: [
		{
			id: 'reinforce',
			lesson: 'REINFORCE in nine lines, and the world around it',
			lede: `An honest interlude in the jax course: <em>neither plate in this chapter uses jax-js</em>, and the reason is worth more than another code sample. Both policies are tiny — a table of logits, a 5 × 33 matrix — and both worlds must be stepped in lockstep with learning. When each "tensor op" is thirty multiplications, the cost of dispatching to a GPU dwarfs the arithmetic itself. jax earns its keep when the maths outweighs the bookkeeping; here it would not, so the chapter is plain TypeScript, and REINFORCE is laid bare because of it.`,
			ml: [
				{
					title: 'The whole learning rule',
					body: `For every leg of a passage: compute the return-to-go <code>G<sub>t</sub></code>, subtract a learned per-cell baseline so the signal means "better or worse <em>than usual</em>", and nudge the chosen heading's logit by exactly the policy-gradient formula the prose derived — <code>(𝟙[i=a] − π<sub>i</sub>)</code> is the derivative of log-softmax, written out. The entropy term is the only addition, and it pays the policy a little to stay undecided.`,
					code: {
						file: 'src/lib/optim-rl/chart.ts',
						code: `const g = returnsToGo(p.rewards, sea.gamma, tail);
for (let t = 0; t < p.states.length; t++) {
	const s = p.states[t], a = p.actions[t];
	policyAt(theta, s, probs);
	const adv = g[t] - baseline[s];        // better or worse than usual?
	for (let i = 0; i < N_HEADINGS; i++) {
		const pg = adv * ((i === a ? 1 : 0) - probs[i]);
		const eg = -probs[i] * (Math.log(probs[i] + 1e-12) + h);  // stay curious
		theta[s * N_HEADINGS + i] += lr * (pg + entropy * eg);
	}
	baseline[s] += 0.15 * (g[t] - baseline[s]);
}`
					}
				},
				{
					title: 'The physics is four lines, and it is the whole chapter',
					body: `Everything that makes this world interesting lives in one function: how fast a boat goes at a given angle to the wind. Zero inside the no-go zone, quickest on a beam reach, easing off downwind. Divide the time cost of a leg by that number and tacking is no longer something you have to program — it is the cheapest route, and the policy is left to find out.`,
					code: {
						file: 'src/lib/optim-rl/chart.ts',
						code: `export function boatSpeed(twa: number): number {
	if (twa <= NO_GO) return 0;
	if (twa < Math.PI / 2) {
		const u = (twa - NO_GO) / (Math.PI / 2 - NO_GO);
		return u * u * (3 - 2 * u);   // smoothstep: no cliff at the edge
	}
	const v = (twa - Math.PI / 2) / (Math.PI / 2);
	return 1 - 0.35 * v;             // running is slower than reaching
}`
					}
				},
				{
					title: 'The line that made it learnable at all',
					body: `The least glamorous three lines in the chapter, and without them it does not work. A wind makes a uniform policy drift downwind, so a harbour to windward is somewhere a random walk essentially never reaches; on some wind bearings the learner never found it and settled instead for sailing onto a shoal, which was cheaper than running out of clock. Starting one passage in five from a random patch of water fixes it completely — measured across five seeds and all eight wind directions, every run arrives. There is a unit test asserting the <em>failure</em> without it, so if that ever stops happening the constant can go.`,
					code: {
						file: 'src/lib/optim-rl/chart.ts',
						code: `const from = scatter > 0 && rand() < scatter
	? water[(rand() * water.length) | 0]   // practise from somewhere reachable
	: sea.start;
const p = runPassage(sea, theta, rand, from);
reinforceUpdate(sea, theta, baseline, p, lr, opts.entropy);`
					}
				}
			],
			ui: [
				{
					title: 'A policy drawn as a current chart',
					body: `Eight streaks per cell, one per heading, length and opacity both riding on the probability — so a confident cell reads as a single stroke and an undecided one as a star. Headings whose sails would luff are drawn in vermilion rather than hidden, because watching them fade is watching the policy learn the one rule the world enforces immediately.`
				},
				{
					title: 'Soundings',
					body: `The value layer is the per-cell baseline, bilinearly upsampled four times and run through the same <code>d3-contour</code> the Prologue's loss surface uses. A value function is a landscape and this book has already taught you to read one; drawing it as depth contours on a chart is the same figure wearing the right clothes.`
				}
			]
		},
		{
			id: 'pendulum',
			lesson: 'when the update rule is the easy part',
			ml: [
				{
					title: 'Same rule, engineered signal',
					body: `The double pendulum's policy is a softmax over 33 hand-crafted features — and every hard-won lesson lives in the <em>reward and curriculum</em>, not the optimizer: whitened advantages so one lucky episode cannot yank the weights, an entropy bonus so the policy keeps exploring, a spin tax so it cannot cheat by becoming a propeller, and a replay buffer of its own successful deliveries so the catch gets practised from states it can actually reach. RL's dirty secret, on display: the update rule is the easy part.`,
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
					body: `REINFORCE's discovery is luck-of-the-stream, so the demo manufactures luck: up to six workers each train an independent policy from an independent seed, and the stage performs whichever is fittest. It is population-based training in miniature — and because each learner is plain TS, six of them fit in the browser without a GPU at all.`,
					code: {
						file: 'src/lib/optim-rl/dpole.worker.ts',
						code: `// each hall: own θ, own baseline, own curriculum, own rng
async function round() {
	const eps = collectEpisodes(theta, curriculum, rand, BATCH);
	dpoleReinforceUpdate(theta, baseline, eps, lr);
	fitness = 0.9 * fitness + 0.1 * scoreOf(eps);
	postMessage({ kind: 'report', fitness, theta, finds });
	setTimeout(round, 0); // yield, then keep practicing
}`
					}
				},
				{
					title: 'What passes between them',
					body: `A hall's own gradient is first-order in its own rollouts and knows nothing about the other five. Two things cross the gap. <em>Experience</em>: a hall that lands a delivery cleaner than any it has managed hands that state to everyone — filed on a small <em>guest</em> shelf, deliberately separate from its own arrivals, because sharing into one buffer made every hall practise the catch on the leader's clean deliveries and never on the sloppy ones its own swing produces. And the <em>gradient</em> of the episode that found it: 165 numbers, where the episode itself is a hundred kilobytes, stepped along at a fraction of the learning rate by the halls behind.`,
					code: {
						file: 'src/lib/optim-rl/dpole.worker.ts',
						code: `// a personal best, announced: the state to practise from…
finds.push({ state: ep.delivered, bonus,
             grad: dpoleGradient(theta, baseline, [ep]) });
// …and on the receiving side, one step along somebody else's lesson
if (m.grad)
	for (let i = 0; i < theta.length; i++)
		theta[i] += lr * SHARED_GRAD * m.grad[i];`
					}
				},
				{
					title: 'And a second learner, for who to listen to',
					body: `Influence applied indiscriminately made the pool fail <em>together</em>: mean pool fitness 0.33, against 0.82 for six halls sharing nothing at all — a hall mid-discovery that keeps being dragged toward the leader never finishes its own idea. So the demo stopped deciding. Every hall runs a second REINFORCE learner whose actions are &ldquo;take hall <em>j</em>'s weights a step&rdquo; and &ldquo;refuse everyone&rdquo;, whose policy is a softmax over one logit per peer plus a refusal logit added to a prior, and whose reward is whether it climbed the pool's ranking over the next two seconds. The adjacency matrix drawn beneath the swarm is not a diagram of the design — it is those weights, live.`,
					code: {
						file: 'src/lib/optim-rl/dpole.worker.ts',
						code: `// graded by RANK, the one thing a competitor cares about
const r = (past.rank - rank) / (n - 1);
const adv = r - socBase;
socBase += SOC_BASE_LR * (r - socBase);
for (let k = 0; k <= n; k++)          // k = n is "refuse everyone"
	socLogit[k] += SOC_LR * adv * ((k === past.pick ? 1 : 0) - past.probs[k]);`
					}
				}
			],
			ui: [
				{
					title: 'Physics you can shove',
					body: `The cart-and-two-links dynamics come from the Lagrangian, integrated with RK4 at 50 Hz, with quadratic drag on the pins so a botched policy cannot wind the system up to numerical infinity. Your click-drag is injected as an external force on the hinge for a few frames — the red arrow — through exactly the same step function training uses.`
				},
				{
					title: 'One big stage, six small ones — or all six at once',
					body: `The race strip renders each worker's live policy into its own small canvas from the same draw routine as the main stage, scaled down. Press Swarm and those six rigs move onto the one rail, each in its own hue with its own tip trail, while the strip becomes a field where every hall stands at its two skills and the ribbons between them are the social weights the halls trained on each other. Holding is drawn as light rather than geometry: the rig burns green in proportion to how near dead-vertical and dead-calm it is, so a shove reads as the glow going out and coming back.`
				}
			],
			lab: {
				file: 'reward.zip',
				note: 'The sailing chart as a terminal script — watch a policy that has never heard of tacking work out how to beat to windward, no GPU required'
			}
		}
	]
};

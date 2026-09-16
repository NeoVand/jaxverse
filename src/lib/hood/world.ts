import type { HoodChapter } from './types';

export const world: HoodChapter = {
	slug: 'world',
	blocks: [
		{
			id: 'learn',
			lesson: 'one encoder on both sides of a prediction',
			lede: 'The target is the next picture passed through the same learned encoder. There is no frozen teacher here: changing the encoder changes both what the predictor receives and what it is asked to predict.',
			ml: [
				{
					title: 'The action belongs between two pictures',
					body: 'Each batch contains three consecutive observations. The predictor receives the first two embeddings and two actions: the action between the context pictures, then the action whose consequence it must predict. The final embedding is the target. The batch contains no joint-angle or velocity labels.',
					code: {
						file: 'src/lib/world/model.ts',
						code: `const z = encode(params.encoder, pixels.reshape([3 * batch, c.resolution ** 2]))
	.reshape([3, batch, c.latent]);
const pred = predict(params.predictor, z.ref.slice(0), z.ref.slice(1), actions);
const prediction = np.mean(np.square(pred.sub(z.ref.slice(2))));
const regularizer = sigreg(z, directions);
const total = prediction.ref.add(regularizer.ref.mul(c.regularization));
return [total, np.stack([prediction, regularizer])];`
					}
				},
				{
					title: 'A small residual predictor',
					body: 'This browser model uses multilayer perceptrons rather than LeWM’s transformer: a 32 × 32 image passes through 128 hidden units into eight coordinates; the predictor has two 128-unit hidden layers. Its output adds a learned change to the current embedding. That residual connection makes persistence available at initialization; it does not supply the correct dynamics. The held-out persistence baseline checks whether predicting a change helps.'
				},
				{
					title: 'A reference is not a detached target',
					body: 'In jax-js, <code>.ref</code> retains an array for another consumer. It does not stop differentiation. The gradient of this objective reaches the encoder through the context, the target, and the regularizer. The optimizer updates one parameter tree containing the encoder and predictor.'
				}
			],
			ui: [
				{
					title: 'One experiment travels through the chapter',
					body: 'The page owns one lab and passes it to all six plates. Training changes the model the later plates use. Leaving the chapter disposes of that lab; returning starts a new local experiment. Observation pixels are generated separately from the display, keeping goals, traces, and forecasts outside the sensor.'
				}
			]
		},
		{
			id: 'spread',
			lesson: 'a distribution constraint with a gradient',
			ml: [
				{
					title: 'Look along many directions',
					body: 'SIGReg projects embeddings onto fresh unit directions. For each projection it compares mean sine and cosine values with the characteristic function of a standard Gaussian. The expectations are over independent examples at each time position. Integration weights, the batch-size factor, and the averaging axes follow the LeWM implementation.',
					code: {
						file: 'src/lib/world/model.ts',
						code: `const x = np.dot(z.reshape([times * batch, dim]), directions)
	.reshape([times, batch, projections, 1])
	.mul(np.array(ts));
const real = np.mean(np.cos(x.ref), 1).sub(np.array(phi));
const imag = np.mean(np.sin(x), 1);
const err = np.square(real).add(np.square(imag));
return np.mean(np.sum(err.mul(np.array(weights)), -1)).mul(batch);`
					}
				},
				{
					title: 'What the constraint cannot certify',
					body: 'No centering, whitening, or unit normalization is applied to the embeddings before this comparison. Those operations would change the objective. Finite batches and projection directions approximate a distributional test; a useful pose representation and accurate dynamics still have to be measured separately. Averaging independent microbatch losses is not equivalent to evaluating the statistic over their combined batch.'
				}
			],
			ui: [
				{
					title: 'Keep the comparison honest',
					body: 'The regularized and unregularized runs use matched initialization and experience. Report the actual prediction term and representation spread, including a run that does not collapse dramatically. Different learned scales make raw latent errors across two encoders an incomplete comparison.'
				}
			]
		},
		{
			id: 'plan',
			lesson: 'optimize actions while holding the model fixed',
			ml: [
				{
					title: 'The goal is a cost on predicted embeddings',
					body: 'The planner samples bounded torque sequences and evaluates them with the learned predictor. It keeps the better candidates to refine its next samples. The selected sequence is one it actually evaluated. Only its first action reaches the mechanism before a fresh observation starts the next search.',
					code: {
						file: 'src/lib/world/planner.ts',
						code: `const window = hold ? Math.min(4, horizon) : 1;
let sum = 0;
for (let t = horizon - window; t < horizon; t++) {
	for (let d = 0; d < goal.length; d++)
		sum += (latents[t * goal.length + d] - goal[d]) ** 2;
}
return sum / (window * goal.length);`
					}
				},
				{
					title: 'The same effort preference in both intentions',
					body: 'The goal term above is part of the cost. Both intentions also add 0.01 times the mean squared normalized motor command over the sequence. This is a planner preference, independent of the model’s training objective. Keeping its coefficient fixed makes the change from a final pose to a final window explicit.',
					code: {
						file: 'src/lib/world/planner.ts',
						code: `const effort = candidateActions.reduce((sum, action) => sum + action * action, 0)
	/ candidateActions.length;
ranked.push({
	cost: latentGoalCost(latents, goal, horizon, options.hold ?? false)
		+ (options.effort ?? 0) * effort,
	actions: candidateActions,
	latents
});`
					}
				},
				{
					title: 'A display readout stays outside the decision',
					body: 'The pose readout fits 1,024 independently sampled, simulator-labeled poses after freezing the encoder; 256 more poses test it. It draws diagnostic ghosts from predicted embeddings. The controller receives neither that readout nor the simulator transition function: its score uses the embedding comparison and action effort above. Rendering a decoded angle with rigid links imposes geometry on the drawing, so the measured replay error matters more than how plausible the ghost looks.'
				}
			],
			ui: [
				{
					title: 'Pause the world while considering an action',
					body: 'Planning takes time. The simulated mechanism remains still during that computation, then advances under the chosen torque. The visible alternatives are a small sample of evaluated plans. Their number and spacing should not be read as confidence or as the probability of an outcome.'
				}
			],
			lab: {
				file: 'world.zip',
				note: 'Generate experience, train a visual world model, and compare latent plans locally'
			}
		}
	]
};

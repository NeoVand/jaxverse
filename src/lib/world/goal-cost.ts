/** The actual controller scores embeddings, never the display readout. */
export function latentGoalCost(
	latents: Float32Array,
	goal: Float32Array,
	horizon: number,
	hold: boolean
): number {
	const window = hold ? Math.min(4, horizon) : 1;
	let sum = 0;
	for (let t = horizon - window; t < horizon; t++) {
		for (let d = 0; d < goal.length; d++) sum += (latents[t * goal.length + d] - goal[d]) ** 2;
	}
	return sum / (window * goal.length);
}

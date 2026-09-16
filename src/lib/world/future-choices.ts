import type { WorldPlan } from './engine';
import { latentGoalCost } from './goal-cost';

/** Rescore a fixed gallery. Neither inference, training, nor candidate search happens here. */
export function rescoreFutures(plan: WorldPlan, goalIndex: number, hold: boolean, effort = 0.01) {
	const goal = plan.comparisonGoals?.[goalIndex];
	if (!goal) return { plan, selectedIndex: 0 };
	const candidates = plan.candidates.map((candidate) => ({
		...candidate,
		cost:
			latentGoalCost(candidate.latents, goal, candidate.poses.length, hold) +
			effort *
				(candidate.actions.reduce((sum, value) => sum + value * value, 0) /
					candidate.actions.length)
	}));
	let selectedIndex = 0;
	for (let i = 1; i < candidates.length; i++)
		if (candidates[i].cost < candidates[selectedIndex].cost) selectedIndex = i;
	const chosen = candidates[selectedIndex];
	return {
		selectedIndex,
		plan: {
			...plan,
			cost: chosen.cost,
			poses: chosen.poses,
			actions: chosen.actions,
			action: [chosen.actions[0], chosen.actions[1]] as [number, number],
			candidates
		}
	};
}

/**
 * The five racers and their update rules, collapsed to the 2-D case. Steps
 * are written against a mutable per-runner state record so one function
 * serves every racer.
 *
 * Colors are design tokens, not hex: resolve them with `racerColors` at draw
 * time so canvas trails follow theme switches.
 */

export type OptimizerId = 'gd' | 'momentum' | 'adam' | 'adamw' | 'lion';

/** Token per trail — resolved against the live theme, never hardcoded.
 *  Chosen for maximum separation among the default three on both themes:
 *  grey / rose / ultramarine, with cyan and amber for the opt-ins. */
export const RACER_TOKENS: Record<OptimizerId, string> = {
	gd: '--ink-3',
	momentum: '--cat-3',
	adam: '--accent',
	adamw: '--cat-6',
	lion: '--cat-1'
};

/**
 * Per-optimizer stride scaling: one γ slider drives all five. Adaptive
 * methods (Adam/AdamW) and Lion take steps of roughly γ per axis whatever
 * the terrain, while SGD and momentum move as γ·|∇ℒ| — invisible on flat
 * floors at a γ the adaptive pair likes.
 *
 * At the default γ ≈ 0.05 these land on Gradient Lab's curated race rates
 * (Adam/AdamW 0.1, Lion 0.05), and the SGD/momentum pair is capped where a
 * simulated audit (default start + 120 random drops × 3 presets) shows zero
 * divergence — every racer reaches its basin instead of stalling or blowing
 * up. Momentum's steady-state push is γ·|∇ℒ|/(1−μ) ≈ 10γ, hence ×1.
 */
export const LR_MULT: Record<OptimizerId, number> = {
	gd: 2,
	momentum: 1,
	adam: 2,
	adamw: 2,
	lion: 1
};

/** Resolved CSS colors for every racer, read from the element's theme. */
export function racerColors(el: Element): Record<OptimizerId, string> {
	const style = getComputedStyle(el);
	const out = {} as Record<OptimizerId, string>;
	for (const id of Object.keys(RACER_TOKENS) as OptimizerId[]) {
		out[id] = style.getPropertyValue(RACER_TOKENS[id]).trim() || '#888';
	}
	return out;
}

export interface RunnerSpec {
	id: OptimizerId;
	label: string;
	token: string;
	blurb: string;
}

export const RUNNERS: RunnerSpec[] = [
	{
		id: 'gd',
		label: 'SGD',
		token: RACER_TOKENS.gd,
		blurb: 'θ ← θ − γ∇ℒ — the raw slope, nothing else'
	},
	{
		id: 'momentum',
		label: 'Momentum',
		token: RACER_TOKENS.momentum,
		blurb: 'Polyak heavy ball: v ← μv + ∇ℒ; θ ← θ − γv'
	},
	{
		id: 'adam',
		label: 'Adam',
		token: RACER_TOKENS.adam,
		blurb: 'bias-corrected moments, per-axis step scaling'
	},
	{
		id: 'adamw',
		label: 'AdamW',
		token: RACER_TOKENS.adamw,
		blurb: 'Adam + decoupled weight decay — the λθ pull is the “W”'
	},
	{
		id: 'lion',
		label: 'Lion',
		token: RACER_TOKENS.lion,
		blurb: 'sign of blended momentum: every step has magnitude γ per axis'
	}
];

// Constants match the reference implementations' defaults exactly.
export const EPS = 1e-8;
export const MU = 0.9; // momentum μ
export const BETA1 = 0.9;
export const BETA2 = 0.999;
export const LION_BETA1 = 0.9;
export const LION_BETA2 = 0.99;

export interface OptState {
	/** momentum / Lion buffer, or Adam's first moment m */
	vx: number;
	vy: number;
	/** Adam's second moment v */
	sx: number;
	sy: number;
	t: number;
}

export function initOptState(): OptState {
	return { vx: 0, vy: 0, sx: 0, sy: 0, t: 0 };
}

/**
 * One update: mutates `st`, returns the next position. `wd` (decoupled λ)
 * only acts in AdamW — everyone else ignores it.
 */
export function stepOptimizer(
	id: OptimizerId,
	x: number,
	y: number,
	gx: number,
	gy: number,
	st: OptState,
	lr: number,
	wd: number
): { x: number; y: number } {
	switch (id) {
		case 'gd':
			// [gd] θ ← θ − γ∇ℒ — the whole algorithm.
			st.t++;
			return { x: x - lr * gx, y: y - lr * gy };
		case 'momentum':
			// [momentum] v ← μv + g; θ ← θ − γv (γ NOT folded into v).
			st.vx = MU * st.vx + gx;
			st.vy = MU * st.vy + gy;
			st.t++;
			return { x: x - lr * st.vx, y: y - lr * st.vy };
		case 'adam':
		case 'adamw': {
			// [adam/adamw] moments + bias correction; AdamW adds the γλθ pull.
			const t = ++st.t;
			const mc1 = 1 - Math.pow(BETA1, t);
			const mc2 = 1 - Math.pow(BETA2, t);
			st.vx = BETA1 * st.vx + (1 - BETA1) * gx;
			st.vy = BETA1 * st.vy + (1 - BETA1) * gy;
			st.sx = BETA2 * st.sx + (1 - BETA2) * gx * gx;
			st.sy = BETA2 * st.sy + (1 - BETA2) * gy * gy;
			const decay = id === 'adamw' ? wd : 0;
			return {
				x: x - lr * (st.vx / mc1 / (Math.sqrt(st.sx / mc2) + EPS) + decay * x),
				y: y - lr * (st.vy / mc1 / (Math.sqrt(st.sy / mc2) + EPS) + decay * y)
			};
		}
		case 'lion': {
			// [lion] direction = sign(β₁m + (1−β₁)g); fixed-size γ steps;
			// the buffer updates with its own, slower decay β₂.
			const cx = LION_BETA1 * st.vx + (1 - LION_BETA1) * gx;
			const cy = LION_BETA1 * st.vy + (1 - LION_BETA1) * gy;
			const nx = x - lr * Math.sign(cx);
			const ny = y - lr * Math.sign(cy);
			st.vx = LION_BETA2 * st.vx + (1 - LION_BETA2) * gx;
			st.vy = LION_BETA2 * st.vy + (1 - LION_BETA2) * gy;
			st.t++;
			return { x: nx, y: ny };
		}
	}
}

import type { HoodChapter } from './types';

export const descent: HoodChapter = {
	slug: 'descent',
	blocks: [
		{
			id: 'gradient',
			lesson: 'Lesson 0a — the gradient, by hand first',
			lede: `A confession to open the course: <em>nothing in this chapter uses jax-js.</em> Two coordinates and a scalar loss need no tensor library, and seeing the algorithms as bare arithmetic once — every moment buffer a named variable — is the best preparation for meeting them again wrapped in <code>grad</code> and <code>jit</code>.`,
			ml: [
				{
					title: 'The landscape, and a gradient measured with a ruler',
					body: `The racers never see a formula for the terrain — they feel it, the way gradient descent always does. Here the feeling is <em>central differences</em>: nudge each coordinate by a hair, subtract, divide. Cheap and honest in two dimensions; hopeless in a million, which is the entire reason automatic differentiation exists and the reason the next chapter switches tools.`,
					code: {
						file: 'src/lib/optim/landscape.ts',
						code: `const H_DIFF = 1e-4;

/** ∇f at (x, y), measured, not derived. */
export function gradAt(preset: Preset, x: number, y: number): [number, number] {
	return [
		(preset.f(x + H_DIFF, y) - preset.f(x - H_DIFF, y)) / (2 * H_DIFF),
		(preset.f(x, y + H_DIFF) - preset.f(x, y - H_DIFF)) / (2 * H_DIFF)
	];
}`
					}
				},
				{
					title: 'Five optimizers, one switch statement',
					body: `Every optimizer in this book — and very nearly every optimizer in the field — is a few lines around <code>θ ← θ − γ·(something built from gradients)</code>. SGD uses the gradient raw. Momentum keeps a velocity. Adam keeps two exponential moments and divides one by the square root of the other, which is what makes its stride roughly γ per axis whatever the terrain. Read the Adam case slowly; you will meet it again as <code>optax.adam</code> in every later chapter, doing exactly this to a million parameters at once.`,
					code: {
						file: 'src/lib/optim/optimizers.ts',
						code: `case 'adam':
case 'adamw': {
	// moments + bias correction; AdamW adds the γλθ pull.
	const t = ++st.t;
	const mc1 = 1 - Math.pow(BETA1, t);
	const mc2 = 1 - Math.pow(BETA2, t);
	st.vx = BETA1 * st.vx + (1 - BETA1) * gx;   // first moment m
	st.vy = BETA1 * st.vy + (1 - BETA1) * gy;
	st.sx = BETA2 * st.sx + (1 - BETA2) * gx * gx;  // second moment v
	st.sy = BETA2 * st.sy + (1 - BETA2) * gy * gy;
	const decay = id === 'adamw' ? wd : 0;
	return {
		x: x - lr * (st.vx / mc1 / (Math.sqrt(st.sx / mc2) + EPS) + decay * x),
		y: y - lr * (st.vy / mc1 / (Math.sqrt(st.sy / mc2) + EPS) + decay * y)
	};
}`
					}
				}
			],
			ui: [
				{
					title: 'Contours from d3, terrain from a heatmap',
					body: `The landscape is rendered twice from one 110 × 110 grid of loss samples: a soft heat wash drawn straight into a canvas <code>ImageData</code>, and sixteen contour lines from <code>d3-contour</code> over the same values, both through <code>log(loss + 0.001)</code> so a valley floor and a high shelf get equal detail. The 3-D view is the same grid again, hand-projected — no WebGL, just painter's-order quads on the same canvas.`
				},
				{
					title: 'Trails that survive a theme switch',
					body: `Racer colors are design tokens (<code>--accent</code>, <code>--cat-3</code>…), resolved with <code>getComputedStyle</code> at draw time rather than baked in, so a dark-mode switch mid-race recolors every trail on the next frame. The animation is one <code>requestAnimationFrame</code> loop that steps all racers, then repaints — no reactive state in the hot path.`
				}
			]
		},
		{
			id: 'jax',
			lesson: 'Lesson 0b — where jax-js enters',
			ml: [
				{
					title: 'Where jax-js enters',
					body: `In the lab, the finite differences disappear. You write the loss as an ordinary function, and <code>valueAndGrad</code> hands back a new function that computes the loss <em>and</em> its exact derivative — no ruler, no <code>H_DIFF</code>, no error term. One jax-js habit to notice immediately: arrays are <em>moved</em> into the operations that consume them, and <code>.ref</code> is how you lend one out twice. It reads strange for a day and then becomes the reason a GPU buffer is never leaked. This is the idea the whole course compounds on: <em>a loss you can write is a loss you can descend.</em>`,
					code: {
						file: 'labs/descent/src/main.ts',
						lang: 'ts',
						code: `import { init, valueAndGrad, numpy as np } from '@jax-js/jax';

await init(); // cpu, wasm, or webgpu — whatever this machine has

// Rosenbrock's banana valley as an ordinary function…
const f = (x: np.Array, y: np.Array) =>
	np.sum(np.square(y.sub(np.square(x.ref))).mul(100).add(np.square(x.sub(1))));

// …and its exact value-and-gradient, for free.
const vg = valueAndGrad(f, { argnums: [0, 1] });
let x = np.array([-1.4]);
let y = np.array([1.6]);
for (let t = 0; t <= 2000; t++) {
	const [loss, [gx, gy]] = vg(x.ref, y.ref); // lend x and y, keep them
	x = x.sub(gx.mul(2e-3)); // θ ← θ − γ∇ℒ, same as ever
	y = y.sub(gy.mul(2e-3));
	if (t % 500 === 0) console.log(t, loss.dataSync()[0]);
	else loss.dispose();
}`
					}
				}
			],
			ui: [],
			lab: {
				file: 'descent.zip',
				note: 'The optimizer race as a terminal script: five update rules descending Rosenbrock, first by hand, then with jax-js grad'
			}
		}
	]
};

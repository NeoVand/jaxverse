import type { HoodChapter } from './types';

// The landing page's one block: the cover illustration, explained. It doubles
// as the reader's first encounter with the under-the-hood format — the quiet
// line they will find beneath every plate in the book, working exactly the
// same way here.

export const home: HoodChapter = {
	slug: 'home',
	blocks: [
		{
			id: 'hero',
			lesson: 'the cover, explained',
			lede: `The map above is not an illustration of training — it <em>is</em> training, the smallest honest kind. A hand-built loss landscape, three optimizers descending it live, sixty times a second. Every plate in this book carries a block like this one directly beneath it; open any of them and you get the code the plate just ran.`,
			ml: [
				{
					title: 'The landscape: four pits in a bowl',
					body: `The terrain is one scalar function — a broad bowl (so every rim drains inward and no walker ever leaves the map) minus four Gaussian pits, arranged with 180° rotational symmetry about the center. One pit is the global minimum; its twin across the origin is deliberately shallower, which is what makes the walkers' choice interesting. A faint sinusoidal texture keeps the gradient from ever being perfectly smooth.`,
					code: {
						file: 'src/lib/components/ui/HeroField.svelte',
						code: `const PITS = [
	{ cx: 0.71, cy: 0.62, d: 1.0, s: 0.18 }, // the global minimum
	{ cx: 0.29, cy: 0.38, d: 0.55, s: 0.16 }, // its tempting, weaker twin
	{ cx: 0.34, cy: 0.74, d: 0.28, s: 0.09 }, // a shallow trap…
	{ cx: 0.66, cy: 0.26, d: 0.28, s: 0.09 } // …and its mirror
];
function f(x: number, y: number): number {
	// a broad centered bowl so every rim drains inward
	let v = 0.55 * ((x - 0.5) * (x - 0.5) + 0.8 * (y - 0.5) * (y - 0.5));
	for (const p of PITS) {
		const dx = x - p.cx;
		const dy = y - p.cy;
		v -= p.d * Math.exp(-(dx * dx + dy * dy) / (2 * p.s * p.s));
	}
	return v + 0.018 * Math.sin(9 * x + 3) * Math.sin(7 * y + 1);
}`
					}
				},
				{
					title: 'Three walkers, three update rules',
					body: `Each walker feels the slope by central differences, plus a dose of noise standing in for the minibatch sampling every real training run lives with — and then the three part ways. SGD takes the noisy gradient at face value. Momentum keeps a velocity, so it can coast through the weak twin pit that stops SGD cold. Adam keeps two exponential moments and divides one by the square root of the other. They spawn together just uphill of the weaker pit, precisely so the rules genuinely disagree about where to end up — chapter 0's whole argument, playing on the cover.`,
					code: {
						file: 'src/lib/components/ui/HeroField.svelte',
						code: `const [tgx, tgy] = gradAt(w.x, w.y); // central differences — no autodiff yet
const gx = tgx + (Math.random() * 2 - 1) * 0.25; // minibatch-style noise
const gy = tgy + (Math.random() * 2 - 1) * 0.25;
if (w.rule === 'sgd') {
	w.x -= 0.0035 * gx;
	w.y -= 0.0035 * gy;
} else if (w.rule === 'momentum') {
	w.vx = 0.93 * w.vx + gx;
	w.vy = 0.93 * w.vy + gy;
	w.x -= 0.0012 * w.vx;
	w.y -= 0.0012 * w.vy;
} else {
	// adam: two moments, bias-corrected, one divided by the other
	w.mx = b1 * w.mx + (1 - b1) * gx;
	w.sx = b2 * w.sx + (1 - b2) * gx * gx;
	const c1 = 1 - Math.pow(b1, w.t);
	const c2 = 1 - Math.pow(b2, w.t);
	w.x -= (0.0022 * (w.mx / c1)) / (Math.sqrt(w.sx / c2) + eps); // …and y likewise
}`
					}
				}
			],
			ui: [
				{
					title: 'Contours that never touch an edge',
					body: `The floating look is a filter, not a border. The function is sampled on a 108 × 66 grid, <code>d3-contour</code> traces eighteen level sets over it, and any ring that comes within a grid cell of the domain boundary is simply dropped — every line drawn is a closed loop, so the map has no clipped ends and no seams. The interior gradient is a per-cell alpha field <em>anchored to the outermost surviving ring</em>: exactly zero wherever the loss sits above that ring's threshold, deepening toward the minima inside it, painted once into a tiny offscreen canvas and stretched with smoothing.`,
					code: {
						file: 'src/lib/components/ui/HeroField.svelte',
						code: `const EDGE = 1.2; // grid cells: anything this close to the border is a cut ring
const touchesEdge = (ring: number[][]) =>
	ring.some(([x, y]) => x < EDGE || y < EDGE || x > GX - 1 - EDGE || y > GY - 1 - EDGE);
const levels = contours()
	.size([GX, GY])
	.thresholds(thresholds)(values)
	.map((c, k) => ({
		k,
		rings: c.coordinates
			.map((poly) => poly.filter((ring) => !touchesEdge(ring)))
			.filter((poly) => poly.length > 0)
	}))
	.filter((c) => c.rings.length > 0);
// the wash stops exactly at the outermost drawn contour: alpha is zero
// wherever the loss sits above that ring's threshold
const tOut = thresholds[Math.max(...levels.map((c) => c.k))];`
					}
				},
				{
					title: 'Colors from the theme, motion from one loop',
					body: `Walker colors are design tokens (<code>--ink-3</code>, <code>--accent</code>, <code>--warm</code>) read with <code>getComputedStyle</code> at draw time, so switching themes recolors the map on the next frame — and the same accent wash reads as basins getting lighter on a dark page, gently darker on a light one. Everything moves in a single <code>requestAnimationFrame</code> loop; every 270 frames the walkers respawn somewhere new, and <code>prefers-reduced-motion</code> collapses the whole thing to one settled frame.`
				}
			]
		}
	]
};

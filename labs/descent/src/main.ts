// Lab 0 — the descent, twice over.
//
// Part one is the prologue's arithmetic laid bare: the Rosenbrock valley, its
// gradient measured with central differences, and three update rules written
// on plain numbers. Part two is the same descent with jax-js doing the
// calculus: valueAndGrad hands back the loss AND its exact derivative.

import { init, valueAndGrad, numpy as np } from '@jax-js/jax';

const out = document.getElementById('out') as HTMLPreElement;
const lines: string[] = [];
function log(s: string) {
	lines.push(s);
	out.textContent = lines.join('\n');
}

// ── the landscape ────────────────────────────────────────────────────────────
// Rosenbrock's banana valley: minimum at (1, 1), floor almost flat.
const f = (x: number, y: number) => 100 * (y - x * x) ** 2 + (1 - x) ** 2;

// The gradient, measured with a ruler — fine in 2-D, hopeless in a million.
const H = 1e-4;
const gradAt = (x: number, y: number): [number, number] => [
	(f(x + H, y) - f(x - H, y)) / (2 * H),
	(f(x, y + H) - f(x, y - H)) / (2 * H)
];

// ── part one: three update rules on plain numbers ───────────────────────────
interface Walker {
	name: string;
	x: number;
	y: number;
	vx: number;
	vy: number;
	sx: number;
	sy: number;
	t: number;
}
const walker = (name: string): Walker => ({
	name,
	x: -1.4,
	y: 1.6,
	vx: 0,
	vy: 0,
	sx: 0,
	sy: 0,
	t: 0
});

function step(w: Walker, lr: number) {
	const [gx, gy] = gradAt(w.x, w.y);
	w.t++;
	if (w.name === 'sgd') {
		w.x -= lr * gx;
		w.y -= lr * gy;
	} else if (w.name === 'momentum') {
		w.vx = 0.9 * w.vx + gx;
		w.vy = 0.9 * w.vy + gy;
		w.x -= lr * w.vx;
		w.y -= lr * w.vy;
	} else {
		// adam: two moments, bias-corrected, per-axis stride
		const [b1, b2, eps] = [0.9, 0.999, 1e-8];
		w.vx = b1 * w.vx + (1 - b1) * gx;
		w.vy = b1 * w.vy + (1 - b1) * gy;
		w.sx = b2 * w.sx + (1 - b2) * gx * gx;
		w.sy = b2 * w.sy + (1 - b2) * gy * gy;
		const [c1, c2] = [1 - b1 ** w.t, 1 - b2 ** w.t];
		w.x -= (lr * (w.vx / c1)) / (Math.sqrt(w.sx / c2) + eps);
		w.y -= (lr * (w.vy / c1)) / (Math.sqrt(w.sy / c2) + eps);
	}
}

async function main() {
	log('── part one: by hand, gradient by central differences ──');
	const walkers = [walker('sgd'), walker('momentum'), walker('adam')];
	const lrOf: Record<string, number> = { sgd: 2e-3, momentum: 4e-4, adam: 2e-2 };
	for (let t = 1; t <= 3000; t++) {
		for (const w of walkers) step(w, lrOf[w.name]);
		if (t % 600 === 0)
			log(
				`t=${String(t).padStart(4)}  ` +
					walkers.map((w) => `${w.name} ${f(w.x, w.y).toFixed(4).padStart(9)}`).join('   ')
			);
	}
	for (const w of walkers)
		log(`  ${w.name.padEnd(8)} ended at (${w.x.toFixed(3)}, ${w.y.toFixed(3)}) — target (1, 1)`);

	// ── part two: jax-js computes the exact gradient ──────────────────────────
	log('');
	log('── part two: the same valley, gradient from jax-js ──');
	const devices = await init();
	log(`jax-js devices: ${devices.join(', ')}`);

	// The loss as an ordinary function of two arrays. Note .ref: jax-js arrays
	// are MOVED into the ops that consume them; .ref lends one out twice.
	const loss = (x: np.Array, y: np.Array) =>
		np.sum(
			np
				.square(y.sub(np.square(x.ref)))
				.mul(100)
				.add(np.square(x.sub(1)))
		);

	const vg = valueAndGrad(loss, { argnums: [0, 1] });
	let x = np.array([-1.4]);
	let y = np.array([1.6]);
	for (let t = 0; t <= 3000; t++) {
		const [l, [gx, gy]] = vg(x.ref, y.ref);
		x = x.sub(gx.mul(2e-3));
		y = y.sub(gy.mul(2e-3));
		if (t % 600 === 0) log(`t=${String(t).padStart(4)}  loss ${l.dataSync()[0].toFixed(6)}`);
		else l.dispose();
		if (t % 100 === 0) await new Promise((r) => setTimeout(r)); // let the page breathe
	}
	log(`  jax-js sgd ended at (${x.dataSync()[0].toFixed(3)}, ${y.dataSync()[0].toFixed(3)})`);
	log('');
	log('Change the learning rates at the top of main() and re-save — vite reloads live.');
}

void main().catch((e) => log(`error: ${e?.message ?? e}`));

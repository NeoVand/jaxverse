<script lang="ts">
	// The hero's living mark, grown up: a two-dimensional loss landscape drawn
	// as a topographic map, with three walkers descending it live — plain SGD,
	// momentum, and Adam, the same three rules the prologue teaches. They spawn
	// together, race to a minimum, settle, and start again somewhere new.
	// Clicking (or tapping) the map drops them at the pointer. Honors
	// prefers-reduced-motion by drawing a single settled frame.
	import { contours } from 'd3-contour';

	let { height = 300 }: { height?: number } = $props();

	let canvas: HTMLCanvasElement;

	// ── the landscape: four pits with 180° rotational symmetry about the
	// center, so the composition balances instead of leaning right — the
	// global minimum lower-right, its near-twin upper-left, and a mirrored
	// pair of shallow traps between them ──────────────────────────────────────
	const PITS = [
		{ cx: 0.71, cy: 0.62, d: 1.0, s: 0.18 }, // the global minimum
		{ cx: 0.29, cy: 0.38, d: 0.55, s: 0.16 }, // its tempting, weaker twin
		{ cx: 0.34, cy: 0.74, d: 0.28, s: 0.09 }, // a shallow trap…
		{ cx: 0.66, cy: 0.26, d: 0.28, s: 0.09 } // …and its mirror
	];
	function f(x: number, y: number): number {
		// a broad centered bowl so every rim drains inward — no walker ever
		// leaves the map
		let v = 0.55 * ((x - 0.5) * (x - 0.5) + 0.8 * (y - 0.5) * (y - 0.5));
		for (const p of PITS) {
			const dx = x - p.cx;
			const dy = y - p.cy;
			v -= p.d * Math.exp(-(dx * dx + dy * dy) / (2 * p.s * p.s));
		}
		v += 0.018 * Math.sin(9 * x + 3) * Math.sin(7 * y + 1); // faint texture
		return v;
	}
	function gradAt(x: number, y: number): [number, number] {
		const h = 1e-3;
		return [(f(x + h, y) - f(x - h, y)) / (2 * h), (f(x, y + h) - f(x, y - h)) / (2 * h)];
	}

	// ── the walkers: one per update rule, in the book's own colors ────────────
	interface Walker {
		rule: 'sgd' | 'momentum' | 'adam';
		x: number;
		y: number;
		vx: number;
		vy: number;
		mx: number;
		my: number;
		sx: number;
		sy: number;
		t: number;
		trail: Array<{ x: number; y: number; a: number }>;
	}
	const mkWalker = (rule: Walker['rule'], x: number, y: number): Walker => ({
		rule,
		x,
		y,
		vx: 0,
		vy: 0,
		mx: 0,
		my: 0,
		sx: 0,
		sy: 0,
		t: 0,
		trail: []
	});

	let walkers: Walker[] = [];
	let frame = 0; // resets on every spawn — drives the periodic restart
	function spawnAll(x: number, y: number) {
		frame = 0;
		walkers = [mkWalker('sgd', x, y), mkWalker('momentum', x, y), mkWalker('adam', x, y)];
	}
	function spawnRandom() {
		// start just uphill of the weaker twin, so the rules genuinely
		// disagree: SGD sinks into the nearest pit while momentum and Adam
		// carry through it toward the global minimum — the chapter's whole
		// argument in miniature, playing on the landing page. The band
		// alternates with its point mirror, matching the symmetric terrain.
		const x = 0.18 + Math.random() * 0.24;
		const y = 0.04 + Math.random() * 0.14;
		if (Math.random() < 0.5) spawnAll(x, y);
		else spawnAll(1 - x, 1 - y);
	}

	function stepWalker(w: Walker) {
		// minibatch-style noise, so each rule shows its personality: SGD
		// jitters with it, momentum averages it away, Adam normalizes it —
		// and near a watershed the three can genuinely part ways
		const [tgx, tgy] = gradAt(w.x, w.y);
		const gx = tgx + (Math.random() * 2 - 1) * 0.25;
		const gy = tgy + (Math.random() * 2 - 1) * 0.25;
		w.t++;
		if (w.rule === 'sgd') {
			w.x -= 0.0035 * gx;
			w.y -= 0.0035 * gy;
		} else if (w.rule === 'momentum') {
			w.vx = 0.93 * w.vx + gx;
			w.vy = 0.93 * w.vy + gy;
			w.x -= 0.0012 * w.vx;
			w.y -= 0.0012 * w.vy;
		} else {
			const [b1, b2, eps] = [0.9, 0.995, 1e-8];
			w.mx = b1 * w.mx + (1 - b1) * gx;
			w.my = b1 * w.my + (1 - b1) * gy;
			w.sx = b2 * w.sx + (1 - b2) * gx * gx;
			w.sy = b2 * w.sy + (1 - b2) * gy * gy;
			const c1 = 1 - Math.pow(b1, w.t);
			const c2 = 1 - Math.pow(b2, w.t);
			w.x -= (0.0022 * (w.mx / c1)) / (Math.sqrt(w.sx / c2) + eps);
			w.y -= (0.0022 * (w.my / c1)) / (Math.sqrt(w.sy / c2) + eps);
		}
		w.x = Math.min(0.985, Math.max(0.015, w.x));
		w.y = Math.min(0.985, Math.max(0.015, w.y));
		w.trail.push({ x: w.x, y: w.y, a: 1 });
		if (w.trail.length > 110) w.trail.shift();
		for (const p of w.trail) p.a *= 0.985;
	}

	// ── contour geometry, computed once (the landscape never changes). Rings
	// that touch the domain boundary are dropped: every line drawn is a closed
	// loop floating in space, so the map has no clipped ends and no seams.
	// Alongside the rings, a per-cell wash field drives the interior gradient:
	// it is *anchored to the outermost surviving contour* — zero everywhere
	// outside that ring, deepening toward the minima inside it — so no tint
	// ever leaks past the last drawn line. ────────────────────────────────────
	const GX = 108;
	const GY = 66;
	const { contourPaths, washField } = (() => {
		const values = new Float64Array(GX * GY);
		let lo = Infinity;
		let hi = -Infinity;
		for (let j = 0; j < GY; j++)
			for (let i = 0; i < GX; i++) {
				const v = f(i / (GX - 1), j / (GY - 1));
				values[j * GX + i] = v;
				lo = Math.min(lo, v);
				hi = Math.max(hi, v);
			}
		const n = 18;
		const thresholds = Array.from({ length: n }, (_, k) => lo + ((k + 0.6) / n) * (hi - lo));
		const EDGE = 1.2; // grid cells: anything this close to the border is a cut ring
		const touchesEdge = (ring: number[][]) =>
			ring.some(([x, y]) => x < EDGE || y < EDGE || x > GX - 1 - EDGE || y > GY - 1 - EDGE);
		const levels = contours()
			.size([GX, GY])
			.thresholds(thresholds)(values as unknown as number[])
			.map((c, k) => ({
				k,
				// 1 at the lowest-loss ring, 0 at the rim — drives the line ramp
				depth: 1 - k / (n - 1),
				rings: c.coordinates
					.map((poly) => poly.filter((ring) => !touchesEdge(ring)))
					.filter((poly) => poly.length > 0)
			}))
			.filter((c) => c.rings.length > 0);
		// the wash stops exactly at the outermost drawn contour: alpha is zero
		// wherever the loss sits above that ring's threshold
		const tOut = thresholds[Math.max(...levels.map((c) => c.k))];
		const field = new Float32Array(GX * GY);
		const FADE = 6; // belt-and-suspenders falloff at the canvas border
		for (let j = 0; j < GY; j++)
			for (let i = 0; i < GX; i++) {
				const idx = j * GX + i;
				const u = Math.max(0, (tOut - values[idx]) / (tOut - lo));
				const e = Math.min(1, Math.min(i, GX - 1 - i, j, GY - 1 - j) / FADE);
				field[idx] = Math.pow(u, 1.9) * (e * e * (3 - 2 * e));
			}
		return { contourPaths: levels, washField: field };
	})();

	// ── the interior wash: the basins filled with a soft accent gradient — the
	// rim is exactly the page background (alpha 0), and the fill deepens toward
	// the minima. Painted once per theme into a tiny offscreen canvas, then
	// stretched with smoothing so it reads as a continuous gradient. ──────────
	let washCanvas: HTMLCanvasElement | null = null;
	let washKey = '';
	function washFor(accentCss: string, ctx: CanvasRenderingContext2D): HTMLCanvasElement {
		if (washCanvas && washKey === accentCss) return washCanvas;
		// let the 2d context normalize whatever the token holds to #rrggbb
		ctx.save();
		ctx.fillStyle = accentCss;
		const norm = String(ctx.fillStyle);
		ctx.restore();
		const m = /^#([0-9a-f]{6})$/i.exec(norm);
		const [r, g, b] = m
			? [0, 2, 4].map((o) => parseInt(m[1].slice(o, o + 2), 16))
			: (norm
					.match(/[\d.]+/g)
					?.slice(0, 3)
					.map(Number) ?? [43, 69, 216]);
		const off = document.createElement('canvas');
		off.width = GX;
		off.height = GY;
		const g2 = off.getContext('2d')!;
		const img = g2.createImageData(GX, GY);
		for (let idx = 0; idx < washField.length; idx++) {
			img.data[idx * 4] = r;
			img.data[idx * 4 + 1] = g;
			img.data[idx * 4 + 2] = b;
			img.data[idx * 4 + 3] = Math.round(washField[idx] * 0.4 * 255);
		}
		g2.putImageData(img, 0, 0);
		washCanvas = off;
		washKey = accentCss;
		return off;
	}

	$effect(() => {
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
		let raf = 0;
		let disposed = false;

		if (reduced) {
			// one settled frame: the walkers rest in the global minimum
			walkers = PITS.slice(0, 1).flatMap((p) => [
				mkWalker('sgd', p.cx - 0.02, p.cy + 0.01),
				mkWalker('momentum', p.cx + 0.02, p.cy - 0.01),
				mkWalker('adam', p.cx, p.cy)
			]);
		} else {
			spawnRandom();
		}

		function draw() {
			if (disposed || !ctx) return;
			const dpr = Math.min(devicePixelRatio || 1, 2);
			const W = canvas.clientWidth;
			const H = canvas.clientHeight;
			if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
				canvas.width = W * dpr;
				canvas.height = H * dpr;
			}
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx.clearRect(0, 0, W, H);

			const style = getComputedStyle(canvas);
			const ink3 = style.getPropertyValue('--ink-3').trim() || '#999';
			const accent = style.getPropertyValue('--accent').trim() || '#2b45d8';
			const warm = style.getPropertyValue('--warm').trim() || '#d3541f';
			const colorOf = { sgd: ink3, momentum: accent, adam: warm } as const;

			const px = (u: number) => u * W;
			const py = (u: number) => u * H;

			// the interior wash first: over a dark page the accent fill reads as
			// the basins getting lighter toward low loss; over a light page, as
			// getting gently darker — the same paint, both directions honest
			ctx.imageSmoothingEnabled = true;
			ctx.imageSmoothingQuality = 'high';
			ctx.drawImage(washFor(accent, ctx), 0, 0, GX, GY, 0, 0, W, H);

			// then the rings, kept thin — quiet hairlines over the gradient
			for (const c of contourPaths) {
				ctx.beginPath();
				for (const poly of c.rings)
					for (const ring of poly) {
						for (let i = 0; i < ring.length; i++) {
							const X = (ring[i][0] / (GX - 1)) * W;
							const Y = (ring[i][1] / (GY - 1)) * H;
							if (i === 0) ctx.moveTo(X, Y);
							else ctx.lineTo(X, Y);
						}
						ctx.closePath();
					}
				const d = Math.pow(c.depth, 1.4);
				ctx.strokeStyle = accent;
				ctx.globalAlpha = 0.05 + 0.32 * d;
				ctx.lineWidth = 0.6;
				ctx.stroke();
			}
			ctx.globalAlpha = 1;

			// physics + trails; every ~4.5 s the walkers set off from somewhere
			// new, so the map never sits converged and still
			if (!reduced) {
				for (const w of walkers) stepWalker(w);
				frame++;
				if (frame > 270) {
					frame = 0;
					spawnRandom();
				}
			}

			for (const w of walkers) {
				ctx.strokeStyle = colorOf[w.rule];
				ctx.lineWidth = 1.4;
				for (let i = 1; i < w.trail.length; i++) {
					const a = w.trail[i - 1];
					const b = w.trail[i];
					ctx.globalAlpha = b.a * 0.7;
					ctx.beginPath();
					ctx.moveTo(px(a.x), py(a.y));
					ctx.lineTo(px(b.x), py(b.y));
					ctx.stroke();
				}
				ctx.globalAlpha = 1;
				// concentric radii, so when all three agree on a minimum they
				// stack into a bullseye instead of hiding one another
				const radius = { sgd: 6.4, momentum: 5, adam: 3.4 }[w.rule];
				ctx.beginPath();
				ctx.arc(px(w.x), py(w.y), radius, 0, Math.PI * 2);
				ctx.fillStyle = colorOf[w.rule];
				ctx.fill();
			}

			if (!reduced) raf = requestAnimationFrame(draw);
		}

		draw();
		return () => {
			disposed = true;
			cancelAnimationFrame(raf);
		};
	});

	function onPointerDown(e: PointerEvent) {
		const r = canvas.getBoundingClientRect();
		spawnAll(
			Math.min(0.98, Math.max(0.02, (e.clientX - r.left) / r.width)),
			Math.min(0.98, Math.max(0.02, (e.clientY - r.top) / r.height))
		);
	}
</script>

<canvas
	bind:this={canvas}
	class="block w-full cursor-crosshair"
	style="height: {height}px;"
	onpointerdown={onPointerDown}
	aria-label="A topographic loss landscape with three optimizers descending it. Click to drop them somewhere new."
></canvas>

<script lang="ts">
	// The chart. A tabular softmax policy learns to sail by REINFORCE, hundreds
	// of passages a second, all on the main thread — and because there is a wind
	// and a boat that cannot point into it, the route it discovers is one nobody
	// wrote down.
	//
	// Four layers on one canvas: the water, the value estimate as a diverging
	// wash under smooth soundings, the policy drawn as one current rose per
	// cell, and the latest passage replayed as a wake. The compass rose is a
	// control: drag it and the whole field reorganizes under a learner that is
	// still running.
	//
	// One rule governs the drawing, learned the hard way: every quantity gets a
	// primitive of its own. The first version drew the policy, the value, the
	// wind and the graticule all as thin line segments of similar weight, and a
	// thousand of them at once is not a chart, it is static.
	import { onDestroy } from 'svelte';
	import { contours } from 'd3-contour';
	import { Pause, Pencil, Play, RotateCcw, StepForward } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { progress } from '$lib/data/progress.svelte';
	import { hexRgb, readTokens, themePulse, watchTheme, type Tokens } from '$lib/viz/tokens.svelte';
	import { sparkPath } from '$lib/viz/spark';
	import {
		bestPassage,
		cellIndex,
		createBaseline,
		createTheta,
		headingAngle,
		makeSea,
		N_HEADINGS,
		policyAt,
		runPassage,
		trainStep,
		waterCells
	} from '$lib/optim-rl/chart';
	import { mulberry32 } from '$lib/optim-rl/rng';

	const SEED = 2049;
	/** Practice passages per second while playing. */
	const EPS = 90;
	const KEEP = 320;

	// ── the learner: plain fields, mutated in place ──
	const sea = makeSea();
	let theta = createTheta(sea);
	let baseline = createBaseline(sea);
	let rand = mulberry32(SEED);
	let water = waterCells(sea);

	// ── reactive state ──
	let ready = $state(false);
	let playing = $state(false);
	let lr = $state(0.09);
	let showPolicy = $state(true);
	let showValue = $state(true);
	let editing = $state(false);
	let episodes = $state(0);
	let returns = $state<number[]>([]);
	let arrived = $state<number[]>([]);
	let best = $state(bestPassage(sea).reward);
	let windDeg = $state(0);
	let canvas: HTMLCanvasElement | undefined = $state();
	let host: HTMLElement | undefined = $state();

	watchTheme();

	const arrivalRate = $derived(
		arrived.length ? arrived.reduce((a, b) => a + b, 0) / arrived.length : 0
	);
	const meanReturn = $derived(
		returns.length ? returns.reduce((a, b) => a + b, 0) / returns.length : 0
	);

	// ── the passage currently being replayed on the chart ──
	let show: { path: number[]; t: number; ms: number } | null = null;
	let sinceShow = 0;

	let raf = 0;
	let last = 0;
	let reduced = false;

	// The arithmetic is not the cost here and never was: a thousand practice
	// passages take 60 ms and a thousand canvas strokes take 0.1 ms, both
	// measured. What is worth avoiding is the layout: a getBoundingClientRect
	// and a getComputedStyle every frame, interleaved with reactive text
	// updates, forces a full synchronous layout of a very long document sixty
	// times a second for numbers that change almost never. So the size comes
	// from a ResizeObserver, the colours are read once per theme, and the
	// counters reach the DOM ten times a second rather than ninety.
	let cssW = 0;
	let cssH = 0;
	let tk: Tokens | null = null;
	let ro: ResizeObserver | undefined;
	/** Counters accumulate here and are flushed to $state on a timer. */
	let pending = 0;
	const buf: number[] = [];
	const bufArr: number[] = [];
	let sinceFlush = 0;

	function boot() {
		if (ready || !canvas) return;
		ready = true;
		reduced =
			typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
		const cv = canvas;
		ro = new ResizeObserver(() => {
			const r = cv.getBoundingClientRect();
			cssW = r.width;
			cssH = r.height;
		});
		ro.observe(cv);
		const r0 = cv.getBoundingClientRect();
		cssW = r0.width;
		cssH = r0.height;
		last = performance.now();
		raf = requestAnimationFrame(frame);
	}

	function learn(n: number) {
		for (let i = 0; i < n; i++) {
			const p = trainStep(sea, theta, baseline, rand, lr, water);
			pending += 1;
			// only passages from the home mooring go on the scoreboard; the
			// scattered practice ones are training, not results
			if (p.states.length && p.path[0] === sea.start) {
				buf.push(p.totalReward);
				bufArr.push(p.end === 'harbour' ? 1 : 0);
			}
		}
	}

	/** Push the accumulated counters into reactive state, ten times a second. */
	function flush() {
		if (!pending) return;
		episodes += pending;
		pending = 0;
		if (buf.length) {
			returns = [...returns, ...buf].slice(-KEEP);
			arrived = [...arrived, ...bufArr].slice(-KEEP);
			buf.length = 0;
			bufArr.length = 0;
		}
	}

	function frame(now: number) {
		const dt = Math.min(0.1, (now - last) / 1000);
		last = now;
		if (playing) {
			learn(Math.max(1, Math.round(EPS * dt)));
			sinceShow += dt;
			if (!reduced && (show === null || show.t >= show.ms) && sinceShow > 0.35) {
				sinceShow = 0;
				const p = runPassage(sea, theta, rand, sea.start);
				show = { path: p.path, t: 0, ms: Math.min(2200, 90 * p.path.length) };
			}
			if (show) show.t += dt * 1000;
			sinceFlush += dt;
			if (sinceFlush > 0.1) {
				sinceFlush = 0;
				flush();
				if (arrivalRate > 0.85 && episodes > 3000) progress.reach('reward:sailed');
			}
		}
		draw();
		raf = requestAnimationFrame(frame);
	}

	// onDestroy also runs during SSR, where there is no animation frame to cancel
	onDestroy(() => {
		if (raf) cancelAnimationFrame(raf);
		ro?.disconnect();
	});

	// Colours are resolved once per theme rather than once per frame.
	$effect(() => {
		void themePulse.tick;
		if (host) tk = readTokens(host);
	});

	function resetTheta() {
		theta = createTheta(sea);
		baseline = createBaseline(sea);
		rand = mulberry32(SEED);
		episodes = 0;
		returns = [];
		arrived = [];
		pending = 0;
		buf.length = 0;
		bufArr.length = 0;
		show = null;
	}

	/** Anything that changes the water invalidates the optimum and the replay. */
	function seaChanged() {
		water = waterCells(sea);
		best = bestPassage(sea).reward;
		show = null;
	}

	$effect(() => {
		sea.windFrom = (windDeg * Math.PI) / 180;
		seaChanged();
	});

	// ── the current rose, precomputed ────────────────────────────────────────
	//
	// A policy is a distribution over eight directions, and the honest way to
	// draw one is as a smooth angular density rather than as eight radii joined
	// by straight lines. Eight vertices is far too few: the polygon comes out
	// faceted, and where one heading dominates its neighbours the corners fold
	// into shapes that look like mistakes because they are — artefacts of the
	// sampling, not of the policy.
	//
	// So the radius is a sum of overlapping angular bumps, one per heading:
	//
	//     r(θ) = r₀ + R · Σ_a  p_a · exp(−Δ(θ,θ_a)² / 2σ²)
	//
	// which is smooth everywhere by construction, cannot self-intersect, and
	// gives a circle for a policy with no opinion and a single clean petal for
	// one that has made up its mind. The kernel weights depend only on the
	// angles, never on the policy, so the whole 96 × 8 matrix is computed once.
	const ROSE_N = 96;
	/** Kernel width. π/4 apart, so this overlaps neighbours enough to stay round. */
	const ROSE_SIGMA = 0.42;
	const ROSE_SIN = new Float64Array(ROSE_N);
	const ROSE_COS = new Float64Array(ROSE_N);
	const ROSE_K = new Float64Array(ROSE_N * N_HEADINGS);
	{
		for (let j = 0; j < ROSE_N; j++) {
			const th = (j / ROSE_N) * Math.PI * 2;
			ROSE_SIN[j] = Math.sin(th);
			ROSE_COS[j] = Math.cos(th);
			for (let a = 0; a < N_HEADINGS; a++) {
				let d = Math.abs(th - headingAngle(a)) % (Math.PI * 2);
				if (d > Math.PI) d = Math.PI * 2 - d;
				ROSE_K[j * N_HEADINGS + a] = Math.exp(-(d * d) / (2 * ROSE_SIGMA * ROSE_SIGMA));
			}
		}
		// normalize so a uniform policy draws a circle of radius r₀ + R/8·(peak)
		let peak = 0;
		for (let a = 0; a < N_HEADINGS; a++) peak += ROSE_K[a];
		for (let i = 0; i < ROSE_K.length; i++) ROSE_K[i] /= peak;
	}

	// ── geometry ──
	function geom(w: number, h: number) {
		const cell = Math.min(w / sea.w, h / sea.h);
		return { cell, ox: (w - cell * sea.w) / 2, oy: (h - cell * sea.h) / 2 };
	}

	function draw() {
		const cv = canvas;
		if (!cv || !host || !tk || cssW < 1) return;
		const ctx = cv.getContext('2d');
		if (!ctx) return;
		const dpr = Math.min(2, window.devicePixelRatio || 1);
		if (cv.width !== Math.round(cssW * dpr)) {
			cv.width = Math.round(cssW * dpr);
			cv.height = Math.round(cssH * dpr);
		}
		const W = cssW;
		const H = cssH;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		const { cell, ox, oy } = geom(W, H);
		// y grows upward on a chart, downward on a canvas
		const px = (c: number) => ox + ((c % sea.w) + 0.5) * cell;
		const py = (c: number) => oy + (sea.h - 1 - ((c / sea.w) | 0) + 0.5) * cell;

		ctx.clearRect(0, 0, W, H);
		ctx.fillStyle = tk.band;
		ctx.fillRect(0, 0, W, H);

		// ── the value estimate, as a soft wash ──
		// Contours were the wrong instrument: the field is 14 × 10, so rings
		// drawn through it trace interpolation artefacts rather than structure,
		// and a chart full of tangled hairlines fights the policy for the same
		// visual channel. Cell data gets drawn as cells.
		if (showValue) drawValue(ctx, cell, ox, oy, tk);

		// ── the graticule: present, and barely ──
		ctx.strokeStyle = tk.lineSoft;
		ctx.lineWidth = 0.5;
		ctx.globalAlpha = 0.45;
		ctx.beginPath();
		for (let x = 0; x <= sea.w; x++) {
			ctx.moveTo(ox + x * cell, oy);
			ctx.lineTo(ox + x * cell, oy + sea.h * cell);
		}
		for (let y = 0; y <= sea.h; y++) {
			ctx.moveTo(ox, oy + y * cell);
			ctx.lineTo(ox + sea.w * cell, oy + y * cell);
		}
		ctx.stroke();
		ctx.globalAlpha = 1;

		// ── land, with a coastline ──
		for (const c of sea.land) {
			ctx.fillStyle = tk.surface2;
			ctx.fillRect(px(c) - cell / 2, py(c) - cell / 2, cell, cell);
			ctx.strokeStyle = tk.ink3;
			ctx.lineWidth = 1;
			ctx.globalAlpha = 0.8;
			ctx.strokeRect(px(c) - cell / 2 + 0.5, py(c) - cell / 2 + 0.5, cell - 1, cell - 1);
			ctx.globalAlpha = 1;
		}

		// ── shoals: the chart symbol, three dots, not a thicket of ticks ──
		ctx.fillStyle = tk.warm;
		for (const c of sea.shoals) {
			for (let k = 0; k < 3; k++) {
				const a = -Math.PI / 2 + (k * Math.PI * 2) / 3;
				ctx.beginPath();
				ctx.arc(
					px(c) + Math.cos(a) * cell * 0.17,
					py(c) + Math.sin(a) * cell * 0.17,
					Math.max(1.3, cell * 0.055),
					0,
					Math.PI * 2
				);
				ctx.fill();
			}
		}

		// ── the policy, as one current rose per cell ──
		if (showPolicy) {
			const probs = new Float64Array(N_HEADINGS);
			const r0 = cell * 0.08;
			const R = cell * 0.4;
			for (let c = 0; c < sea.w * sea.h; c++) {
				if (sea.land.has(c) || sea.shoals.has(c) || c === sea.harbour) continue;
				policyAt(theta, c, probs);
				let mx = 0;
				for (const p of probs) if (p > mx) mx = p;
				ctx.beginPath();
				for (let j = 0; j < ROSE_N; j++) {
					// r(θ) = r0 + R · Σ_a p_a · K(θ − θ_a): a smooth angular density
					// rather than eight radii joined by straight lines
					let acc = 0;
					const row = j * N_HEADINGS;
					for (let a = 0; a < N_HEADINGS; a++) acc += probs[a] * ROSE_K[row + a];
					const r = r0 + acc * R;
					const x = px(c) + ROSE_SIN[j] * r;
					const y = py(c) - ROSE_COS[j] * r;
					if (j === 0) ctx.moveTo(x, y);
					else ctx.lineTo(x, y);
				}
				ctx.closePath();
				ctx.fillStyle = tk.accent;
				ctx.globalAlpha = 0.16 + 0.2 * mx;
				ctx.fill();
				ctx.strokeStyle = tk.accent;
				// a decided cell is drawn firmly; an undecided one stays a whisper
				ctx.globalAlpha = 0.38 + 0.5 * mx;
				ctx.lineWidth = 1.1;
				ctx.stroke();
				// the anchor: without a mark at the cell's centre a lobe is just a
				// blob, and the eye has nothing to read its direction against
				ctx.beginPath();
				ctx.arc(px(c), py(c), Math.max(0.9, cell * 0.035), 0, Math.PI * 2);
				ctx.fillStyle = tk.accent;
				ctx.globalAlpha = 0.5 + 0.4 * mx;
				ctx.fill();
			}
			ctx.globalAlpha = 1;
		}

		// ── the markers ──
		ctx.lineWidth = 1.6;
		ctx.strokeStyle = tk.ink2;
		ctx.beginPath();
		ctx.arc(px(sea.start), py(sea.start), cell * 0.24, 0, Math.PI * 2);
		ctx.stroke();

		ctx.fillStyle = tk.accent;
		const hx = px(sea.harbour);
		const hy = py(sea.harbour);
		ctx.beginPath();
		ctx.moveTo(hx, hy - cell * 0.28);
		ctx.lineTo(hx + cell * 0.26, hy);
		ctx.lineTo(hx, hy + cell * 0.28);
		ctx.lineTo(hx - cell * 0.26, hy);
		ctx.closePath();
		ctx.fill();

		// ── the wind, as a cartouche rather than as weather over the water ──
		// It used to be drawn as streaks combed across the whole chart, which put
		// a third set of lines at a third angle behind everything else. The wind
		// is one fact, so it gets one mark.
		{
			const r = Math.min(cell * 0.82, 26);
			const cx = ox + sea.w * cell - r - 6;
			const cy = oy + r + 6;
			const a = sea.windFrom + Math.PI; // the way it blows
			ctx.save();
			ctx.beginPath();
			ctx.arc(cx, cy, r, 0, Math.PI * 2);
			ctx.fillStyle = tk.band;
			ctx.globalAlpha = 0.82;
			ctx.fill();
			ctx.globalAlpha = 0.5;
			ctx.strokeStyle = tk.line;
			ctx.lineWidth = 1;
			ctx.stroke();
			ctx.globalAlpha = 1;
			const ux = Math.sin(a);
			const uy = -Math.cos(a);
			ctx.strokeStyle = tk.cats[1];
			ctx.fillStyle = tk.cats[1];
			ctx.lineWidth = 2;
			ctx.lineCap = 'round';
			ctx.beginPath();
			ctx.moveTo(cx - ux * r * 0.55, cy - uy * r * 0.55);
			ctx.lineTo(cx + ux * r * 0.3, cy + uy * r * 0.3);
			ctx.stroke();
			ctx.beginPath();
			ctx.moveTo(cx + ux * r * 0.62, cy + uy * r * 0.62);
			ctx.lineTo(cx + (-ux * 0.22 - uy * 0.24) * r, cy + (-uy * 0.22 + ux * 0.24) * r);
			ctx.lineTo(cx + (-ux * 0.22 + uy * 0.24) * r, cy + (-uy * 0.22 - ux * 0.24) * r);
			ctx.closePath();
			ctx.fill();
			ctx.restore();
		}

		// ── the wake ──
		if (show && show.path.length > 1) {
			const upto = Math.min(show.path.length - 1, (show.t / show.ms) * (show.path.length - 1));
			ctx.strokeStyle = tk.warm;
			ctx.lineWidth = 2;
			ctx.lineJoin = 'round';
			ctx.beginPath();
			for (let i = 0; i <= Math.floor(upto); i++) {
				const c = show.path[i];
				if (i === 0) ctx.moveTo(px(c), py(c));
				else ctx.lineTo(px(c), py(c));
			}
			ctx.globalAlpha = 0.75;
			ctx.stroke();
			ctx.globalAlpha = 1;
			const c = show.path[Math.floor(upto)];
			ctx.fillStyle = tk.warm;
			ctx.beginPath();
			ctx.arc(px(c), py(c), cell * 0.13, 0, Math.PI * 2);
			ctx.fill();
		}
	}

	/**
	 * Soundings: the value estimate as a diverging wash under smooth contours.
	 *
	 * Two colours rather than one, and the pivot is zero rather than the middle
	 * of the range, because zero means something here — a cell washed vermilion
	 * is one a passage tends to *lose* from, ultramarine is one it tends to win
	 * from, and the line between them is a real boundary rather than a quantile.
	 *
	 * The contours are the reason for the blur. The field is 14 × 10, and rings
	 * traced through a bilinear upsampling of it kink at every cell edge and
	 * trace the interpolation rather than the value. Upsampling and then
	 * smoothing gives a field that is nearly C², so the rings come out as
	 * soundings instead of as scribble.
	 */
	const VAL_U = 6;
	let valCanvas: HTMLCanvasElement | undefined;

	function drawValue(
		ctx: CanvasRenderingContext2D,
		cell: number,
		ox: number,
		oy: number,
		tk: Tokens
	) {
		const GX = sea.w * VAL_U;
		const GY = sea.h * VAL_U;

		// Land never gets visited, so its baseline sits at zero and would carve
		// a false trough through the middle of the chart. Fill it in from its
		// neighbours before smoothing.
		const src = new Float64Array(sea.w * sea.h);
		for (let c = 0; c < src.length; c++) src[c] = baseline[c];
		for (const c of sea.land) {
			let sum = 0;
			let n = 0;
			const x = c % sea.w;
			const y = (c / sea.w) | 0;
			for (let dy = -1; dy <= 1; dy++)
				for (let dx = -1; dx <= 1; dx++) {
					const nx = x + dx;
					const ny = y + dy;
					if (nx < 0 || nx >= sea.w || ny < 0 || ny >= sea.h) continue;
					const k = cellIndex(nx, ny, sea.w);
					if (sea.land.has(k)) continue;
					sum += baseline[k];
					n++;
				}
			if (n) src[c] = sum / n;
		}

		const at = (x: number, y: number) =>
			src[
				cellIndex(Math.max(0, Math.min(sea.w - 1, x)), Math.max(0, Math.min(sea.h - 1, y)), sea.w)
			];

		let f = new Float64Array(GX * GY);
		for (let j = 0; j < GY; j++) {
			for (let i = 0; i < GX; i++) {
				const gx = i / VAL_U - 0.5;
				const gy = sea.h - 1 - (j / VAL_U - 0.5);
				const x0 = Math.floor(gx);
				const y0 = Math.floor(gy);
				const tx = gx - x0;
				const ty = gy - y0;
				f[j * GX + i] =
					at(x0, y0) * (1 - tx) * (1 - ty) +
					at(x0 + 1, y0) * tx * (1 - ty) +
					at(x0, y0 + 1) * (1 - tx) * ty +
					at(x0 + 1, y0 + 1) * tx * ty;
			}
		}
		f = blur(blur(f, GX, GY, 4), GX, GY, 4);

		let span = 0;
		for (const v of f) span = Math.max(span, Math.abs(v));
		if (span < 1e-6) return;

		// the wash, painted small and scaled up so the gradient stays smooth
		if (!valCanvas) valCanvas = document.createElement('canvas');
		valCanvas.width = GX;
		valCanvas.height = GY;
		const vctx = valCanvas.getContext('2d');
		if (!vctx) return;
		const img = vctx.createImageData(GX, GY);
		const hi = hexRgb(tk.accent);
		const lo = hexRgb(tk.warm);
		const bg = hexRgb(tk.band);
		for (let p = 0; p < GX * GY; p++) {
			const t = Math.max(-1, Math.min(1, f[p] / span));
			const c = t >= 0 ? hi : lo;
			// quiet: the wash is the ground the roses are drawn on, not the figure
			const a = Math.pow(Math.abs(t), 1.15) * 0.2;
			img.data[p * 4] = Math.round(bg[0] + (c[0] - bg[0]) * a);
			img.data[p * 4 + 1] = Math.round(bg[1] + (c[1] - bg[1]) * a);
			img.data[p * 4 + 2] = Math.round(bg[2] + (c[2] - bg[2]) * a);
			img.data[p * 4 + 3] = 255;
		}
		vctx.putImageData(img, 0, 0);
		ctx.save();
		ctx.imageSmoothingEnabled = true;
		ctx.imageSmoothingQuality = 'high';
		ctx.drawImage(valCanvas, 0, 0, GX, GY, ox, oy, sea.w * cell, sea.h * cell);

		// the soundings themselves
		const LEVELS = 11;
		const th: number[] = [];
		for (let k = 0; k < LEVELS; k++) th.push(-span + ((k + 0.5) / LEVELS) * 2 * span);
		th.push(0);
		th.sort((a, b) => a - b);
		for (const ring of contours().size([GX, GY]).thresholds(th)(f as unknown as number[])) {
			const zero = Math.abs(ring.value) < 1e-9;
			ctx.beginPath();
			for (const poly of ring.coordinates)
				for (const r of poly) {
					for (let i = 0; i < r.length; i++) {
						const X = ox + (r[i][0] / VAL_U) * cell;
						const Y = oy + (r[i][1] / VAL_U) * cell;
						if (i === 0) ctx.moveTo(X, Y);
						else ctx.lineTo(X, Y);
					}
					ctx.closePath();
				}
			ctx.strokeStyle = ring.value >= 0 ? tk.accent : tk.warm;
			ctx.globalAlpha = zero ? 0.36 : 0.08 + 0.14 * Math.abs(ring.value / span);
			ctx.lineWidth = zero ? 1.6 : 0.9;
			ctx.stroke();
		}
		ctx.restore();
	}

	/** Separable box blur — two passes of this is close enough to a Gaussian. */
	function blur(f: Float64Array, GX: number, GY: number, r: number): Float64Array<ArrayBuffer> {
		const tmp = new Float64Array(f.length);
		const out = new Float64Array(f.length);
		for (let j = 0; j < GY; j++)
			for (let i = 0; i < GX; i++) {
				let s = 0;
				let n = 0;
				for (let k = -r; k <= r; k++) {
					const x = i + k;
					if (x < 0 || x >= GX) continue;
					s += f[j * GX + x];
					n++;
				}
				tmp[j * GX + i] = s / n;
			}
		for (let j = 0; j < GY; j++)
			for (let i = 0; i < GX; i++) {
				let s = 0;
				let n = 0;
				for (let k = -r; k <= r; k++) {
					const y = j + k;
					if (y < 0 || y >= GY) continue;
					s += tmp[y * GX + i];
					n++;
				}
				out[j * GX + i] = s / n;
			}
		return out;
	}

	// ── the map editor ──
	function eventCell(ev: PointerEvent): number {
		if (!canvas) return -1;
		const r = canvas.getBoundingClientRect();
		const { cell, ox, oy } = geom(r.width, r.height);
		const x = Math.floor((ev.clientX - r.left - ox) / cell);
		const y = sea.h - 1 - Math.floor((ev.clientY - r.top - oy) / cell);
		if (x < 0 || x >= sea.w || y < 0 || y >= sea.h) return -1;
		return cellIndex(x, y, sea.w);
	}

	let dragging: 'start' | 'harbour' | null = null;

	function down(ev: PointerEvent) {
		if (!editing) return;
		const c = eventCell(ev);
		if (c === -1) return;
		if (c === sea.start) dragging = 'start';
		else if (c === sea.harbour) dragging = 'harbour';
		else {
			// empty → land → shoal → empty
			if (sea.land.has(c)) {
				sea.land.delete(c);
				sea.shoals.add(c);
			} else if (sea.shoals.has(c)) {
				sea.shoals.delete(c);
			} else {
				sea.land.add(c);
			}
			seaChanged();
			return;
		}
		(ev.target as Element).setPointerCapture(ev.pointerId);
	}

	function move(ev: PointerEvent) {
		if (!dragging) return;
		const c = eventCell(ev);
		if (c === -1 || sea.land.has(c) || sea.shoals.has(c)) return;
		if (dragging === 'start' && c !== sea.harbour) sea.start = c;
		if (dragging === 'harbour' && c !== sea.start) sea.harbour = c;
		seaChanged();
	}

	function up() {
		dragging = null;
	}

	// ── the compass rose ──
	let rose: HTMLElement | undefined = $state();
	let spinning = false;

	function roseAngle(ev: PointerEvent): number {
		if (!rose) return windDeg;
		const r = rose.getBoundingClientRect();
		const dx = ev.clientX - (r.left + r.width / 2);
		const dy = ev.clientY - (r.top + r.height / 2);
		const a = (Math.atan2(dx, -dy) * 180) / Math.PI;
		return Math.round((a + 360) % 360);
	}

	function roseDown(ev: PointerEvent) {
		spinning = true;
		windDeg = roseAngle(ev);
		(ev.target as Element).setPointerCapture(ev.pointerId);
	}
	function roseMove(ev: PointerEvent) {
		if (spinning) windDeg = roseAngle(ev);
	}
	function roseUp() {
		spinning = false;
	}
	function roseKey(ev: KeyboardEvent) {
		const d = ev.key === 'ArrowLeft' ? -15 : ev.key === 'ArrowRight' ? 15 : 0;
		if (!d) return;
		windDeg = (windDeg + d + 360) % 360;
		ev.preventDefault();
	}

	// The rose's geometry: {@const} is only legal directly inside a block, so
	// the arc the no-go zone sweeps is computed here rather than in the markup.
	const NO_GO_DEG = 35;
	const rosePt = (deg: number, r: number) => {
		const a = (deg * Math.PI) / 180;
		return `${(50 + r * Math.sin(a)).toFixed(2)} ${(50 - r * Math.cos(a)).toFixed(2)}`;
	};
	const noGoPath = $derived(
		`M 50 50 L ${rosePt(windDeg - NO_GO_DEG, 42)} A 42 42 0 0 1 ${rosePt(windDeg + NO_GO_DEG, 42)} Z`
	);
	const windTip = $derived({
		x: 50 + 40 * Math.sin((windDeg * Math.PI) / 180),
		y: 50 - 40 * Math.cos((windDeg * Math.PI) / 180)
	});
	const ticks = Array.from({ length: 8 }, (_, i) => {
		const a = (i * Math.PI) / 4;
		return {
			x1: 50 + 36 * Math.sin(a),
			y1: 50 - 36 * Math.cos(a),
			x2: 50 + 42 * Math.sin(a),
			y2: 50 - 42 * Math.cos(a)
		};
	});

	const spark = $derived(sparkPath(returns, 240, 44));
	const bestY = $derived.by(() => {
		if (!returns.length || !Number.isFinite(best)) return null;
		const lo = Math.min(...returns, best);
		const hi = Math.max(...returns, best);
		if (hi - lo < 1e-9) return null;
		return 44 - ((best - lo) / (hi - lo)) * 40 - 2;
	});
</script>

<Plate
	id="chart"
	title="The chart"
	live
	caption="A boat, a harbour to windward, and a rule: no sailing within 35° of the wind — the shaded wedge on the compass. Every cell carries one current rose, a radius per heading, so an undecided policy is a small blob and a decided one is a kite pointing where it means to go. Nothing in the reward mentions zigzagging. Watch it appear anyway. Then drag the compass rose and the whole field reorganizes under a learner that never stopped running; switch on edit to move the harbour, or drop an island in the way."
>
	{#snippet status()}
		<span>{episodes.toLocaleString()} passages</span>
		{#if returns.length > 20}
			<span class="text-ink-3">· {(arrivalRate * 100).toFixed(0)}% arrive</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn onclick={() => (playing = !playing)} kind={playing ? 'ghost' : 'primary'}>
			{#if playing}<Pause size={12} /> Pause{:else}<Play size={12} /> Sail{/if}
		</Btn>
		<Btn
			onclick={() => {
				learn(200);
				flush(); // the play loop flushes on a timer; a manual step must do it itself
			}}
			disabled={playing}><StepForward size={12} /> +200</Btn
		>
		<Btn onclick={resetTheta}><RotateCcw size={12} /> Reset θ</Btn>
	{/snippet}

	<div class="sc px-4" bind:this={host} use:inview={boot}>
		<div class="body">
			<canvas
				bind:this={canvas}
				class:editing
				onpointerdown={down}
				onpointermove={move}
				onpointerup={up}
				onpointercancel={up}
				aria-label="A sea chart: the policy drawn as headings, the wind, land and shoals"
			></canvas>

			<div class="side">
				<div class="rosewrap">
					<div
						class="rose"
						bind:this={rose}
						role="slider"
						tabindex="0"
						aria-label="Wind direction, degrees the wind blows from"
						aria-valuenow={windDeg}
						aria-valuemin="0"
						aria-valuemax="359"
						onpointerdown={roseDown}
						onpointermove={roseMove}
						onpointerup={roseUp}
						onpointercancel={roseUp}
						onkeydown={roseKey}
					>
						<svg viewBox="0 0 100 100" aria-hidden="true">
							<circle cx="50" cy="50" r="42" class="ring" />
							{#each ticks as t, i (i)}
								<line x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2} class="tick" />
							{/each}
							<!-- the no-go zone, drawn where it actually is -->
							<path d={noGoPath} class="nogo" />
							<line x1="50" y1="50" x2={windTip.x} y2={windTip.y} class="windline" />
							<circle cx={windTip.x} cy={windTip.y} r="4.5" class="windknob" />
							<text x="50" y="16" text-anchor="middle" class="rose-n">N</text>
						</svg>
					</div>
					<span class="num rose-cap">wind from {windDeg}°</span>
				</div>

				<dl class="stats num">
					<div>
						<dt>mean return</dt>
						<dd>{returns.length ? meanReturn.toFixed(2) : '—'}</dd>
					</div>
					<div>
						<dt>best possible</dt>
						<dd>{Number.isFinite(best) ? best.toFixed(2) : 'unreachable'}</dd>
					</div>
					<div>
						<dt>arrive</dt>
						<dd>{returns.length > 20 ? `${(arrivalRate * 100).toFixed(0)}%` : '—'}</dd>
					</div>
				</dl>

				<figure class="spark">
					<svg viewBox="0 0 240 44" preserveAspectRatio="none" aria-label="Return per passage">
						{#if bestY !== null}
							<line x1="0" y1={bestY} x2="240" y2={bestY} class="bestline" />
						{/if}
						{#if spark}<path d={spark} class="curve" fill="none" />{/if}
					</svg>
					<figcaption class="num">
						return per passage · dashed is the best any route can do
					</figcaption>
				</figure>

				<div class="toggles">
					<button
						class="chip"
						class:chip-on={showPolicy}
						onclick={() => (showPolicy = !showPolicy)}
					>
						policy
					</button>
					<button class="chip" class:chip-on={showValue} onclick={() => (showValue = !showValue)}>
						value
					</button>
					<button class="chip" class:chip-on={editing} onclick={() => (editing = !editing)}>
						<Pencil size={10} /> edit
					</button>
				</div>

				<Slider
					label="learning rate"
					bind:value={lr}
					min={0.01}
					max={0.3}
					step={0.01}
					format={(v) => v.toFixed(2)}
					tone="knob"
				/>
				{#if editing}
					<p class="hint">
						Click water to drop land, again for a shoal, again to clear it. Drag the ring or the
						harbour. The learner keeps going the whole time.
					</p>
				{/if}
			</div>
		</div>
	</div>
</Plate>

<style>
	.body {
		display: grid;
		grid-template-columns: minmax(0, 2.15fr) minmax(190px, 0.85fr);
		gap: 1.1rem;
		align-items: start;
	}
	canvas {
		display: block;
		width: 100%;
		aspect-ratio: 14 / 10;
		height: auto;
		border: 1px solid var(--line-soft);
		border-radius: var(--r-2);
		touch-action: none;
	}
	canvas.editing {
		cursor: crosshair;
		border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
	}
	.side {
		display: flex;
		flex-direction: column;
		gap: 0.85rem;
	}
	.rosewrap {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.25rem;
	}
	.rose {
		width: 108px;
		cursor: grab;
		touch-action: none;
	}
	.rose:active {
		cursor: grabbing;
	}
	.rose:focus-visible {
		outline: none;
		border-radius: 50%;
		box-shadow: var(--focus-ring);
	}
	.rose svg {
		display: block;
		width: 100%;
		height: auto;
	}
	.ring {
		fill: var(--surface);
		stroke: var(--line);
		stroke-width: 1;
	}
	.tick {
		stroke: var(--ink-3);
		stroke-width: 1;
	}
	.nogo {
		fill: var(--warm);
		opacity: 0.16;
	}
	.windline {
		stroke: var(--cat-1);
		stroke-width: 2;
	}
	.windknob {
		fill: var(--cat-1);
	}
	.rose-n {
		font-family: var(--font-sans);
		font-size: 9px;
		letter-spacing: 0.1em;
		fill: var(--ink-3);
	}
	.rose-cap {
		font-size: 10.5px;
		color: var(--ink-3);
	}
	.stats {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		margin: 0;
		font-size: 11.5px;
		border-top: 1px solid var(--line-soft);
		padding-top: 0.6rem;
	}
	.stats > div {
		display: flex;
		justify-content: space-between;
		gap: 0.75rem;
	}
	.stats dt {
		color: var(--ink-3);
	}
	.stats dd {
		margin: 0;
		color: var(--ink-2);
	}
	.spark svg {
		display: block;
		width: 100%;
		height: 44px;
	}
	.curve {
		stroke: var(--accent);
		stroke-width: 1.2;
		vector-effect: non-scaling-stroke;
	}
	.bestline {
		stroke: var(--ink-3);
		stroke-width: 1;
		stroke-dasharray: 3 3;
		vector-effect: non-scaling-stroke;
	}
	.spark figcaption {
		font-size: 9.5px;
		color: var(--ink-3);
		line-height: 1.35;
	}
	.toggles {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
	}
	.hint {
		margin: 0;
		font-size: 11.5px;
		line-height: 1.45;
		color: var(--ink-2);
		border-left: 2px solid var(--accent);
		padding-left: 0.55rem;
	}
	@media (max-width: 780px) {
		.body {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>

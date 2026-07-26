<script lang="ts">
	// Plate II — the map. Reads the shared engine from Plate I: every trained
	// chunk re-encodes the 2000 held-out digits into the bottleneck and
	// scatters them — as ink, as the digit images themselves, or tinted by
	// their never-seen labels. A two-number waist is the flat sheet; three, or a
	// projection of a wider one, is a turnable cloud. The right pane decodes one
	// point live: on the flat sheet the cursor itself, and in a turnable view the
	// digit nearest the cursor, since a screen ray has no single preimage there.
	import { onDestroy } from 'svelte';
	import { Play, Pause, RotateCcw, Shuffle } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { lab } from './latent-context.svelte';
	import {
		DIM,
		SIDE,
		TileAtlas,
		TileStrip,
		binSubsample,
		project3,
		readTokens,
		setupCanvas
	} from './common';

	interface Props {
		n: number;
		title: string;
		caption: string;
	}
	let { n, title, caption }: Props = $props();

	// The waist is only bounded under tanh, so the frame is a multiple of the
	// measured cloud instead of a constant. The margin is generous because the
	// frame is eased: a cloud still drifting should not touch the edges.
	const PAD = 1.2;
	const PAD3 = 1.55;
	const EYE_MS = 60; // cursor-decode throttle
	const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

	type Mode = 'images' | 'ink';
	const MODES: Mode[] = ['images', 'ink'];
	/** Points glide to their new coordinates instead of teleporting there. */
	const MORPH_MS = 420;
	let mode = $state<Mode>('ink');
	/** Tints by true label — orthogonal to how the map is drawn, so you can put
	 * colour on the points as well as on the thumbnails. */
	let colorize = $state(false);
	/** Cells per axis for the thumbnail subsample: how many digits get printed,
	 * and how big each one can be before they collide. */
	let detail = $state(26);
	let userPicked = false;
	let t = $state(0);
	let pair = $state<[number, number] | null>(null);
	let eyeZ = $state('—');
	let eyeSrc = $state('cloud centre');
	let visibleCount = $state(0);
	let dragging = $state(false);
	/** Reactive mirrors of the view's shape, for the template. */
	let viewDim = $state(2);
	let projected = $state(false);

	let scatterCanvas: HTMLCanvasElement | undefined = $state();
	let eyeCanvas: HTMLCanvasElement | undefined = $state();
	let miniA: HTMLCanvasElement | undefined = $state();
	let miniB: HTMLCanvasElement | undefined = $state();

	// ── non-reactive machinery (the rAF painter reads these directly) ──
	let latents: Float32Array | null = null; // 2000 × latDim, display coordinates
	let latDim = 2; // true width of `latents` (a rebuild can lag the lab)
	let full: Float32Array | null = null; // 2000 × the real waist, for decoding
	let fullDim = 2;
	// Every trained chunk re-encodes all two thousand digits, which moves every
	// point at once. Painting the previous positions gliding into the new ones
	// turns that jump into drift — the same information, far easier to follow.
	let shown: Float32Array | null = null; // what is on screen right now
	let morphFrom: Float32Array | null = null;
	let morphAt = 1; // 0 → 1 across MORPH_MS
	let visible: Int32Array | null = null; // subsampled indices for thumbnails
	let centroidV: number[] = [0, 0];
	let hoverZ: [number, number] | null = null; // 2-D free cursor
	let hoverIdx = -1; // 3-D nearest digit
	let proj: Float32Array | null = null; // n×3 screen coords + depth (3-D hit-tests)
	let yaw = 0.7;
	let pitch = 0.35;
	let zoom = 1;
	let dragLast: [number, number] | null = null;
	let lastDraw = 0;
	let walkTouched = false;
	let prevT = 0;
	let origA: Float32Array | null = null;
	let origB: Float32Array | null = null;
	let pairVersion = 0;
	let eyePixels: Float32Array | null = null;
	let eyeVersion = 0;
	let raf = 0;
	let eyeBusy = false;
	let eyeWanted = false;
	let lastEyeAt = 0;
	let eyeTimer: ReturnType<typeof setTimeout> | null = null;
	const eyeStrip = new TileStrip();
	const aStrip = new TileStrip();
	const bStrip = new TileStrip();
	const inkAtlas = new TileAtlas();
	const catAtlas = new TileAtlas();
	const reduced =
		typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

	// ── re-encode on every trained chunk, then adopt the shared result ──
	$effect(() => {
		void lab.tick;
		if (lab.phase !== 'ready') return;
		void lab.refreshTestLatents();
	});

	// once reconstructions clear the quality bar, the map earns its images
	$effect(() => {
		if (lab.trained && !userPicked && mode === 'ink') mode = 'images';
	});

	function pickMode(m: Mode): void {
		userPicked = true;
		mode = m;
	}

	$effect(() => {
		void lab.zVersion;
		const v = lab.viewZ;
		const vd = lab.viewDim;
		const m = lab.mnist;
		if (!v || !vd || !m) return;
		if (vd !== latDim) {
			// the view changed shape — screen-space state is stale
			hoverZ = null;
			hoverIdx = -1;
			proj = null;
		}
		const n = m.testY.length;
		// hand the painter a start point before adopting the new coordinates
		if (shown && shown.length === v.length && !reduced) {
			morphFrom = shown;
			morphAt = 0;
		} else {
			morphFrom = null;
			morphAt = 1;
			shown = v;
		}
		latents = v;
		latDim = vd;
		viewDim = vd;
		projected = !!lab.basis;
		full = lab.testZ;
		fullDim = lab.testZd;
		visible = binSubsample(v, n, vd, lab.zCenter, lab.zSpan, detail);
		visibleCount = visible.length;
		const c = new Array(vd).fill(0);
		for (let i = 0; i < n; i++) for (let k = 0; k < vd; k++) c[k] += v[i * vd + k];
		centroidV = c.map((x) => x / n);
		if (!pair) defaultPair();
		requestEye(); // the weights moved — whatever the eye shows is stale
	});

	// ── the walk: two held-out digits and the segment between them ──
	function setPair(a: number, b: number): void {
		const m = lab.mnist;
		if (!m) return;
		pair = [a, b];
		origA = m.testX.subarray(a * DIM, (a + 1) * DIM);
		origB = m.testX.subarray(b * DIM, (b + 1) * DIM);
		pairVersion++;
		requestEye();
	}

	function defaultPair(): void {
		const m = lab.mnist;
		if (!m) return;
		// a 3 morphing into an 8 is the canonical first walk
		let a = -1;
		let b = -1;
		for (let i = 0; i < m.testY.length && (a < 0 || b < 0); i++) {
			if (a < 0 && m.testY[i] === 3) a = i;
			else if (b < 0 && m.testY[i] === 8) b = i;
		}
		setPair(a < 0 ? 0 : a, b < 0 ? 1 : b);
	}

	function randomPair(): void {
		const m = lab.mnist;
		if (!m) return;
		const n = m.testY.length;
		const a = Math.floor(Math.random() * n);
		let b = Math.floor(Math.random() * n);
		let guard = 0;
		while ((b === a || m.testY[b] === m.testY[a]) && guard++ < 50)
			b = Math.floor(Math.random() * n);
		setPair(a, b);
	}

	/** The walk, in the coordinates the map is drawn in — the gliding ones, so
	 * the segment stays pinned to its two endpoint digits while they move. */
	function walkView(): number[] {
		const z = shown ?? latents!;
		const [a, b] = pair!;
		const out: number[] = [];
		for (let k = 0; k < latDim; k++)
			out.push(z[a * latDim + k] + (z[b * latDim + k] - z[a * latDim + k]) * t);
		return out;
	}

	/** The same walk in the real waist, which is what the decoder is asked for —
	 * interpolating there keeps both endpoints exactly on their own digits even
	 * when the map is only a shadow of a wider space. */
	function walkFull(): Float32Array | null {
		if (!full || !pair) return null;
		const [a, b] = pair;
		const out = new Float32Array(fullDim);
		for (let k = 0; k < fullDim; k++)
			out[k] = full[a * fullDim + k] + (full[b * fullDim + k] - full[a * fullDim + k]) * t;
		return out;
	}

	function viewRow(i: number): number[] {
		const z = latents!;
		const out: number[] = [];
		for (let k = 0; k < latDim; k++) out.push(z[i * latDim + k]);
		return out;
	}

	// moving the slider hands the eye to the walk point
	$effect(() => {
		const now = t;
		if (now !== prevT) {
			prevT = now;
			walkTouched = true;
			requestEye();
		}
	});

	// ── the decoder's eye: throttled single-point decodes ──
	/** What the decoder is asked (`z`, in the real waist) and what the readout
	 * prints (`view`, the two or three numbers you can actually see). */
	function currentEye(): { z: Float32Array; view: number[]; src: string } {
		if (latDim === 2 && hoverZ) return { z: lab.lift(hoverZ), view: hoverZ, src: 'cursor' };
		if (latDim === 3 && hoverIdx >= 0 && latents && full) {
			const row = full.slice(hoverIdx * fullDim, (hoverIdx + 1) * fullDim);
			if (row.length === fullDim) return { z: row, view: viewRow(hoverIdx), src: 'nearest digit' };
		}
		if (walkTouched && pair && latents) {
			const w = walkFull();
			if (w) return { z: w, view: walkView(), src: 'walk' };
		}
		return { z: lab.lift(centroidV), view: centroidV, src: 'cloud centre' };
	}

	function requestEye(): void {
		eyeWanted = true;
		void pumpEye();
	}

	async function pumpEye(): Promise<void> {
		if (eyeBusy || !eyeWanted || lab.phase !== 'ready' || !lab.engine) return;
		const now = performance.now();
		const wait = lastEyeAt + EYE_MS - now;
		if (wait > 0) {
			eyeTimer ??= setTimeout(() => {
				eyeTimer = null;
				void pumpEye();
			}, wait);
			return;
		}
		eyeWanted = false;
		lastEyeAt = now;
		eyeBusy = true;
		const { z, view, src } = currentEye();
		try {
			eyePixels = await lab.decode(z, 1);
			eyeVersion++;
			eyeZ = `(${view.map(fz).join(', ')})`;
			eyeSrc = src;
		} catch {
			// engine went away — nothing to draw
		}
		eyeBusy = false;
		if (eyeWanted) void pumpEye();
	}

	const fz = (v: number) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2);
	const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(4) : '—');

	// ── pointer: in 2-D the cursor IS a latent coordinate; in 3-D it
	// rotates (drag) or picks the nearest digit (hover) ──
	function eventZ(ev: PointerEvent): [number, number] | null {
		if (!scatterCanvas) return null;
		const r = scatterCanvas.getBoundingClientRect();
		if (r.width < 4) return null;
		const s = Math.min(r.width, r.height) / (2 * lab.zSpan * PAD);
		const [cx, cy] = lab.zCenter;
		return [
			cx + (ev.clientX - r.left - r.width / 2) / s,
			cy - (ev.clientY - r.top - r.height / 2) / s
		];
	}

	function hoverNearest(ev: PointerEvent): void {
		if (!proj || !scatterCanvas) return;
		const r = scatterCanvas.getBoundingClientRect();
		const mx = ev.clientX - r.left;
		const my = ev.clientY - r.top;
		let best = -1;
		let bestD = 22 * 22;
		const n = proj.length / 3;
		for (let i = 0; i < n; i++) {
			const dx = proj[3 * i] - mx;
			const dy = proj[3 * i + 1] - my;
			const dd = dx * dx + dy * dy;
			if (dd < bestD) {
				bestD = dd;
				best = i;
			}
		}
		if (best !== hoverIdx) {
			hoverIdx = best;
			requestEye();
		}
	}

	function ptrMove(ev: PointerEvent): void {
		if (latDim === 3) {
			if (dragging && dragLast) {
				yaw += (ev.clientX - dragLast[0]) * 0.008;
				pitch = Math.max(-1.3, Math.min(1.3, pitch + (ev.clientY - dragLast[1]) * 0.006));
				dragLast = [ev.clientX, ev.clientY];
			} else {
				hoverNearest(ev);
			}
			return;
		}
		hoverZ = eventZ(ev);
		requestEye();
	}
	function ptrDown(ev: PointerEvent): void {
		scatterCanvas?.setPointerCapture(ev.pointerId);
		if (latDim === 3) {
			dragging = true;
			dragLast = [ev.clientX, ev.clientY];
			return;
		}
		hoverZ = eventZ(ev);
		requestEye();
	}
	function ptrUp(ev: PointerEvent): void {
		dragging = false;
		dragLast = null;
		if (latDim === 2 && ev.pointerType !== 'mouse') {
			hoverZ = null;
			requestEye();
		}
	}
	function ptrLeave(): void {
		dragging = false;
		dragLast = null;
		hoverZ = null;
		if (hoverIdx !== -1) hoverIdx = -1;
		requestEye();
	}

	// wheel zoom (3-D only) — attached manually because Svelte's wheel
	// handlers are passive and preventDefault must win over page scroll
	$effect(() => {
		if (lab.phase !== 'ready') return;
		const el = scatterCanvas;
		if (!el) return;
		const onWheel = (e: WheelEvent) => {
			if (latDim !== 3) return;
			e.preventDefault();
			zoom = Math.max(0.5, Math.min(2.6, zoom * Math.exp(-e.deltaY * 0.0012)));
		};
		el.addEventListener('wheel', onWheel, { passive: false });
		return () => el.removeEventListener('wheel', onWheel);
	});

	// ── painter ──
	$effect(() => {
		if (lab.phase !== 'ready') return;
		const paint = (now: number) => {
			raf = requestAnimationFrame(paint);
			const dt = lastDraw ? Math.min(50, now - lastDraw) : 16;
			lastDraw = now;
			// the 3-D cloud turns slowly on its own — never under reduced motion
			if (latDim === 3 && !reduced && !dragging) yaw += dt * 0.00013;
			advanceMorph(dt);
			drawScatter();
			drawTile(eyeCanvas, eyeStrip, eyePixels, eyeVersion);
			drawTile(miniA, aStrip, origA, pairVersion);
			drawTile(miniB, bStrip, origB, pairVersion);
		};
		raf = requestAnimationFrame(paint);
		return () => cancelAnimationFrame(raf);
	});

	/** Walk `shown` from the previous coordinates toward the newest ones. */
	function advanceMorph(dt: number): void {
		const to = latents;
		const from = morphFrom;
		if (!to) return;
		if (!from || from.length !== to.length || morphAt >= 1) {
			shown = to;
			morphAt = 1;
			morphFrom = null;
			return;
		}
		morphAt = Math.min(1, morphAt + dt / MORPH_MS);
		const e = morphAt < 0.5 ? 2 * morphAt * morphAt : 1 - 2 * (1 - morphAt) * (1 - morphAt);
		const out = shown && shown !== to && shown !== from ? shown : new Float32Array(to.length);
		for (let i = 0; i < to.length; i++) out[i] = from[i] + (to[i] - from[i]) * e;
		shown = out;
		if (morphAt >= 1) morphFrom = null;
	}

	/** Tiles just big enough to cover their cell of the subsample grid: ask for
	 * more digits and each one gets smaller, so the sheet stays readable. */
	function thumbSize(W: number): number {
		return Math.max(7, Math.min(26, (W / detail) * 1.25));
	}

	function atlasFor(
		tk: ReturnType<typeof readTokens>
	): { sheet: HTMLCanvasElement; cols: number } | null {
		const m = lab.mnist;
		if (!m) return null;
		const n = m.testY.length;
		if (colorize) {
			const sheet = catAtlas.ensure(m.testX, n, m.testY, tk.surface, tk.cats);
			return { sheet, cols: catAtlas.cols };
		}
		const sheet = inkAtlas.ensure(m.testX, n, null, tk.surface, [tk.ink]);
		return { sheet, cols: inkAtlas.cols };
	}

	function thumb(
		ctx: CanvasRenderingContext2D,
		sheet: HTMLCanvasElement,
		cols: number,
		i: number,
		cx: number,
		cy: number,
		size: number
	): void {
		ctx.drawImage(
			sheet,
			(i % cols) * SIDE,
			Math.floor(i / cols) * SIDE,
			SIDE,
			SIDE,
			cx - size / 2,
			cy - size / 2,
			size,
			size
		);
	}

	function drawScatter(): void {
		if (!scatterCanvas) return;
		const { ctx, W, H } = setupCanvas(scatterCanvas);
		if (W < 40) return;
		const tk = readTokens(scatterCanvas);
		ctx.clearRect(0, 0, W, H);
		if (latDim === 3) drawScatter3(ctx, W, H, tk);
		else drawScatter2(ctx, W, H, tk);
	}

	/** Screen mapping for the cloud's frame. One scale for both axes, so the map
	 * keeps its geometry whatever shape the cell is; a taller cell simply shows
	 * more empty space above and below rather than stretching the country. */
	function frameMap(W: number, H: number, pad: number) {
		const s = Math.min(W, H) / (2 * lab.zSpan * pad);
		const [cx, cy] = lab.zCenter;
		return {
			s,
			px: (x: number) => W / 2 + (x - cx) * s,
			py: (y: number) => H / 2 - (y - cy) * s
		};
	}

	function drawScatter2(
		ctx: CanvasRenderingContext2D,
		W: number,
		H: number,
		tk: ReturnType<typeof readTokens>
	): void {
		const { px, py } = frameMap(W, H, PAD);

		// the origin's cross where it falls — the one piece of reference
		// geometry the map keeps
		ctx.lineWidth = 1;
		ctx.strokeStyle = tk.lineSoft;
		ctx.beginPath();
		ctx.moveTo(0, py(0));
		ctx.lineTo(W, py(0));
		ctx.moveTo(px(0), 0);
		ctx.lineTo(px(0), H);
		ctx.stroke();

		const z = shown;
		const m = lab.mnist;
		if (z && m) {
			const n = m.testY.length;
			if (mode === 'ink') {
				ctx.globalAlpha = colorize ? 0.85 : 0.9;
				let fill = '';
				for (let i = 0; i < n; i++) {
					const want = colorize ? tk.cats[m.testY[i]] : tk.ink3;
					if (want !== fill) {
						fill = want;
						ctx.fillStyle = want;
					}
					ctx.beginPath();
					ctx.arc(px(z[2 * i]), py(z[2 * i + 1]), 2.5, 0, Math.PI * 2);
					ctx.fill();
				}
				ctx.globalAlpha = 1;
			} else if (visible) {
				const at = atlasFor(tk);
				if (at) {
					ctx.imageSmoothingEnabled = false; // digit pixels stay crisp
					const size = thumbSize(W);
					for (let k = 0; k < visible.length; k++) {
						const i = visible[k];
						thumb(ctx, at.sheet, at.cols, i, px(z[2 * i]), py(z[2 * i + 1]), size);
					}
				}
			}

			const p = pair;
			if (p) {
				const ax = px(z[2 * p[0]]);
				const ay = py(z[2 * p[0] + 1]);
				const bx = px(z[2 * p[1]]);
				const by = py(z[2 * p[1] + 1]);
				drawSegment(ctx, tk, ax, ay, bx, by);
				const wz = walkView();
				drawWalker(ctx, tk, px(wz[0]), py(wz[1]));
			}
		}

		if (hoverZ) {
			const hx = px(hoverZ[0]);
			const hy = py(hoverZ[1]);
			ctx.globalAlpha = 0.55;
			ctx.strokeStyle = tk.ink3;
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(0, hy);
			ctx.lineTo(W, hy);
			ctx.moveTo(hx, 0);
			ctx.lineTo(hx, H);
			ctx.stroke();
			ctx.globalAlpha = 1;
			ctx.beginPath();
			ctx.arc(hx, hy, 5, 0, Math.PI * 2);
			ctx.strokeStyle = tk.ink;
			ctx.lineWidth = 1.4;
			ctx.stroke();
		}
	}

	function drawScatter3(
		ctx: CanvasRenderingContext2D,
		W: number,
		H: number,
		tk: ReturnType<typeof readTokens>
	): void {
		const span = lab.zSpan;
		const c = lab.zCenter;
		const s = Math.min(W, H) / (2 * span * PAD3);
		// centred on the cloud, so an unbounded waist that drifts off the origin
		// still lands in frame
		const toXY = (x: number, y: number, zc: number): [number, number, number] => {
			const [vx, vy, vz] = project3(
				(x - (c[0] ?? 0)) * zoom,
				(y - (c[1] ?? 0)) * zoom,
				(zc - (c[2] ?? 0)) * zoom,
				yaw,
				pitch
			);
			return [W / 2 + vx * s, H / 2 - vy * s, vz];
		};

		// a cage at the reach of the cloud, as a wireframe — without it a
		// rotating cloud has no depth cue at all
		ctx.strokeStyle = tk.line;
		ctx.lineWidth = 1;
		ctx.globalAlpha = 0.55;
		for (const [a, b] of CUBE_EDGES) {
			const [x0, y0] = toXY(
				(c[0] ?? 0) + CUBE[a][0] * span,
				(c[1] ?? 0) + CUBE[a][1] * span,
				(c[2] ?? 0) + CUBE[a][2] * span
			);
			const [x1, y1] = toXY(
				(c[0] ?? 0) + CUBE[b][0] * span,
				(c[1] ?? 0) + CUBE[b][1] * span,
				(c[2] ?? 0) + CUBE[b][2] * span
			);
			ctx.beginPath();
			ctx.moveTo(x0, y0);
			ctx.lineTo(x1, y1);
			ctx.stroke();
		}
		ctx.globalAlpha = 1;

		const z = shown;
		const m = lab.mnist;
		if (!z || !m || latDim !== 3) {
			proj = null;
			return;
		}
		const n = m.testY.length;
		const pr = proj && proj.length === n * 3 ? proj : new Float32Array(n * 3);
		proj = pr;
		for (let i = 0; i < n; i++) {
			const [sx, sy, sz] = toXY(z[3 * i], z[3 * i + 1], z[3 * i + 2]);
			pr[3 * i] = sx;
			pr[3 * i + 1] = sy;
			pr[3 * i + 2] = sz;
		}
		const depthOf = (i: number) => pr[3 * i + 2];
		// depth → 0..1 for shading, over the diagonal reach of the cage
		const far = 1.9 * span;
		const dn = (i: number) => Math.max(0, Math.min(1, (depthOf(i) + far) / (2 * far)));

		if (mode === 'ink') {
			const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => depthOf(a) - depthOf(b));
			let fill = '';
			for (const i of order) {
				const d = dn(i);
				const want = colorize ? tk.cats[m.testY[i]] : tk.ink3;
				if (want !== fill) {
					fill = want;
					ctx.fillStyle = want;
				}
				ctx.globalAlpha = 0.35 + 0.6 * d;
				ctx.beginPath();
				ctx.arc(pr[3 * i], pr[3 * i + 1], 1.8 + 1.6 * d, 0, Math.PI * 2);
				ctx.fill();
			}
			ctx.globalAlpha = 1;
		} else if (visible) {
			const at = atlasFor(tk);
			if (at) {
				const order = Array.from(visible).sort((a, b) => depthOf(a) - depthOf(b));
				const size = thumbSize(W) * 0.82;
				ctx.imageSmoothingEnabled = false;
				for (const i of order) {
					const d = dn(i);
					ctx.globalAlpha = 0.45 + 0.55 * d;
					thumb(ctx, at.sheet, at.cols, i, pr[3 * i], pr[3 * i + 1], size * (0.7 + 0.6 * d));
				}
				ctx.globalAlpha = 1;
			}
		}

		const p = pair;
		if (p) {
			drawSegment(ctx, tk, pr[3 * p[0]], pr[3 * p[0] + 1], pr[3 * p[1]], pr[3 * p[1] + 1]);
			const wz = walkView();
			const [wx, wy] = toXY(wz[0], wz[1], wz[2]);
			drawWalker(ctx, tk, wx, wy);
		}

		if (hoverIdx >= 0 && hoverIdx < n) {
			ctx.beginPath();
			ctx.arc(pr[3 * hoverIdx], pr[3 * hoverIdx + 1], 11, 0, Math.PI * 2);
			ctx.strokeStyle = tk.ink;
			ctx.lineWidth = 1.5;
			ctx.stroke();
		}
	}

	function drawSegment(
		ctx: CanvasRenderingContext2D,
		tk: ReturnType<typeof readTokens>,
		ax: number,
		ay: number,
		bx: number,
		by: number
	): void {
		ctx.strokeStyle = tk.ink;
		ctx.lineWidth = 1.3;
		ctx.beginPath();
		ctx.moveTo(ax, ay);
		ctx.lineTo(bx, by);
		ctx.stroke();
		for (const [ex, ey] of [
			[ax, ay],
			[bx, by]
		] as const) {
			ctx.beginPath();
			ctx.arc(ex, ey, 4, 0, Math.PI * 2);
			ctx.fillStyle = tk.surface;
			ctx.fill();
			ctx.strokeStyle = tk.ink;
			ctx.stroke();
		}
	}

	function drawWalker(
		ctx: CanvasRenderingContext2D,
		tk: ReturnType<typeof readTokens>,
		x: number,
		y: number
	): void {
		ctx.beginPath();
		ctx.arc(x, y, 4.5, 0, Math.PI * 2);
		ctx.fillStyle = tk.accent;
		ctx.fill();
		ctx.strokeStyle = tk.paper;
		ctx.lineWidth = 1.5;
		ctx.stroke();
	}

	const CUBE: Array<[number, number, number]> = [
		[-1, -1, -1],
		[1, -1, -1],
		[-1, 1, -1],
		[1, 1, -1],
		[-1, -1, 1],
		[1, -1, 1],
		[-1, 1, 1],
		[1, 1, 1]
	];
	const CUBE_EDGES: Array<[number, number]> = [
		[0, 1],
		[0, 2],
		[0, 4],
		[1, 3],
		[1, 5],
		[2, 3],
		[2, 6],
		[3, 7],
		[4, 5],
		[4, 6],
		[5, 7],
		[6, 7]
	];

	function drawTile(
		canvas: HTMLCanvasElement | undefined,
		strip: TileStrip,
		data: Float32Array | null,
		version: number
	): void {
		if (!canvas) return;
		const { ctx, W, H } = setupCanvas(canvas);
		if (W < 8) return;
		ctx.clearRect(0, 0, W, H);
		const tk = readTokens(canvas);
		const tiles = strip.ensure(data, 1, version, tk.surface, tk.ink);
		if (!tiles) return;
		ctx.imageSmoothingEnabled = false; // digit pixels stay crisp
		const s = Math.min(W, H);
		ctx.drawImage(tiles[0], (W - s) / 2, (H - s) / 2, s, s);
	}

	const pairLabels = $derived.by(() => {
		const m = lab.mnist;
		if (!pair || !m) return ['·', '·'];
		return [String(m.testY[pair[0]]), String(m.testY[pair[1]])];
	});

	onDestroy(() => {
		if (eyeTimer !== null) clearTimeout(eyeTimer);
	});
</script>

<Plate {n} {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'ready'}
			<span>step {lab.step}</span>
			<span aria-hidden="true">·</span>
			<span>val {fmt(lab.valLoss)}</span>
			<span aria-hidden="true">·</span>
			<span>{lab.latentDim}-number waist</span>
			{#if projected}
				<span aria-hidden="true">·</span>
				<span>shown as its 3 strongest directions</span>
			{/if}
		{:else if lab.phase === 'loading'}
			<span>warming up…</span>
		{:else if lab.phase === 'error'}
			<span style="color: var(--bad);">error</span>
		{/if}
	{/snippet}

	{#snippet actions()}
		<Btn
			disabled={lab.phase !== 'ready' || lab.rebuilding}
			onclick={() => lab.setTraining(!lab.training)}
		>
			{#if lab.training}
				<Pause size={12} aria-hidden="true" /> Pause
			{:else}
				<Play size={12} aria-hidden="true" /> Train
			{/if}
		</Btn>
		<Btn
			disabled={lab.phase !== 'ready' || lab.rebuilding}
			onclick={() => void lab.resetWeights()}
			title="Fresh random weights"
		>
			<RotateCcw size={12} aria-hidden="true" /> Reset
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void lab.boot()}>
		{#if lab.phase !== 'ready'}
			<div class="flex h-[280px] flex-col items-center justify-center gap-1">
				<span class="eyebrow">
					{lab.phase === 'error' ? 'the engine stalled' : 'warming up the same network as plate I…'}
				</span>
				{#if lab.phase === 'error'}
					<span class="mb-2 text-[12.5px] text-bad">{lab.errorMsg}</span>
					<Btn onclick={() => void lab.boot()}>Retry</Btn>
				{:else}
					<span class="text-[12.5px] text-ink-3">
						two thousand held-out digits will place themselves here
					</span>
				{/if}
			</div>
		{:else}
			<!-- the map, and beside it the instrument that reads it -->
			<div
				class="grid grid-cols-1 gap-px bg-line-soft lg:grid-cols-[minmax(0,520px)_minmax(0,1fr)]"
			>
				<div class="relative min-h-[440px] bg-surface">
					<span class="eyebrow absolute top-3 left-3 z-10">
						{projected ? 'latent space · shadow of z = E(x)' : 'latent space · z = E(x)'}
						{#if viewDim === 3}
							<span class="tracking-normal text-ink-3 normal-case"
								>— drag to turn · wheel to zoom</span
							>
						{/if}
					</span>
					{#if mode === 'images'}
						<div class="absolute right-3 bottom-2.5 left-3 z-10 flex items-center gap-2.5">
							<input
								class="dens"
								type="range"
								min="12"
								max="64"
								step="1"
								bind:value={detail}
								aria-label="How many digits to print"
							/>
							<span class="num text-[10.5px] text-ink-3">{visibleCount} of 2000 shown</span>
						</div>
					{/if}
					<canvas
						bind:this={scatterCanvas}
						class="absolute inset-0 block h-full w-full touch-none"
						class:cursor-crosshair={viewDim === 2}
						class:cursor-grab={viewDim === 3 && !dragging}
						class:cursor-grabbing={dragging}
						aria-label="Two thousand held-out digits scattered by their latent coordinates"
						onpointermove={ptrMove}
						onpointerdown={ptrDown}
						onpointerup={ptrUp}
						onpointercancel={ptrUp}
						onpointerleave={ptrLeave}
					></canvas>
				</div>

				<div class="flex flex-col gap-3 bg-surface p-4">
					<!-- what the map is made of -->
					<div class="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
						<span class="eyebrow">the view</span>
						<span class="flex items-center gap-1.5">
							<span class="flex items-center gap-1.5" role="group" aria-label="Map view">
								{#each MODES as m (m)}
									<button
										class="chip"
										class:chip-on={mode === m}
										aria-pressed={mode === m}
										onclick={() => pickMode(m)}
									>
										{m}
									</button>
								{/each}
							</span>
							<button
								class="chip ml-1"
								class:chip-on={colorize}
								aria-pressed={colorize}
								onclick={() => (colorize = !colorize)}
							>
								colorize
							</button>
						</span>
					</div>
					<div class="flex min-h-[19px] flex-wrap items-center gap-x-3 gap-y-1">
						{#if colorize}
							{#each DIGITS as d (d)}
								<span class="num flex items-center gap-1.5 text-[11px] text-ink-2">
									<i class="dot" style="background: var(--cat-{d});" aria-hidden="true"></i>{d}
								</span>
							{/each}
						{:else}
							<span class="text-[11px] text-ink-3">
								{mode === 'ink'
									? 'anonymous points — the map as the model knows it'
									: 'one digit printed per occupied cell of the sheet'}
							</span>
						{/if}
					</div>

					<!-- the decoder's eye: the reason to point at the map at all,
					     so it gets the room -->
					<div class="flex flex-col gap-1.5">
						<div class="relative mx-auto w-full max-w-[300px]">
							<canvas
								bind:this={eyeCanvas}
								class="block aspect-square w-full rounded border border-line-soft"
								aria-label="The decoder's output for the current latent point"
							></canvas>
							<span class="eyebrow absolute top-2 left-2.5">the decoder's eye</span>
						</div>
						<div
							class="mx-auto flex w-full max-w-[300px] flex-wrap items-baseline justify-between gap-x-3"
						>
							<span class="num text-[12.5px] text-ink">z = {eyeZ}</span>
							<span class="num text-[10.5px] text-ink-3">from the {eyeSrc}</span>
						</div>
						<span class="mx-auto max-w-[300px] text-[11px] leading-snug text-ink-3">
							{#if viewDim === 2}
								every point of the plane decodes to something digit-shaped — even where no digit has
								ever been
							{:else}
								a screen ray has no single preimage in the cloud, so the eye snaps to the digit
								nearest your cursor
							{/if}
						</span>
					</div>

					<!-- the walk -->
					<div class="mt-auto flex flex-col gap-2 border-t border-line-soft pt-3">
						<div class="flex items-center justify-between gap-3">
							<span class="eyebrow">walk the map</span>
							<Btn onclick={randomPair}>
								<Shuffle size={12} aria-hidden="true" /> Random pair
							</Btn>
						</div>
						<div class="flex items-center gap-3">
							<span class="flex shrink-0 items-center gap-1.5">
								<canvas
									bind:this={miniA}
									class="h-10 w-10 rounded border border-line-soft"
									aria-label="First endpoint digit"
								></canvas>
								<span class="num text-[11px] text-ink-2">{pairLabels[0]}</span>
							</span>
							<span class="min-w-32 flex-1">
								<Slider
									label="interpolation t"
									bind:value={t}
									min={0}
									max={1}
									step={0.01}
									format={(v) => v.toFixed(2)}
								/>
							</span>
							<span class="flex shrink-0 items-center gap-1.5">
								<span class="num text-[11px] text-ink-2">{pairLabels[1]}</span>
								<canvas
									bind:this={miniB}
									class="h-10 w-10 rounded border border-line-soft"
									aria-label="Second endpoint digit"
								></canvas>
							</span>
						</div>
					</div>
				</div>
			</div>
		{/if}
	</div>
</Plate>

<style>
	.dot {
		display: inline-block;
		width: 7px;
		height: 7px;
		border-radius: 50%;
	}

	/* how many digits to print — a hairline control that sits on the map
	   without competing with it */
	.dens {
		appearance: none;
		width: 104px;
		height: 12px;
		background: transparent;
		cursor: pointer;
	}
	.dens::-webkit-slider-runnable-track {
		height: 2px;
		border-radius: 1px;
		background: var(--line);
	}
	.dens::-webkit-slider-thumb {
		appearance: none;
		width: 9px;
		height: 9px;
		margin-top: -3.5px;
		border: none;
		border-radius: 50%;
		background: var(--ink-3);
		transition: background 100ms ease;
	}
	.dens:hover::-webkit-slider-thumb {
		background: var(--ink-2);
	}
	.dens::-moz-range-track {
		height: 2px;
		border-radius: 1px;
		background: var(--line);
	}
	.dens::-moz-range-thumb {
		width: 9px;
		height: 9px;
		border: none;
		border-radius: 50%;
		background: var(--ink-3);
	}

	.chip {
		font-family: var(--font-sans);
		font-size: 11px;
		font-weight: 500;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		padding: 4px 11px;
		border-radius: 5px;
		border: 1px solid var(--line);
		color: var(--ink-2);
		background: var(--surface);
		transition: all 100ms ease;
	}
	.chip:hover {
		border-color: var(--ink-3);
		color: var(--ink);
	}
	.chip-on {
		background: var(--ink);
		border-color: var(--ink);
		color: var(--paper);
	}
	/* without this the selected chip inherits the hover ink colour and goes
	   dark-on-dark the moment the pointer touches it */
	.chip-on:hover {
		color: var(--paper);
		background: color-mix(in srgb, var(--ink) 88%, var(--paper));
		border-color: color-mix(in srgb, var(--ink) 88%, var(--paper));
	}
</style>

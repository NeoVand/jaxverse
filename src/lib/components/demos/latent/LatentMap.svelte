<script lang="ts">
	// Plate II — the map. Reads the shared engine from Plate I: every trained
	// chunk re-encodes the 2000 held-out digits into the bottleneck and
	// scatters them — as ink, as the digit images themselves, or tinted by
	// their never-seen labels. With a width-3 bottleneck the sheet becomes a
	// turnable, zoomable cloud. The right pane decodes one latent point live:
	// in 2-D the cursor itself; in 3-D the nearest digit to the cursor
	// (a screen ray has no single latent preimage, so we snap to data).
	import { onDestroy } from 'svelte';
	import { Play, Pause, Shuffle } from 'lucide-svelte';
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

	const EXTENT = 1.12; // 2-D view half-width in latent units — tanh lives in (−1,1)
	const EXTENT3 = 1.5; // 3-D view half-width (rotated cube corners reach √3)
	const EYE_MS = 60; // cursor-decode throttle
	const THUMB = 16; // 2-D thumbnail edge, px
	const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

	type Mode = 'images' | 'ink' | 'reveal';
	const MODES: Mode[] = ['images', 'ink', 'reveal'];
	let mode = $state<Mode>('ink');
	let userPicked = false;
	let t = $state(0);
	let pair = $state<[number, number] | null>(null);
	let eyeLabel = $state('z = —');
	let visibleCount = $state(0);
	let dragging = $state(false);

	let scatterCanvas: HTMLCanvasElement | undefined = $state();
	let eyeCanvas: HTMLCanvasElement | undefined = $state();
	let miniA: HTMLCanvasElement | undefined = $state();
	let miniB: HTMLCanvasElement | undefined = $state();

	// ── non-reactive machinery (the rAF painter reads these directly) ──
	let latents: Float32Array | null = null; // 2000 × latDim
	let latDim = 2; // true width of `latents` (a rebuild can lag latentDim)
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
	let encBusy = false;
	let encWant = false;
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

	// ── re-encode on every trained chunk ──
	$effect(() => {
		void lab.tick;
		if (lab.phase !== 'ready') return;
		void refreshLatents();
	});

	// once reconstructions clear the quality bar, the map earns its images
	$effect(() => {
		if (lab.trained && !userPicked && mode === 'ink') mode = 'images';
	});

	function pickMode(m: Mode): void {
		userPicked = true;
		mode = m;
	}

	async function refreshLatents(): Promise<void> {
		if (encBusy) {
			encWant = true;
			return;
		}
		const m = lab.mnist;
		if (!m || !lab.engine) return;
		encBusy = true;
		try {
			const n = m.testY.length;
			const { z, d } = await lab.encode(m.testX, n);
			if (d !== latDim) {
				// the bottleneck was rebuilt — screen-space state is stale
				hoverZ = null;
				hoverIdx = -1;
				proj = null;
			}
			latents = z;
			latDim = d;
			visible = binSubsample(z, n, d);
			visibleCount = visible.length;
			const c = new Array(d).fill(0);
			for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) c[k] += z[i * d + k];
			centroidV = c.map((v) => v / n);
			if (!pair) defaultPair();
			requestEye(); // the weights moved — whatever the eye shows is stale
		} catch {
			// engine disposed or rebuilding mid-flight
		}
		encBusy = false;
		if (encWant) {
			encWant = false;
			void refreshLatents();
		}
	}

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

	function walkZ(): number[] {
		const z = latents!;
		const [a, b] = pair!;
		const out: number[] = [];
		for (let k = 0; k < latDim; k++)
			out.push(z[a * latDim + k] + (z[b * latDim + k] - z[a * latDim + k]) * t);
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
	function currentEye(): { z: number[]; src: string } {
		if (latDim === 2 && hoverZ) return { z: hoverZ, src: 'cursor' };
		if (latDim === 3 && hoverIdx >= 0 && latents) {
			const z = latents;
			const i = hoverIdx;
			return { z: [z[3 * i], z[3 * i + 1], z[3 * i + 2]], src: 'nearest digit' };
		}
		if (walkTouched && pair && latents) return { z: walkZ(), src: 'walk' };
		return { z: centroidV, src: 'cloud centre' };
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
		const { z, src } = currentEye();
		try {
			eyePixels = await lab.decode(Float32Array.from(z), 1);
			eyeVersion++;
			eyeLabel = `z = (${z.map(fz).join(', ')}) · ${src}`;
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
		return [
			((ev.clientX - r.left) / r.width) * 2 * EXTENT - EXTENT,
			EXTENT - ((ev.clientY - r.top) / r.height) * 2 * EXTENT
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
			drawScatter();
			drawTile(eyeCanvas, eyeStrip, eyePixels, eyeVersion);
			drawTile(miniA, aStrip, origA, pairVersion);
			drawTile(miniB, bStrip, origB, pairVersion);
		};
		raf = requestAnimationFrame(paint);
		return () => cancelAnimationFrame(raf);
	});

	function atlasFor(
		tk: ReturnType<typeof readTokens>
	): { sheet: HTMLCanvasElement; cols: number } | null {
		const m = lab.mnist;
		if (!m) return null;
		const n = m.testY.length;
		if (mode === 'reveal') {
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

	function drawScatter2(
		ctx: CanvasRenderingContext2D,
		W: number,
		H: number,
		tk: ReturnType<typeof readTokens>
	): void {
		const px = (x: number) => ((x + EXTENT) / (2 * EXTENT)) * W;
		const py = (y: number) => H - ((y + EXTENT) / (2 * EXTENT)) * H;

		// zero axes and tanh's (−1,1)² cage — quiet reference geometry
		ctx.lineWidth = 1;
		ctx.strokeStyle = tk.lineSoft;
		ctx.beginPath();
		ctx.moveTo(0, py(0));
		ctx.lineTo(W, py(0));
		ctx.moveTo(px(0), 0);
		ctx.lineTo(px(0), H);
		ctx.stroke();
		ctx.strokeStyle = tk.line;
		ctx.strokeRect(px(-1), py(1), px(1) - px(-1), py(-1) - py(1));

		const z = latents;
		const m = lab.mnist;
		if (z && m) {
			const n = m.testY.length;
			if (mode === 'ink') {
				ctx.globalAlpha = 0.9;
				ctx.fillStyle = tk.ink3;
				for (let i = 0; i < n; i++) {
					ctx.beginPath();
					ctx.arc(px(z[2 * i]), py(z[2 * i + 1]), 2.5, 0, Math.PI * 2);
					ctx.fill();
				}
				ctx.globalAlpha = 1;
			} else if (visible) {
				const at = atlasFor(tk);
				if (at) {
					ctx.imageSmoothingEnabled = false; // digit pixels stay crisp
					for (let k = 0; k < visible.length; k++) {
						const i = visible[k];
						thumb(ctx, at.sheet, at.cols, i, px(z[2 * i]), py(z[2 * i + 1]), THUMB);
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
				const wz = walkZ();
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
		const toXY = (x: number, y: number, zc: number): [number, number, number] => {
			const [vx, vy, vz] = project3(x * zoom, y * zoom, zc * zoom, yaw, pitch);
			return [((vx + EXTENT3) / (2 * EXTENT3)) * W, H - ((vy + EXTENT3) / (2 * EXTENT3)) * H, vz];
		};

		// the (−1,1)³ cage as a wireframe
		ctx.strokeStyle = tk.line;
		ctx.lineWidth = 1;
		ctx.globalAlpha = 0.55;
		for (const [a, b] of CUBE_EDGES) {
			const [x0, y0] = toXY(CUBE[a][0], CUBE[a][1], CUBE[a][2]);
			const [x1, y1] = toXY(CUBE[b][0], CUBE[b][1], CUBE[b][2]);
			ctx.beginPath();
			ctx.moveTo(x0, y0);
			ctx.lineTo(x1, y1);
			ctx.stroke();
		}
		ctx.globalAlpha = 1;

		const z = latents;
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
		const dn = (i: number) => Math.max(0, Math.min(1, (depthOf(i) + 1.9) / 3.8));

		if (mode === 'ink') {
			const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => depthOf(a) - depthOf(b));
			ctx.fillStyle = tk.ink3;
			for (const i of order) {
				const d = dn(i);
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
				ctx.imageSmoothingEnabled = false;
				for (const i of order) {
					const d = dn(i);
					ctx.globalAlpha = 0.45 + 0.55 * d;
					thumb(ctx, at.sheet, at.cols, i, pr[3 * i], pr[3 * i + 1], 11 + 9 * d);
				}
				ctx.globalAlpha = 1;
			}
		}

		const p = pair;
		if (p) {
			drawSegment(ctx, tk, pr[3 * p[0]], pr[3 * p[0] + 1], pr[3 * p[1]], pr[3 * p[1] + 1]);
			const wz = walkZ();
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

<div class="flex flex-col" use:inview={() => void lab.boot()}>
	{#if lab.phase !== 'ready'}
		<div class="flex min-h-72 flex-col items-center justify-center gap-2 px-6 py-16 text-center">
			<span class="eyebrow">
				{lab.phase === 'error' ? 'the engine stalled — retry it at Plate I' : 'warming up…'}
			</span>
			<p class="max-w-md font-serif text-[15px] text-ink-3 italic">
				This map reads the same network as Plate I. Once it is awake, two thousand held-out digits
				will place themselves here.
			</p>
		</div>
	{:else}
		<!-- controls -->
		<div
			class="flex min-h-12 flex-wrap items-center gap-x-5 gap-y-3 border-b border-line-soft px-4 py-3"
		>
			<span class="flex items-center gap-1" role="group" aria-label="Map view">
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
			<span class="num ml-auto text-[11px] text-ink-3">
				step {lab.step} · val {fmt(lab.valLoss)}
			</span>
			<Btn disabled={lab.rebuilding} onclick={() => lab.setTraining(!lab.training)}>
				{#if lab.training}
					<Pause size={12} aria-hidden="true" /> Pause
				{:else}
					<Play size={12} aria-hidden="true" /> Train
				{/if}
			</Btn>
		</div>

		<!-- the map and the decoder's eye -->
		<div class="grid grid-cols-1 gap-px bg-line-soft md:grid-cols-[minmax(0,1fr)_15rem]">
			<div class="relative bg-surface">
				<span class="eyebrow absolute top-3 left-3 z-10">
					latent space · z = E(x)
					{#if lab.latentDim === 3}
						<span class="tracking-normal text-ink-3 normal-case"
							>— drag to turn · wheel to zoom</span
						>
					{/if}
				</span>
				{#if mode !== 'ink'}
					<span class="num absolute bottom-2 left-3 z-10 text-[10.5px] text-ink-3">
						{visibleCount} of 2000 shown
					</span>
				{/if}
				<canvas
					bind:this={scatterCanvas}
					class="block aspect-square w-full touch-none"
					class:cursor-crosshair={lab.latentDim === 2}
					class:cursor-grab={lab.latentDim === 3 && !dragging}
					class:cursor-grabbing={dragging}
					aria-label="Two thousand held-out digits scattered by their latent coordinates"
					onpointermove={ptrMove}
					onpointerdown={ptrDown}
					onpointerup={ptrUp}
					onpointercancel={ptrUp}
					onpointerleave={ptrLeave}
				></canvas>
			</div>
			<div class="flex flex-col gap-2 bg-surface px-4 py-4">
				<span class="eyebrow">the decoder's eye</span>
				<canvas
					bind:this={eyeCanvas}
					class="mx-auto block aspect-square w-full max-w-56"
					aria-label="The decoder's output for the current latent point"
				></canvas>
				<span class="num self-center text-[11px] text-ink-3">{eyeLabel}</span>
				<p class="mt-auto font-serif text-[13px] text-ink-3 italic">
					{#if lab.latentDim === 2}
						sweep the map — every point of the plane decodes to something digit-shaped, even where
						no digit has ever been
					{:else}
						hover near a digit to decode its coordinates; the walk slider still crosses the space
						between two of them
					{/if}
				</p>
			</div>
		</div>

		<!-- reveal legend -->
		{#if mode === 'reveal'}
			<div
				class="flex flex-wrap items-center gap-x-3.5 gap-y-1 border-t border-line-soft px-4 py-2"
			>
				<span class="eyebrow">true labels — used for the tints only, never for training</span>
				{#each DIGITS as d (d)}
					<span class="num flex items-center gap-1.5 text-[11px]">
						<span
							class="inline-block h-2 w-2 rounded-full"
							style="background: var(--cat-{d})"
							aria-hidden="true"
						></span>{d}
					</span>
				{/each}
			</div>
		{/if}

		<!-- walk the map -->
		<div class="flex flex-wrap items-center gap-x-4 gap-y-3 border-t border-line-soft px-4 py-3">
			<span class="eyebrow">walk the map</span>
			<span class="flex items-center gap-1.5">
				<canvas
					bind:this={miniA}
					class="h-9 w-9 rounded border border-line-soft"
					aria-label="First endpoint digit"
				></canvas>
				<span class="num text-[11px] text-ink-2">{pairLabels[0]}</span>
			</span>
			<span class="min-w-44 flex-1">
				<Slider
					label="interpolation t"
					bind:value={t}
					min={0}
					max={1}
					step={0.01}
					format={(v) => v.toFixed(2)}
				/>
			</span>
			<span class="flex items-center gap-1.5">
				<span class="num text-[11px] text-ink-2">{pairLabels[1]}</span>
				<canvas
					bind:this={miniB}
					class="h-9 w-9 rounded border border-line-soft"
					aria-label="Second endpoint digit"
				></canvas>
			</span>
			<Btn onclick={randomPair}>
				<Shuffle size={12} aria-hidden="true" /> Random pair
			</Btn>
		</div>
	{/if}
</div>

<style>
	.chip {
		font-family: var(--font-sans);
		font-size: 11px;
		font-weight: 500;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		padding: 3px 9px;
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
</style>

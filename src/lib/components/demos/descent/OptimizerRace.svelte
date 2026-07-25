<script lang="ts">
	import { Pause, Play, RotateCcw, StepForward } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import SpeedChips from '$lib/components/ui/SpeedChips.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { heatmapRGBA, parseColor } from '$lib/optim/colormap';
	import { contourPaths } from '$lib/optim/contours';
	import {
		PRESETS,
		computeGrid,
		gradAt,
		normalizedLogLoss,
		presetById,
		type LossGrid,
		type PresetId,
		type Pt
	} from '$lib/optim/landscape';
	import {
		LR_MULT,
		RUNNERS,
		initOptState,
		racerColors,
		stepOptimizer,
		type OptState,
		type OptimizerId
	} from '$lib/optim/optimizers';

	let { onraced }: { onraced?: () => void } = $props();

	const STEP_HZ = 40; // optimizer updates per second at speed ×1
	const MAX_STEPS_FRAME = 16; // speed 'max': per-frame batch, no clock
	const WD = 0.1; // decoupled weight decay — only AdamW feels it
	const DIVERGE_AT = 1e6; // past this a racer is frozen and flagged
	const TRAIL_MAX = 220;
	const MESH = 48; // 3-D wireframe resolution (quads per side)
	const H_SCALE = 0.55; // 3-D terrain relief, world units

	interface TrailPt {
		x: number;
		y: number;
		/** normalized log-loss height, for the 3-D view */
		h: number;
	}

	interface Racer {
		x: number;
		y: number;
		st: OptState;
		trail: TrailPt[];
		loss: number;
		diverged: boolean;
	}

	interface LegendEntry {
		id: OptimizerId;
		label: string;
		token: string;
		loss: number;
		diverged: boolean;
	}

	let presetId = $state<PresetId>('basins');
	let logGamma = $state(-1.3); // γ ≈ 0.05 — everyone lively, nobody diverges
	let running = $state(false);
	let t = $state(0);
	let view = $state<'2d' | '3d'>('2d');
	let speed = $state(1);
	let dragging = $state(false);
	let enabled = $state<Record<OptimizerId, boolean>>({
		gd: true,
		momentum: true,
		adam: true,
		adamw: false,
		lion: false
	});
	let legend = $state<LegendEntry[]>([]);

	const gamma = $derived(Math.pow(10, logGamma));
	const preset = $derived(presetById(presetId));

	let canvas: HTMLCanvasElement;

	// Physics and camera live outside $state: the rAF loop mutates them and
	// publishes snapshots (t, legend) for the DOM to render.
	const racers = {} as Record<OptimizerId, Racer>;
	let start: Pt = { ...presetById('basins').start };
	let dirty = true;
	let raced = false;
	let reducedMotion = false; // read by handlers, never by the template
	let yaw = 0.7;
	let pitch = 0.42;
	let zoom = 1;
	let dragLast: [number, number] | null = null;

	// grids and 3-D height meshes are pure per-preset caches
	const grids: Partial<Record<PresetId, LossGrid>> = {};
	const meshes: Partial<Record<PresetId, Float32Array>> = {};

	function gridFor(id: PresetId): LossGrid {
		return (grids[id] ??= computeGrid(presetById(id)));
	}

	function meshFor(id: PresetId): Float32Array {
		let m = meshes[id];
		if (!m) {
			const p = presetById(id);
			const grid = gridFor(id);
			const n1 = MESH + 1;
			m = new Float32Array(n1 * n1);
			for (let j = 0; j < n1; j++) {
				const y = p.yMin + (j / MESH) * (p.yMax - p.yMin);
				for (let i = 0; i < n1; i++) {
					const x = p.xMin + (i / MESH) * (p.xMax - p.xMin);
					m[j * n1 + i] = normalizedLogLoss(grid, p.f(x, y));
				}
			}
			meshes[id] = m;
		}
		return m;
	}

	// projection scratch, reused every 3-D frame
	const sx3 = new Float32Array((MESH + 1) * (MESH + 1));
	const sy3 = new Float32Array((MESH + 1) * (MESH + 1));
	const sd3 = new Float32Array((MESH + 1) * (MESH + 1));

	function fmtGamma(g: number): string {
		return g.toPrecision(2);
	}

	function fmtLoss(v: number): string {
		return Math.abs(v) >= 1000 ? v.toExponential(1) : v.toPrecision(3);
	}

	function markRaced() {
		if (raced) return;
		raced = true;
		onraced?.();
	}

	function syncLegend() {
		const key = (e: LegendEntry) =>
			e.diverged || !Number.isFinite(e.loss) ? Number.MAX_VALUE : e.loss;
		legend = RUNNERS.filter((r) => enabled[r.id])
			.map((r) => ({
				id: r.id,
				label: r.label,
				token: r.token,
				loss: racers[r.id].loss,
				diverged: racers[r.id].diverged
			}))
			.sort((a, b) => key(a) - key(b));
	}

	function freshRacer(p: Pt): Racer {
		const pr = presetById(presetId);
		const loss = pr.f(p.x, p.y);
		return {
			x: p.x,
			y: p.y,
			st: initOptState(),
			trail: [{ x: p.x, y: p.y, h: normalizedLogLoss(gridFor(presetId), loss) }],
			loss,
			diverged: false
		};
	}

	function resetRacers(p: Pt) {
		for (const spec of RUNNERS) racers[spec.id] = freshRacer(p);
		t = 0;
		syncLegend();
		dirty = true;
	}

	resetRacers(start);

	function stepAll(n: number) {
		const pr = presetById(presetId);
		const grid = gridFor(presetId);
		const lr = Math.pow(10, logGamma);
		for (let k = 0; k < n; k++) {
			for (const spec of RUNNERS) {
				if (!enabled[spec.id]) continue;
				const r = racers[spec.id];
				if (r.diverged) continue;
				const [gx, gy] = gradAt(pr, r.x, r.y);
				const next = stepOptimizer(spec.id, r.x, r.y, gx, gy, r.st, lr * LR_MULT[spec.id], WD);
				const loss = pr.f(next.x, next.y);
				if (
					!Number.isFinite(next.x) ||
					!Number.isFinite(next.y) ||
					!Number.isFinite(loss) ||
					loss > DIVERGE_AT
				) {
					// frozen where it lost its footing — a lesson, not a bug
					r.diverged = true;
					continue;
				}
				r.x = next.x;
				r.y = next.y;
				r.loss = loss;
				r.trail.push({ x: next.x, y: next.y, h: normalizedLogLoss(grid, loss) });
				if (r.trail.length > TRAIL_MAX) r.trail.shift();
			}
		}
		t += n;
		syncLegend();
		dirty = true;
	}

	// auto-start when the plate scrolls near — the house inview convention
	function boot() {
		reducedMotion =
			typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
		if (!reducedMotion) running = true;
	}

	function restartAt(p: Pt) {
		start = p;
		resetRacers(p);
		if (!reducedMotion) running = true;
		markRaced();
	}

	// click, not pointerdown: browsers suppress click after a touch-scroll,
	// so readers can swipe past the canvas without restarting the race
	function onClick(e: MouseEvent) {
		if (view !== '2d') return;
		const rect = canvas.getBoundingClientRect();
		const pr = presetById(presetId);
		restartAt({
			x: pr.xMin + ((e.clientX - rect.left) / rect.width) * (pr.xMax - pr.xMin),
			y: pr.yMax - ((e.clientY - rect.top) / rect.height) * (pr.yMax - pr.yMin)
		});
	}

	function onKeyDown(e: KeyboardEvent) {
		if (view !== '2d') return;
		if (e.key !== 'Enter' && e.key !== ' ') return;
		e.preventDefault();
		const pr = presetById(presetId);
		restartAt({
			x: pr.xMin + (0.1 + 0.8 * Math.random()) * (pr.xMax - pr.xMin),
			y: pr.yMin + (0.1 + 0.8 * Math.random()) * (pr.yMax - pr.yMin)
		});
	}

	// 3-D camera: drag to orbit, wheel to zoom (the SpaceLab conventions)
	function onPointerDown(e: PointerEvent) {
		if (view !== '3d') return;
		dragging = true;
		dragLast = [e.clientX, e.clientY];
		(e.target as Element).setPointerCapture(e.pointerId);
	}

	function onPointerMove(e: PointerEvent) {
		if (!dragging || !dragLast) return;
		yaw += (e.clientX - dragLast[0]) * 0.008;
		pitch = Math.max(0.05, Math.min(1.35, pitch + (e.clientY - dragLast[1]) * 0.006));
		dragLast = [e.clientX, e.clientY];
		dirty = true;
	}

	function onPointerUp() {
		dragging = false;
		dragLast = null;
	}

	function onWheel(e: WheelEvent) {
		if (view !== '3d') return;
		e.preventDefault();
		zoom = Math.min(2.5, Math.max(0.5, zoom * Math.exp(-e.deltaY * 0.0016)));
		dirty = true;
	}

	function selectPreset(id: PresetId) {
		if (id === presetId) return;
		presetId = id;
		start = { ...presetById(id).start };
		resetRacers(start);
	}

	function selectView(v: '2d' | '3d') {
		if (v === view) return;
		view = v;
		dirty = true;
	}

	function toggleRacer(id: OptimizerId) {
		enabled[id] = !enabled[id];
		if (enabled[id]) racers[id] = freshRacer(start);
		syncLegend();
		dirty = true;
	}

	function togglePlay() {
		running = !running;
		if (running) markRaced();
	}

	function stepOnce() {
		running = false;
		stepAll(1);
		markRaced();
	}

	function resetRace() {
		start = { ...presetById(presetId).start };
		resetRacers(start);
	}

	/** Rotate-project a 3-D point (y up); returns [x, y, depth]. Local copy
	 *  of the space chapter's convention so the demos stay decoupled. */
	function project3(
		x: number,
		y: number,
		z: number,
		cy: number,
		sy: number,
		cp: number,
		sp: number
	): [number, number, number] {
		const x1 = cy * x + sy * z;
		const z1 = -sy * x + cy * z;
		return [x1, cp * y - sp * z1, sp * y + cp * z1];
	}

	$effect(() => {
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

		let disposed = false;
		let raf = 0;
		let acc = 0;
		let last = performance.now();
		let tokenKey = '';
		let heatKey = '';
		let contourKey = '';
		let rings: Path2D[] = [];
		const heat = document.createElement('canvas');

		interface Tokens {
			paper: string;
			ink: string;
			hairline: string;
			ink3: string;
			colors: Record<OptimizerId, string>;
		}

		function draw2d(W: number, H: number, tk: Tokens) {
			if (!ctx) return;
			const pr = presetById(presetId);
			const grid = gridFor(presetId);

			const px = (x: number) => ((x - pr.xMin) / (pr.xMax - pr.xMin)) * W;
			const py = (y: number) => H - ((y - pr.yMin) / (pr.yMax - pr.yMin)) * H;

			const hk = `${presetId}|${tk.paper}|${tk.ink}`;
			if (hk !== heatKey) {
				heatKey = hk;
				heat.width = grid.res;
				heat.height = grid.res;
				const hctx = heat.getContext('2d');
				hctx?.putImageData(
					new ImageData(
						heatmapRGBA(grid, parseColor(tk.paper), parseColor(tk.ink)),
						grid.res,
						grid.res
					),
					0,
					0
				);
			}

			const ck = `${presetId}|${W}x${H}`;
			if (ck !== contourKey) {
				contourKey = ck;
				rings = contourPaths(grid, (x, y) => [px(x), py(y)]).map((d) => new Path2D(d));
			}

			ctx.imageSmoothingEnabled = true;
			ctx.drawImage(
				heat,
				px(grid.extXMin),
				py(grid.extYMax),
				px(grid.extXMax) - px(grid.extXMin),
				py(grid.extYMin) - py(grid.extYMax)
			);

			ctx.lineWidth = 1;
			for (let i = 0; i < rings.length; i++) {
				const major = (i + 1) % 4 === 0;
				ctx.strokeStyle = major ? tk.ink3 : tk.hairline;
				ctx.globalAlpha = major ? 0.55 : 1;
				ctx.stroke(rings[i]);
			}
			ctx.globalAlpha = 1;

			// θ₀ — a quiet ring where the walkers were dropped
			ctx.beginPath();
			ctx.arc(px(start.x), py(start.y), 3.5, 0, Math.PI * 2);
			ctx.strokeStyle = tk.ink3;
			ctx.lineWidth = 1.2;
			ctx.stroke();

			ctx.lineCap = 'round';
			for (const spec of RUNNERS) {
				if (!enabled[spec.id]) continue;
				const r = racers[spec.id];
				const c = tk.colors[spec.id];
				const n = r.trail.length;
				ctx.strokeStyle = c;
				ctx.lineWidth = 2.2;
				for (let i = 1; i < n; i++) {
					ctx.globalAlpha = 0.08 + 0.5 * Math.pow(i / (n - 1), 1.5);
					ctx.beginPath();
					ctx.moveTo(px(r.trail[i - 1].x), py(r.trail[i - 1].y));
					ctx.lineTo(px(r.trail[i].x), py(r.trail[i].y));
					ctx.stroke();
				}
				ctx.globalAlpha = r.diverged ? 0.35 : 1;
				ctx.beginPath();
				ctx.arc(px(r.x), py(r.y), 5.5, 0, Math.PI * 2);
				ctx.fillStyle = c;
				ctx.fill();
				ctx.strokeStyle = tk.paper;
				ctx.lineWidth = 1.5;
				ctx.stroke();
			}
			ctx.globalAlpha = 1;
		}

		function draw3d(W: number, H: number, tk: Tokens) {
			if (!ctx) return;
			const pr = presetById(presetId);
			const grid = gridFor(presetId);
			const mesh = meshFor(presetId);
			const spanX = pr.xMax - pr.xMin;
			const spanY = pr.yMax - pr.yMin;
			const AZ = spanY / spanX; // world half-depth; half-width is 1

			const cy = Math.cos(yaw);
			const sy = Math.sin(yaw);
			const cp = Math.cos(pitch);
			const sp = Math.sin(pitch);
			const K = (H / 2.6) * zoom;

			const toScreen = (x: number, y: number, h: number): [number, number, number] => {
				const [vx, vy, vz] = project3(
					((x - pr.xMin) / spanX - 0.5) * 2,
					(h - 0.5) * H_SCALE,
					((y - pr.yMin) / spanY - 0.5) * 2 * AZ,
					cy,
					sy,
					cp,
					sp
				);
				return [W / 2 + vx * K, H * 0.52 - vy * K, vz];
			};

			// project the terrain lattice once, then bucket wireframe segments
			// by mean depth: near strokes dark, far strokes fading out
			const n1 = MESH + 1;
			let dMin = Infinity;
			let dMax = -Infinity;
			for (let j = 0; j < n1; j++) {
				const y = pr.yMin + (j / MESH) * spanY;
				for (let i = 0; i < n1; i++) {
					const x = pr.xMin + (i / MESH) * spanX;
					const k = j * n1 + i;
					const [X, Y, D] = toScreen(x, y, mesh[k]);
					sx3[k] = X;
					sy3[k] = Y;
					sd3[k] = D;
					if (D < dMin) dMin = D;
					if (D > dMax) dMax = D;
				}
			}
			const dSpan = dMax - dMin || 1;

			const BUCKETS = 6;
			const paths = Array.from({ length: BUCKETS }, () => new Path2D());
			const seg = (a: number, b: number) => {
				const q = (0.5 * (sd3[a] + sd3[b]) - dMin) / dSpan; // 1 = nearest
				const p = paths[Math.min(BUCKETS - 1, Math.max(0, Math.round(q * (BUCKETS - 1))))];
				p.moveTo(sx3[a], sy3[a]);
				p.lineTo(sx3[b], sy3[b]);
			};
			for (let j = 0; j < n1; j++) {
				for (let i = 0; i < MESH; i++) {
					seg(j * n1 + i, j * n1 + i + 1); // along x
					if (j < MESH) seg(i * n1 + j, (i + 1) * n1 + j); // along y
				}
			}
			ctx.lineWidth = 1;
			ctx.strokeStyle = tk.hairline;
			for (let k = 0; k < BUCKETS; k++) {
				ctx.globalAlpha = 0.22 + (k / (BUCKETS - 1)) * 0.65;
				ctx.stroke(paths[k]);
			}
			ctx.globalAlpha = 1;

			// trails ride the surface…
			ctx.lineCap = 'round';
			for (const spec of RUNNERS) {
				if (!enabled[spec.id]) continue;
				const r = racers[spec.id];
				ctx.strokeStyle = tk.colors[spec.id];
				ctx.lineWidth = 2.2;
				const n = r.trail.length;
				let prev: [number, number, number] | null = null;
				for (let i = 0; i < n; i++) {
					const p = r.trail[i];
					const cur = toScreen(p.x, p.y, p.h);
					if (prev) {
						ctx.globalAlpha = 0.08 + 0.5 * Math.pow(i / (n - 1), 1.5);
						ctx.beginPath();
						ctx.moveTo(prev[0], prev[1]);
						ctx.lineTo(cur[0], cur[1]);
						ctx.stroke();
					}
					prev = cur;
				}
			}
			ctx.globalAlpha = 1;

			// …then the balls, painted far to near
			const balls = RUNNERS.filter((s) => enabled[s.id]).map((s) => {
				const r = racers[s.id];
				const [X, Y, D] = toScreen(r.x, r.y, normalizedLogLoss(grid, r.loss));
				return { X, Y, D, color: tk.colors[s.id], diverged: r.diverged };
			});
			balls.sort((a, b) => a.D - b.D);
			for (const b of balls) {
				ctx.globalAlpha = b.diverged ? 0.35 : 1;
				ctx.beginPath();
				ctx.arc(b.X, b.Y, 5.5, 0, Math.PI * 2);
				ctx.fillStyle = b.color;
				ctx.fill();
				ctx.strokeStyle = tk.paper;
				ctx.lineWidth = 1.5;
				ctx.stroke();
			}
			ctx.globalAlpha = 1;
		}

		function draw() {
			if (!ctx) return;
			const dpr = Math.min(devicePixelRatio || 1, 2);
			const W = canvas.clientWidth;
			const H = canvas.clientHeight;
			if (W === 0 || H === 0) return;
			if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
				canvas.width = Math.round(W * dpr);
				canvas.height = Math.round(H * dpr);
				dirty = true;
			}

			// tokens are re-read every frame so a theme flip repaints the field
			const style = getComputedStyle(canvas);
			const tk: Tokens = {
				paper: style.getPropertyValue('--paper').trim(),
				ink: style.getPropertyValue('--ink').trim(),
				hairline: style.getPropertyValue('--line').trim(),
				ink3: style.getPropertyValue('--ink-3').trim(),
				colors: racerColors(canvas)
			};
			const key = [tk.paper, tk.ink, tk.hairline, tk.ink3, ...Object.values(tk.colors)].join('|');
			if (key !== tokenKey) {
				tokenKey = key;
				heatKey = '';
				dirty = true;
			}
			if (!dirty) return;
			dirty = false;

			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx.fillStyle = tk.paper;
			ctx.fillRect(0, 0, W, H);
			if (view === '2d') draw2d(W, H, tk);
			else draw3d(W, H, tk);
		}

		function frame(now: number) {
			if (disposed) return;
			const dt = Math.min(now - last, 100);
			last = now;
			if (view === '3d' && !dragging && !reducedMotion) {
				yaw += dt * 0.00013; // idle orbit, same pace as the space chapter
				dirty = true;
			}
			if (running && !reducedMotion) {
				if (speed === 0) {
					stepAll(MAX_STEPS_FRAME);
				} else {
					acc += (dt / 1000) * STEP_HZ * speed;
					const n = Math.floor(acc);
					if (n > 0) {
						acc -= n;
						stepAll(Math.min(n, 8));
					}
				}
			}
			draw();
			raf = requestAnimationFrame(frame);
		}

		raf = requestAnimationFrame(frame);
		return () => {
			disposed = true;
			cancelAnimationFrame(raf);
		};
	});
</script>

<Plate
	n={1}
	title="The descent race"
	caption="Click anywhere to drop the walkers there. Try the saddle's ridge and watch who escapes; raise γ until Adam overshoots. Flip to 3-D and drag to see the terrain the contours were hiding."
>
	{#snippet status()}
		<span>t = {t} · γ = {fmtGamma(gamma)}</span>
	{/snippet}

	<div class="p-4 sm:p-5" use:inview={boot}>
		<div class="mb-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
			<div class="flex flex-wrap items-center gap-x-4 gap-y-2">
				<div class="flex items-center gap-1.5" role="group" aria-label="View">
					<button
						class="chip"
						class:chip-on={view === '2d'}
						aria-pressed={view === '2d'}
						onclick={() => selectView('2d')}
					>
						2-D
					</button>
					<button
						class="chip"
						class:chip-on={view === '3d'}
						aria-pressed={view === '3d'}
						onclick={() => selectView('3d')}
					>
						3-D
					</button>
				</div>
				<div class="flex flex-wrap items-center gap-1.5" role="group" aria-label="Landscape preset">
					{#each PRESETS as p (p.id)}
						<button
							class="chip"
							class:chip-on={presetId === p.id}
							aria-pressed={presetId === p.id}
							onclick={() => selectPreset(p.id)}
						>
							{p.label}
						</button>
					{/each}
				</div>
			</div>
			<div class="flex flex-wrap items-center gap-1.5" role="group" aria-label="Optimizers">
				{#each RUNNERS as r (r.id)}
					<button
						class="chip"
						class:chip-on={enabled[r.id]}
						aria-pressed={enabled[r.id]}
						title="{r.blurb} — stride ×{LR_MULT[r.id]}"
						onclick={() => toggleRacer(r.id)}
					>
						<span
							class="dot"
							style="border-color: var({r.token}); background: {enabled[r.id]
								? `var(${r.token})`
								: 'transparent'};"
						></span>
						{r.label}
					</button>
				{/each}
			</div>
		</div>

		<canvas
			bind:this={canvas}
			class="block w-full rounded-md border border-line-soft"
			class:cursor-crosshair={view === '2d'}
			class:cursor-grab={view === '3d' && !dragging}
			class:cursor-grabbing={dragging}
			style="aspect-ratio: {preset.xMax - preset.xMin} / {preset.yMax -
				preset.yMin}; touch-action: {view === '3d' ? 'none' : 'manipulation'};"
			role="button"
			tabindex="0"
			aria-label="Loss landscape. In 2-D, click anywhere to drop the walkers there and restart the race; in 3-D, drag to rotate the surface."
			onclick={onClick}
			onkeydown={onKeyDown}
			onpointerdown={onPointerDown}
			onpointermove={onPointerMove}
			onpointerup={onPointerUp}
			onpointercancel={onPointerUp}
			onwheel={onWheel}
		></canvas>

		<div
			class="num mt-2.5 flex min-h-5 flex-wrap items-baseline gap-x-5 gap-y-1 text-[11px] text-ink-3"
		>
			{#each legend as e (e.id)}
				<span class="inline-flex items-baseline gap-1.5">
					<span
						class="inline-block h-2 w-2 self-center rounded-full"
						style="background: var({e.token});"
					></span>
					<span class="text-ink-2">{e.label}</span>
					{#if e.diverged}
						<span style="color: var(--bad);">diverged</span>
					{:else}
						<span>{fmtLoss(e.loss)}</span>
					{/if}
				</span>
			{/each}
			{#if view === '3d'}
				<span class="ml-auto">drag to turn · wheel to zoom · click-to-drop lives in 2-D</span>
			{/if}
		</div>

		<div class="mt-4 flex flex-wrap items-center gap-x-6 gap-y-3">
			<div class="flex items-center gap-2">
				<Btn kind="primary" onclick={togglePlay}>
					{#if running}
						<Pause size={13} aria-hidden="true" /> Pause
					{:else}
						<Play size={13} aria-hidden="true" /> Play
					{/if}
				</Btn>
				<Btn onclick={stepOnce}><StepForward size={13} aria-hidden="true" /> Step</Btn>
				<Btn onclick={resetRace}><RotateCcw size={13} aria-hidden="true" /> Reset</Btn>
			</div>
			<SpeedChips bind:value={speed} />
			<div class="min-w-56 flex-1">
				<Slider
					label="learning rate γ"
					bind:value={logGamma}
					min={-2.5}
					max={-0.3}
					step={0.01}
					format={(v) => fmtGamma(Math.pow(10, v))}
					tone="accent"
				/>
			</div>
		</div>
	</div>
</Plate>

<style>
	.chip {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-family: var(--font-sans);
		font-size: 10.5px;
		font-weight: 550;
		letter-spacing: 0.09em;
		text-transform: uppercase;
		color: var(--ink-3);
		background: transparent;
		border: 1px solid var(--line-soft);
		border-radius: 999px;
		padding: 3px 10px;
		cursor: pointer;
		transition:
			color 120ms ease,
			border-color 120ms ease,
			background 120ms ease;
	}
	.chip:hover {
		color: var(--ink-2);
		border-color: var(--line);
	}
	.chip-on {
		color: var(--ink);
		border-color: var(--line);
		background: var(--surface-2);
	}
	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		border: 1.5px solid;
		flex: none;
	}
</style>

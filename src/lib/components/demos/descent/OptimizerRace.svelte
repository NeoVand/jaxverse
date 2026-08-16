<script lang="ts">
	import { Pause, Play, RotateCcw, StepForward } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
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

	const STEP_HZ = 40; // optimizer updates per second
	const WD = 0.1; // decoupled weight decay — only AdamW feels it
	const DIVERGE_AT = 1e6; // past this a racer is frozen and flagged
	const TRAIL_MAX = 220;
	const MESH = 160; // 3-D surface resolution (quads per side)
	const H_SCALE = 0.55; // 3-D terrain relief, world units
	// Static world-space light for the 3-D surface — high and from the left,
	// nearly unit length, so flats stay airy and walls collect ink.
	const LIGHT: [number, number, number] = [-0.4, 0.8, 0.45];
	const SHADE_LEVELS = 256; // paper→ink LUT resolution for surface fills

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
	let pitch = 0.55;
	let zoom = 1;
	let dragLast: [number, number] | null = null;

	// grids, 3-D height meshes, and per-quad shade indices: per-preset caches
	const grids: Partial<Record<PresetId, LossGrid>> = {};
	const meshes: Partial<Record<PresetId, Float32Array>> = {};
	const shades: Partial<Record<PresetId, Uint8Array>> = {};

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

	/** Per-quad LUT index into the paper→ink ramp: the exact log-loss shading
	 *  the 2-D heatmap uses (basins collect ink), deepened by a Lambert term
	 *  from the surface normal so the relief reads in both themes. Static per
	 *  preset — terrain and light never move relative to each other. */
	function shadeFor(id: PresetId): Uint8Array {
		let s = shades[id];
		if (!s) {
			const p = presetById(id);
			const mesh = meshFor(id);
			const n1 = MESH + 1;
			const AZ = (p.yMax - p.yMin) / (p.xMax - p.xMin);
			const dX = 2 / MESH; // world cell size along x (half-width 1)
			const dZ = (2 * AZ) / MESH;
			s = new Uint8Array(MESH * MESH);
			for (let j = 0; j < MESH; j++) {
				for (let i = 0; i < MESH; i++) {
					const k00 = j * n1 + i;
					const h00 = mesh[k00];
					const h10 = mesh[k00 + 1];
					const h01 = mesh[k00 + n1];
					const h11 = mesh[k00 + n1 + 1];
					// loss term: same direction as the heatmap, deeper for 3-D
					const t = (h00 + h10 + h01 + h11) / 4;
					const wLoss = 0.58 * Math.pow(1 - t, 1.25);
					// Lambert term from the world-space normal of this quad
					const dhdx = (((h10 + h11 - h00 - h01) / 2) * H_SCALE) / dX;
					const dhdz = (((h01 + h11 - h00 - h10) / 2) * H_SCALE) / dZ;
					const inv = 1 / Math.hypot(dhdx, 1, dhdz);
					const lambert = Math.max(0, (-dhdx * LIGHT[0] + LIGHT[1] - dhdz * LIGHT[2]) * inv);
					const w = Math.min(1, 0.05 + wLoss + 0.24 * (1 - lambert));
					s[j * MESH + i] = Math.round(w * (SHADE_LEVELS - 1));
				}
			}
			shades[id] = s;
		}
		return s;
	}

	// projected-lattice scratch, reused every 3-D frame
	const sx3 = new Float32Array((MESH + 1) * (MESH + 1));
	const sy3 = new Float32Array((MESH + 1) * (MESH + 1));

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

	// Fixed RUNNERS order — re-sorting by loss made entries swap places every
	// few frames, which read as flicker rather than a ranking.
	function syncLegend() {
		legend = RUNNERS.filter((r) => enabled[r.id]).map((r) => ({
			id: r.id,
			label: r.label,
			token: r.token,
			loss: racers[r.id].loss,
			diverged: racers[r.id].diverged
		}));
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

	// inview still boots reduced-motion; Play is what starts the race, so a
	// click can drop the walkers without them running off before 3-D.
	function boot() {
		reducedMotion =
			typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
	}

	function restartAt(p: Pt) {
		start = p;
		resetRacers(p);
		running = false;
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
		running = false;
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
		let lutKey = '';
		let lut: string[] = [];
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

			// paper→ink fill ramp, rebuilt only on theme flips
			const lk = `${tk.paper}|${tk.ink}`;
			if (lk !== lutKey) {
				lutKey = lk;
				const paper = parseColor(tk.paper);
				const ink = parseColor(tk.ink);
				lut = Array.from({ length: SHADE_LEVELS }, (_, i) => {
					const w = i / (SHADE_LEVELS - 1);
					const r = Math.round(paper[0] + (ink[0] - paper[0]) * w);
					const g = Math.round(paper[1] + (ink[1] - paper[1]) * w);
					const b = Math.round(paper[2] + (ink[2] - paper[2]) * w);
					return `rgb(${r},${g},${b})`;
				});
			}

			// project the terrain lattice once…
			const n1 = MESH + 1;
			for (let j = 0; j < n1; j++) {
				const y = pr.yMin + (j / MESH) * spanY;
				for (let i = 0; i < n1; i++) {
					const x = pr.xMin + (i / MESH) * spanX;
					const k = j * n1 + i;
					const [X, Y] = toScreen(x, y, mesh[k]);
					sx3[k] = X;
					sy3[k] = Y;
				}
			}

			// …then paint quads far to near, each filled from the shade LUT.
			// A heightfield never overhangs, so a per-axis back-to-front sweep
			// (loop directions from the yaw's quadrant: depth toward the viewer
			// grows with −sin(yaw)·x and cos(yaw)·y) is an exact painter's
			// order — no per-frame depth sort needed. Each quad is inflated
			// half a pixel from its centroid so adjacent fills overlap and
			// antialiasing never opens paper-colored seams.
			const shade = shadeFor(presetId);
			const iAsc = -sy >= 0;
			const jAsc = cy >= 0;
			const GROW = 0.5; // px pushed outward from the quad centroid
			let lastShade = -1;
			for (let jj = 0; jj < MESH; jj++) {
				const j = jAsc ? jj : MESH - 1 - jj;
				for (let ii = 0; ii < MESH; ii++) {
					const i = iAsc ? ii : MESH - 1 - ii;
					const k00 = j * n1 + i;
					const k10 = k00 + 1;
					const k01 = k00 + n1;
					const k11 = k01 + 1;
					const ci = shade[j * MESH + i];
					if (ci !== lastShade) {
						lastShade = ci;
						ctx.fillStyle = lut[ci];
					}
					const mx = (sx3[k00] + sx3[k10] + sx3[k01] + sx3[k11]) / 4;
					const my = (sy3[k00] + sy3[k10] + sy3[k01] + sy3[k11]) / 4;
					ctx.beginPath();
					for (let c = 0; c < 4; c++) {
						const k = c === 0 ? k00 : c === 1 ? k10 : c === 2 ? k11 : k01;
						const dx = sx3[k] - mx;
						const dy = sy3[k] - my;
						const g = 1 + GROW / (Math.hypot(dx, dy) || 1);
						if (c === 0) ctx.moveTo(mx + dx * g, my + dy * g);
						else ctx.lineTo(mx + dx * g, my + dy * g);
					}
					ctx.closePath();
					ctx.fill();
				}
			}

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
				acc += (dt / 1000) * STEP_HZ;
				const n = Math.floor(acc);
				if (n > 0) {
					acc -= n;
					stepAll(Math.min(n, 8));
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
	id="race"
	live
	title="The descent race"
	caption="Five walkers, one landscape, the same gradient at every point — only the rule for taking the step differs. Click anywhere to drop them there together; they wait for Play, so you can flip to 3-D and look before they move. On the saddle's ridge, plain descent stalls where the ground flattens, momentum coasts through on stored velocity, and Adam's per-coordinate stride shrugs the geometry off. Raise γ until somebody overshoots; drag the 3-D view to see the terrain the contours were hiding."
>
	{#snippet status()}
		<span>t = {t} · γ = {fmtGamma(gamma)}</span>
	{/snippet}

	{#snippet actions()}
		<Btn kind="primary" onclick={togglePlay}>
			{#if running}
				<Pause size={13} aria-hidden="true" /> Pause
			{:else}
				<Play size={13} aria-hidden="true" /> Play
			{/if}
		</Btn>
		<Btn onclick={stepOnce}><StepForward size={13} aria-hidden="true" /> Step</Btn>
		<Btn onclick={resetRace}><RotateCcw size={13} aria-hidden="true" /> Reset</Btn>
	{/snippet}

	<div class="p-4 sm:p-5" use:inview={boot}>
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
			aria-label="Loss landscape. In 2-D, click anywhere to drop the walkers there; they wait until Play. In 3-D, drag to rotate the surface."
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
		</div>

		<!-- the bench: view, terrain, racers, stride — the smaller choices, below the stage -->
		<div class="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
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
			<div class="min-w-56 flex-1">
				<Slider
					label="learning rate γ"
					bind:value={logGamma}
					min={-2.5}
					max={-0.3}
					step={0.01}
					format={(v) => fmtGamma(Math.pow(10, v))}
					tone="knob"
				/>
			</div>
		</div>
	</div>
</Plate>

<style>
	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		border: 1.5px solid;
		flex: none;
	}
</style>

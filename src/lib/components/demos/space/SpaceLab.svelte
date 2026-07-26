<script lang="ts">
	// The Bending Space instrument: input space on the left (what the network
	// sees), hidden space on the right (what the network makes of it). One
	// engine, one snapshot/lerp pipeline, three hidden-view modes:
	//   width 2 → the warped plane (colah's picture, live)
	//   width 3 → a slowly turning 3-D projection with the separating plane
	//   wider   → a PCA shadow, honestly labeled
	import { onDestroy } from 'svelte';
	import { Play, Pause, RotateCcw } from 'lucide-svelte';
	import { MlpEngine } from '$lib/nn/mlp-engine';
	import type { Activation, LayerWeights } from '$lib/nn/engine';
	import { makeDataset2d, DATASET_LABELS, type Dataset2dId } from '$lib/nn/datasets2d';
	import ArchDiagram, { type NodeRef } from '$lib/components/ui/ArchDiagram.svelte';
	import {
		makeGridLines,
		makeProbeGrid,
		pcaProject,
		project3,
		planeCubePolygon,
		readTokens,
		hexRgb
	} from './common';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import { inview } from '$lib/components/ui/inview';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { progress } from '$lib/data/progress.svelte';

	interface Props {
		variant: 'guided' | 'free';
		n?: number;
		title: string;
		caption?: string;
	}
	let { variant, n, title, caption }: Props = $props();

	// ── configuration state ──
	// variant never changes after mount — reading it for initial values is deliberate
	// svelte-ignore state_referenced_locally
	let dataset = $state<Dataset2dId>(variant === 'guided' ? 'circles' : 'spirals');
	// svelte-ignore state_referenced_locally
	let width = $state(variant === 'guided' ? 2 : 8);
	// svelte-ignore state_referenced_locally
	let depth = $state(variant === 'guided' ? 1 : 2);
	// gelu by default: tanh saturates and crawls, gelu bends nearly as smoothly
	// and trains in a fraction of the steps.
	let activation = $state<Activation>('gelu');
	const ACTS: Activation[] = ['tanh', 'relu', 'gelu', 'silu'];
	// guided runs a touch gentler so the warp stays legible before saturation
	// svelte-ignore state_referenced_locally
	let logLr = $state(variant === 'guided' ? -2.35 : -2.1);

	// ── run state ──
	let phase = $state<'idle' | 'loading' | 'ready' | 'error'>('idle');
	let training = $state(false);
	let step = $state(0);
	let lossNow = $state(NaN);
	let accNow = $state(NaN);
	let msPerStep = $state(0);
	let device = $state('');
	let paramCount = $state(0);
	let errorMsg = $state('');
	let stage = $state<'flat' | 'lifted' | 'done'>('flat'); // guided narrative
	let selected = $state<number | null>(null);
	let lossHist = $state<number[]>([]);
	let accHist = $state<number[]>([]);
	let weightsView = $state<LayerWeights[] | null>(null);
	let hovered = $state<NodeRef | null>(null);

	const hiddenMode = $derived(width === 2 ? '2d' : width === 3 ? '3d' : 'pca');
	const archLabel = $derived(`2 → ${Array(depth).fill(width).join(' → ')} → 2`);

	// ── non-reactive machinery ──
	let engine: MlpEngine | null = null;
	let gen = 0; // arch generation — cancels stale loops
	// non-reactive holder seeded from the initial dataset; switchDataset reassigns it
	// svelte-ignore state_referenced_locally
	let data = makeDataset2d(dataset, 400, 0.06, 7);

	// Dataset thumbnails: a miniature of each tangle, drawn once, picked by eye.
	// svelte-ignore state_referenced_locally
	const thumbs =
		variant === 'free'
			? (Object.entries(DATASET_LABELS) as Array<[Dataset2dId, string]>).map(([id, label]) => {
					const d = makeDataset2d(id, 90, 0.05, 11);
					const pts: Array<[number, number, number]> = [];
					for (let i = 0; i < d.n; i++) {
						const x = ((d.x[2 * i] + 1.15) / 2.3) * 40;
						const y = 40 - ((d.x[2 * i + 1] + 1.15) / 2.3) * 40;
						pts.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10, d.labels[i]]);
					}
					return { id, label, pts };
				})
			: [];
	const PROBE_RES = 96;
	// probe extents follow the input canvas's aspect so the wash fills it edge to edge
	let probeExt = { x: 1.15, y: 1.15 };
	let probeGrid = makeProbeGrid(PROBE_RES, probeExt.x, probeExt.y);
	// dense sampling along each line keeps the warped grid smooth, not polygonal
	const grid = makeGridLines(16, 65, 1.1);
	let inputCanvas: HTMLCanvasElement | undefined = $state();
	let hiddenCanvas: HTMLCanvasElement | undefined = $state();
	let raf = 0;
	let yaw = 0.7;
	let pitch = 0.42;
	let zoom = 1; // wheel-adjustable view scale (3-D starts wider — see boot)
	let dragging = $state(false);
	let lastDraw = 0;
	// fold ∈ [0, 1]: 0 = the untouched flat sheet, 1 = the network's deformation.
	// The button animates between the two so the folding itself becomes visible.
	let foldTarget = $state(1);
	let foldAnim = { from: 1, to: 1, t0: 0 };
	let foldCur = 1;
	const reduced =
		typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

	interface Snap {
		t: number;
		mode: '2d' | '3d' | 'pca';
		hDim: number;
		ptsV: Float32Array; // n × hDim view coords
		gridV: Float32Array; // g × hDim
		sepN: number[];
		sepC: number;
		prob: Float32Array | null;
		scale: number;
	}
	let snapA: Snap | null = null;
	let snapB: Snap | null = null;
	let offscreen: HTMLCanvasElement | null = null;

	// ── engine lifecycle ──
	// auto-boot when the plate scrolls near (use:inview) — never on mount
	function autoload() {
		if (phase === 'idle') void boot();
	}

	async function boot() {
		phase = 'loading';
		errorMsg = '';
		zoom = hiddenMode === '2d' ? 1 : 0.72; // turning views start wide enough to see the cube
		const myGen = ++gen;
		try {
			engine?.dispose();
			engine = new MlpEngine();
			const layers = [2, ...Array(depth).fill(width), 2];
			await engine.init(
				{
					layers,
					activation,
					loss: 'xent',
					lr: 10 ** logLr,
					batchSize: 256,
					seed: 7,
					valFraction: 0.15
				},
				{ x: data.x, y: data.labels, n: data.n }
			);
			if (myGen !== gen) return;
			device = engine.device;
			paramCount = engine.paramCount;
			step = 0;
			lossHist = [];
			accHist = [];
			snapA = snapB = null;
			hovered = null;
			foldTarget = 1;
			foldAnim = { from: 1, to: 1, t0: 0 };
			foldCur = 1;
			phase = 'ready';
			startPainter();
			await refresh();
			if (variant === 'guided') setTraining(true);
		} catch (err) {
			if (myGen !== gen) return;
			phase = 'error';
			errorMsg = err instanceof Error ? err.message : String(err);
		}
	}

	async function rebuild() {
		const wasTraining = training;
		training = false;
		if (phase !== 'idle') {
			await boot();
			if (wasTraining && variant === 'free') setTraining(true);
		}
	}

	function setTraining(on: boolean) {
		if (on && !training && phase === 'ready') {
			training = true;
			void trainLoop();
		} else {
			training = false;
		}
	}

	async function trainLoop() {
		const myGen = gen;
		while (training && engine && myGen === gen) {
			try {
				await engine.train(
					40,
					(m) => {
						step = m.step;
						lossNow = m.loss;
						msPerStep = msPerStep ? msPerStep * 0.7 + m.stepMs * 0.3 : m.stepMs;
						lossHist = [...lossHist.slice(-239), m.loss];
					},
					8
				);
				if (myGen !== gen || !engine) return;
				const ev = await engine.eval();
				accNow = ev.accuracy ?? NaN;
				accHist = [...accHist.slice(-239), accNow];
				await refresh();
				guidedBeats();
			} catch {
				return; // engine disposed mid-flight
			}
		}
	}

	function guidedBeats() {
		if (variant !== 'guided') return;
		if (stage === 'lifted' && accNow >= 0.97) {
			stage = 'done';
			progress.reach('space:untangled');
		}
	}

	// the γ slider acts on the LIVE run, debounced (Adam moments reset — honest cost)
	let lrTimer: ReturnType<typeof setTimeout> | undefined;
	$effect(() => {
		const lr = 10 ** logLr;
		if (phase !== 'ready' || !engine) return;
		clearTimeout(lrTimer);
		lrTimer = setTimeout(() => {
			engine?.setLr(lr).catch(() => {});
		}, 250);
	});

	// guided variant: the hidden space is either the plane (width 2) or 3-D (width 3)
	async function setGuidedWidth(w: 2 | 3) {
		if (width === w) return;
		width = w;
		stage = w === 2 ? 'flat' : 'lifted';
		await rebuild();
	}

	async function resetWeights() {
		if (!engine) return;
		const wasTraining = training;
		training = false;
		await engine.reset(Math.floor(Math.random() * 1e9));
		step = 0;
		lossHist = [];
		accHist = [];
		if (variant === 'guided') stage = width === 2 ? 'flat' : 'lifted';
		await refresh();
		if (wasTraining) setTraining(true);
	}

	function switchActivation(a: Activation) {
		if (activation === a) return;
		activation = a;
		// a fresh bend restarts the guided story at the current width's beat
		if (variant === 'guided') stage = width === 2 ? 'flat' : 'lifted';
		void rebuild();
	}

	async function switchDataset(id: Dataset2dId) {
		if (dataset === id) return;
		dataset = id;
		data = makeDataset2d(id, 400, 0.06, 7);
		if (engine && phase === 'ready') {
			await engine.setData({ x: data.x, y: data.labels, n: data.n });
			await resetWeights();
		}
	}

	onDestroy(() => {
		// also runs after server prerender — keep it browser-safe
		gen++;
		training = false;
		if (typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(raf);
		engine?.dispose();
	});

	// ── snapshots ──
	async function refresh() {
		if (!engine || phase !== 'ready') return;
		const myGen = gen;
		const n = data.n;
		const g = grid.n;
		const combined = new Float32Array((n + g) * 2);
		combined.set(data.x, 0);
		combined.set(grid.verts, n * 2);

		const [probLogits, acts, w] = await Promise.all([
			engine.predict(probeGrid, PROBE_RES * PROBE_RES, 1024),
			engine.activations(combined, n + g, 1024),
			engine.weights()
		]);
		if (myGen !== gen) return;
		weightsView = w;

		const L = acts.layers.length;
		const hDim = acts.widths[L - 2];
		const hAll = acts.layers[L - 2];

		// separator in hidden coordinates: (w·h + b) class-0 minus class-1
		const wl = w[w.length - 1];
		const sepN: number[] = [];
		for (let k = 0; k < hDim; k++) sepN.push(wl.w[k * 2] - wl.w[k * 2 + 1]);
		const sepC = wl.b[0] - wl.b[1];

		const prob = new Float32Array(PROBE_RES * PROBE_RES);
		for (let i = 0; i < prob.length; i++) {
			const d = probLogits[2 * i + 1] - probLogits[2 * i];
			prob[i] = 1 / (1 + Math.exp(-d));
		}

		let mode: Snap['mode'] = hDim === 2 ? '2d' : hDim === 3 ? '3d' : 'pca';
		let ptsV: Float32Array;
		let gridV: Float32Array;
		let outDim = hDim;
		if (mode === 'pca') {
			// top-3 components so the shadow can be turned like the true 3-D view
			const proj = pcaProject(hAll, n + g, hDim, 3);
			ptsV = proj.slice(0, n * 3);
			gridV = proj.slice(n * 3);
			outDim = 3;
		} else {
			ptsV = hAll.slice(0, n * hDim);
			gridV = hAll.slice(n * hDim);
		}
		// view scale: tanh is bounded; the unbounded bends and pca must be fit
		let scale = 1;
		if (activation !== 'tanh' || mode === 'pca') {
			let mx = 1e-6;
			for (let i = 0; i < ptsV.length; i++) mx = Math.max(mx, Math.abs(ptsV[i]));
			scale = 1 / (mx * 1.1);
		}

		snapA = snapB;
		snapB = { t: performance.now(), mode, hDim: outDim, ptsV, gridV, sepN, sepC, prob, scale };
	}

	// ── painting ──
	function startPainter() {
		cancelAnimationFrame(raf);
		const paint = (t: number) => {
			raf = requestAnimationFrame(paint);
			const dt = lastDraw ? Math.min(50, t - lastDraw) : 16;
			lastDraw = t;
			if (!reduced && !dragging && snapB && snapB.mode !== '2d') yaw += dt * 0.00013;
			drawInput();
			drawHidden(t);
		};
		raf = requestAnimationFrame(paint);
	}

	function setupCanvas(canvas: HTMLCanvasElement) {
		const dpr = Math.min(devicePixelRatio || 1, 2);
		const W = canvas.clientWidth;
		const H = canvas.clientHeight;
		if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
			canvas.width = W * dpr;
			canvas.height = H * dpr;
		}
		const ctx = canvas.getContext('2d')!;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		return { ctx, W, H };
	}

	/** Aspect-true extents: the short axis spans [-E, E]; the long axis shows more plane. */
	function viewExtents(E: number, W: number, H: number): [number, number] {
		const base = Math.min(W, H) || 1;
		return [(E * W) / base, (E * H) / base];
	}

	function drawInput() {
		if (!inputCanvas) return;
		const { ctx, W, H } = setupCanvas(inputCanvas);
		const tk = readTokens(inputCanvas);
		ctx.clearRect(0, 0, W, H);
		const [Ex, Ey] = viewExtents(1.15, W, H);
		const px = (x: number) => ((x + Ex) / (2 * Ex)) * W;
		const py = (y: number) => H - ((y + Ey) / (2 * Ey)) * H;

		// keep the decision wash covering the whole (possibly non-square) canvas
		if (Math.abs(Ex - probeExt.x) > 0.01 || Math.abs(Ey - probeExt.y) > 0.01) {
			probeExt = { x: Ex, y: Ey };
			probeGrid = makeProbeGrid(PROBE_RES, Ex, Ey);
			if (!training && phase === 'ready') void refresh();
		}

		// decision field — soft, low-contrast wash
		const snap = snapB;
		if (snap?.prob) {
			if (!offscreen) {
				offscreen = document.createElement('canvas');
				offscreen.width = PROBE_RES;
				offscreen.height = PROBE_RES;
			}
			const octx = offscreen.getContext('2d')!;
			const img = octx.createImageData(PROBE_RES, PROBE_RES);
			const ca = hexRgb(tk.accent);
			const cw = hexRgb(tk.warm);
			// prob index k = j·res + i with y ascending; canvas rows grow downward
			for (let j = 0; j < PROBE_RES; j++) {
				for (let i = 0; i < PROBE_RES; i++) {
					const p = snap.prob[j * PROBE_RES + i];
					const conf = Math.abs(p - 0.5) * 2;
					const c = p > 0.5 ? cw : ca;
					const k = 4 * ((PROBE_RES - 1 - j) * PROBE_RES + i);
					img.data[k] = c[0];
					img.data[k + 1] = c[1];
					img.data[k + 2] = c[2];
					img.data[k + 3] = Math.round(conf * 46);
				}
			}
			octx.putImageData(img, 0, 0);
			ctx.imageSmoothingEnabled = true;
			ctx.imageSmoothingQuality = 'high';
			ctx.drawImage(offscreen, 0, 0, W, H);
		}

		// points
		const n = data.n;
		for (let i = 0; i < n; i++) {
			const x = px(data.x[2 * i]);
			const y = py(data.x[2 * i + 1]);
			ctx.beginPath();
			ctx.arc(x, y, 3, 0, Math.PI * 2);
			ctx.fillStyle = data.labels[i] === 0 ? tk.accent : tk.warm;
			ctx.fill();
		}
		if (selected !== null)
			ring(ctx, px(data.x[2 * selected]), py(data.x[2 * selected + 1]), tk.ink);
	}

	function ring(ctx: CanvasRenderingContext2D, x: number, y: number, color: string) {
		ctx.beginPath();
		ctx.arc(x, y, 6.5, 0, Math.PI * 2);
		ctx.strokeStyle = color;
		ctx.lineWidth = 1.5;
		ctx.stroke();
	}

	const ease = (u: number) => 1 - (1 - u) * (1 - u) * (1 - u);
	const easeInOut = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);

	function toggleFold() {
		const to = foldTarget === 1 ? 0 : 1;
		foldTarget = to;
		foldAnim = { from: foldCur, to, t0: performance.now() };
	}

	/** Current fold factor — eased between button presses, honored by every frame. */
	function foldAt(t: number): number {
		const { from, to, t0 } = foldAnim;
		const dur = reduced ? 1 : 1400;
		const u = Math.min(1, Math.max(0, (t - t0) / dur));
		foldCur = from + (to - from) * easeInOut(u);
		return foldCur;
	}

	/** Interpolated view coords for index i of pts/grid arrays. */
	function lerpAt(
		a: Float32Array | null,
		b: Float32Array,
		i: number,
		d: number,
		u: number,
		out: number[]
	) {
		for (let k = 0; k < d; k++) {
			const vb = b[i * d + k];
			out[k] = a && a.length === b.length ? a[i * d + k] + (vb - a[i * d + k]) * u : vb;
		}
	}

	function drawHidden(t: number) {
		if (!hiddenCanvas) return;
		const { ctx, W, H } = setupCanvas(hiddenCanvas);
		const tk = readTokens(hiddenCanvas);
		ctx.clearRect(0, 0, W, H);
		const snap = snapB;
		if (!snap) return;
		const prev = snapA && snapA.mode === snap.mode && snapA.hDim === snap.hDim ? snapA : null;
		const u = prev ? ease(Math.min(1, (t - snap.t) / 380)) : 1;
		const S = snap.scale;
		const [Ex, Ey] = viewExtents(1.18 / zoom, W, H); // wheel zoom widens or narrows the window
		const d = snap.hDim;
		const turns = snap.mode !== '2d'; // both 3-D and the PCA shadow rotate

		// fold factor: pull every vertex back toward its input-plane position
		const f = foldAt(t);
		const blendFlat = (v: number[], gx: number, gy: number) => {
			v[0] = (gx / S) * (1 - f) + v[0] * f;
			v[1] = (gy / S) * (1 - f) + v[1] * f;
			for (let k = 2; k < d; k++) v[k] *= f;
		};

		const toXY = (v: number[]): [number, number, number] => {
			if (turns) {
				const [x, y, z] = project3(v[0] * S, v[1] * S, v[2] * S, yaw, pitch);
				return [((x + Ex) / (2 * Ex)) * W, H - ((y + Ey) / (2 * Ey)) * H, z];
			}
			return [((v[0] * S + Ex) / (2 * Ex)) * W, H - ((v[1] * S + Ey) / (2 * Ey)) * H, 0];
		};

		// the warped reference grid
		const tmp: number[] = [0, 0, 0];
		ctx.lineWidth = 1;
		for (const [start, count] of grid.lines) {
			ctx.beginPath();
			for (let s = 0; s < count; s++) {
				lerpAt(prev?.gridV ?? null, snap.gridV, start + s, d, u, tmp);
				if (f < 1) blendFlat(tmp, grid.verts[(start + s) * 2], grid.verts[(start + s) * 2 + 1]);
				const [x, y] = toXY(tmp);
				if (s === 0) ctx.moveTo(x, y);
				else ctx.lineTo(x, y);
			}
			ctx.strokeStyle = tk.ink3;
			ctx.globalAlpha = turns ? 0.45 : 0.6;
			ctx.stroke();
		}
		ctx.globalAlpha = 1;

		// the separator: a straight line (2-D) or a flat plane (3-D).
		// It lives in hidden coordinates, so it fades away as the space unfolds.
		if (f < 0.02) {
			// fully unfolded — nothing but the flat sheet and the points
		} else if (snap.mode === '2d') {
			const [a, b] = snap.sepN;
			const c = snap.sepC;
			// n·h + c = 0 → draw across the view box (hidden coords, scaled by S)
			const ptsLine: Array<[number, number]> = [];
			const limX = Ex / S;
			const limY = Ey / S;
			for (const xx of [-limX, limX]) if (Math.abs(b) > 1e-9) ptsLine.push([xx, (-c - a * xx) / b]);
			for (const yy of [-limY, limY]) if (Math.abs(a) > 1e-9) ptsLine.push([(-c - b * yy) / a, yy]);
			const inBox = ptsLine.filter(
				([x, y]) => Math.abs(x) <= limX * 1.01 && Math.abs(y) <= limY * 1.01
			);
			if (inBox.length >= 2) {
				// take the two most distant crossings (corners can duplicate)
				let p0 = inBox[0];
				let p1 = inBox[1];
				let best = -1;
				for (let a = 0; a < inBox.length; a++)
					for (let b = a + 1; b < inBox.length; b++) {
						const dd = (inBox[a][0] - inBox[b][0]) ** 2 + (inBox[a][1] - inBox[b][1]) ** 2;
						if (dd > best) {
							best = dd;
							p0 = inBox[a];
							p1 = inBox[b];
						}
					}
				const [x0, y0] = toXY([p0[0], p0[1], 0]);
				const [x1, y1] = toXY([p1[0], p1[1], 0]);
				ctx.beginPath();
				ctx.moveTo(x0, y0);
				ctx.lineTo(x1, y1);
				ctx.strokeStyle = tk.ink;
				ctx.lineWidth = 1.6;
				ctx.globalAlpha = f;
				ctx.stroke();
				ctx.globalAlpha = 1;
			}
		} else if (snap.mode === '3d') {
			const [n0, n1, n2] = snap.sepN;
			const poly = planeCubePolygon(n0, n1, n2, snap.sepC * S, 1.0 /* scaled cube */);
			if (poly.length >= 3) {
				ctx.beginPath();
				poly.forEach((p, i) => {
					const [x, y] = toXY([p[0] / S, p[1] / S, p[2] / S]);
					if (i === 0) ctx.moveTo(x, y);
					else ctx.lineTo(x, y);
				});
				ctx.closePath();
				const ca = hexRgb(tk.ink);
				ctx.fillStyle = `rgba(${ca[0]},${ca[1]},${ca[2]},${(0.06 * f).toFixed(3)})`;
				ctx.fill();
				ctx.strokeStyle = tk.ink;
				ctx.globalAlpha = 0.55 * f;
				ctx.lineWidth = 1.2;
				ctx.stroke();
				ctx.globalAlpha = 1;
			}
		}

		// points — depth-sorted in 3-D
		const n = data.n;
		const order: number[] = Array.from({ length: n }, (_, i) => i);
		const coords: Array<[number, number, number]> = [];
		for (let i = 0; i < n; i++) {
			lerpAt(prev?.ptsV ?? null, snap.ptsV, i, d, u, tmp);
			if (f < 1) blendFlat(tmp, data.x[2 * i], data.x[2 * i + 1]);
			coords.push(toXY(tmp));
		}
		if (turns) order.sort((a, b) => coords[a][2] - coords[b][2]);
		for (const i of order) {
			const [x, y, z] = coords[i];
			const dn = turns ? (z + 1.6) / 3.2 : 1;
			ctx.beginPath();
			ctx.arc(x, y, turns ? 1.9 + 1.5 * dn : 3, 0, Math.PI * 2);
			ctx.globalAlpha = turns ? 0.45 + 0.55 * Math.max(0, Math.min(1, dn)) : 1;
			ctx.fillStyle = data.labels[i] === 0 ? tk.accent : tk.warm;
			ctx.fill();
		}
		ctx.globalAlpha = 1;
		if (selected !== null) {
			const [x, y] = coords[selected];
			ring(ctx, x, y, tk.ink);
		}
	}

	// ── pointer interactions ──
	function pickPoint(ev: PointerEvent) {
		if (!inputCanvas) return;
		const r = inputCanvas.getBoundingClientRect();
		const [Ex, Ey] = viewExtents(1.15, r.width, r.height);
		const mx = ((ev.clientX - r.left) / r.width) * 2 * Ex - Ex;
		const my = -(((ev.clientY - r.top) / r.height) * 2 * Ey - Ey);
		let best = -1;
		let bestD = 0.008;
		for (let i = 0; i < data.n; i++) {
			const dx = data.x[2 * i] - mx;
			const dy = data.x[2 * i + 1] - my;
			const dd = dx * dx + dy * dy;
			if (dd < bestD) {
				bestD = dd;
				best = i;
			}
		}
		selected = best === -1 ? null : best;
	}

	let dragLast: [number, number] | null = null;
	function hiddenWheel(ev: WheelEvent) {
		ev.preventDefault();
		zoom = Math.min(3, Math.max(0.3, zoom * Math.exp(-ev.deltaY * 0.0016)));
	}
	function hiddenDown(ev: PointerEvent) {
		if (!snapB || snapB.mode === '2d') return;
		dragging = true;
		dragLast = [ev.clientX, ev.clientY];
		(ev.target as Element).setPointerCapture(ev.pointerId);
	}
	function hiddenMove(ev: PointerEvent) {
		if (!dragging || !dragLast) return;
		yaw += (ev.clientX - dragLast[0]) * 0.008;
		pitch = Math.max(-1.2, Math.min(1.2, pitch + (ev.clientY - dragLast[1]) * 0.006));
		dragLast = [ev.clientX, ev.clientY];
	}
	function hiddenUp() {
		dragging = false;
		dragLast = null;
	}

	// ── sparkline paths ──
	function sparkPath(vals: number[], w: number, h: number, logY: boolean): string {
		if (vals.length < 2) return '';
		let lo = Infinity;
		let hi = -Infinity;
		for (const v of vals) {
			const y = logY ? Math.log(Math.max(v, 1e-4)) : v;
			lo = Math.min(lo, y);
			hi = Math.max(hi, y);
		}
		if (hi - lo < 1e-9) hi = lo + 1e-9;
		return vals
			.map((v, i) => {
				const yv = logY ? Math.log(Math.max(v, 1e-4)) : v;
				const x = (i / (vals.length - 1)) * w;
				const y = h - ((yv - lo) / (hi - lo)) * (h - 4) - 2;
				return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
			})
			.join(' ');
	}
</script>

<Plate {n} {title} {caption}>
	{#snippet status()}
		{#if phase === 'ready'}
			<span>step {step}</span>
			<span aria-hidden="true">·</span>
			<span>loss {Number.isFinite(lossNow) ? lossNow.toFixed(3) : '—'}</span>
			<span aria-hidden="true">·</span>
			<span>acc {Number.isFinite(accNow) ? (accNow * 100).toFixed(0) + '%' : '—'}</span>
		{:else if phase === 'loading'}
			<span>waking…</span>
		{:else if phase === 'error'}
			<span style="color: var(--bad);">error</span>
		{:else}
			<span>—</span>
		{/if}
	{/snippet}

	{#snippet actions()}
		<Btn onclick={() => setTraining(!training)} disabled={phase !== 'ready'}>
			{#if training}
				<Pause size={12} aria-hidden="true" /> Pause
			{:else}
				<Play size={12} aria-hidden="true" /> Train
			{/if}
		</Btn>
		<Btn onclick={resetWeights} disabled={phase !== 'ready'} title="Fresh random weights">
			<RotateCcw size={12} aria-hidden="true" /> Reset
		</Btn>
	{/snippet}

	<div use:inview={autoload}>
		{#if phase === 'error'}
			<div class="flex flex-wrap items-center gap-3 px-4 py-4">
				<span class="text-[12.5px] text-bad">{errorMsg}</span>
				<Btn onclick={boot}>Retry</Btn>
			</div>
		{:else if phase !== 'ready'}
			<div class="flex h-[280px] flex-col items-center justify-center gap-1">
				<span class="eyebrow">waking the network…</span>
				<span class="text-[12.5px] text-ink-3">
					it trains right here, on your machine — nothing leaves this page
				</span>
			</div>
		{:else}
			<!-- twin views + the inspector column: one row, no scrolling to see the machine -->
			<div
				class="grid grid-cols-1 gap-px bg-line-soft sm:grid-cols-2 lg:grid-cols-[1fr_1fr_minmax(264px,300px)]"
			>
				<div class="relative flex flex-col bg-surface">
					<span class="eyebrow absolute top-3 left-3 z-10">input space · x</span>
					<canvas
						bind:this={inputCanvas}
						class="block aspect-square w-full cursor-crosshair"
						onpointerdown={pickPoint}
					></canvas>

					{#if variant === 'free'}
						<!-- pick the tangle by sight: one miniature per dataset, below the plot -->
						<div class="flex gap-1.5 px-2 pt-1.5 pb-2" role="group" aria-label="Dataset">
							{#each thumbs as th (th.id)}
								<button
									class="thumb"
									class:thumb-on={dataset === th.id}
									title={th.label}
									aria-label="Dataset: {th.label}"
									aria-pressed={dataset === th.id}
									onclick={() => void switchDataset(th.id)}
								>
									<svg viewBox="0 0 40 40" aria-hidden="true">
										{#each th.pts as p, i (i)}
											<circle
												cx={p[0]}
												cy={p[1]}
												r="1.2"
												fill={p[2] === 0 ? 'var(--accent)' : 'var(--warm)'}
											/>
										{/each}
									</svg>
								</button>
							{/each}
						</div>
					{/if}
				</div>
				<div class="group relative bg-surface">
					<span class="eyebrow absolute top-3 left-3 z-10">
						hidden space · h
						{#if hiddenMode === '3d'}<span class="tracking-normal text-ink-3 normal-case">
								— drag to turn</span
							>{/if}
						{#if hiddenMode === 'pca'}<span class="tracking-normal text-ink-3 normal-case">
								— PCA shadow of {width}-D · drag to turn</span
							>{/if}
					</span>
					<canvas
						bind:this={hiddenCanvas}
						class="block aspect-square h-full w-full touch-none"
						class:cursor-grab={hiddenMode !== '2d' && !dragging}
						class:cursor-grabbing={dragging}
						onpointerdown={hiddenDown}
						onpointermove={hiddenMove}
						onpointerup={hiddenUp}
						onpointercancel={hiddenUp}
						onwheel={hiddenWheel}
					></canvas>

					<!-- the fold replay: flatten the sheet, then let the network fold it again -->
					<button
						class="chip absolute top-2 right-2 z-10 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
						class:opacity-100={foldTarget === 0}
						aria-pressed={foldTarget === 0}
						title="Animate between the flat sheet and the network's deformation"
						onclick={toggleFold}
					>
						{foldTarget === 1 ? 'unfold' : 'fold'}
					</button>

					{#if variant === 'guided' && stage === 'done'}
						<p class="absolute right-3 bottom-3 z-10 font-serif text-[13.5px] text-good italic">
							the rings came apart — a flat cut through a lifted space
						</p>
					{/if}
				</div>

				<!-- the network + its settings, harmonized in one column -->
				<div class="flex flex-col bg-surface px-3 pt-3 pb-3 sm:col-span-2 lg:col-span-1">
					<span class="eyebrow block px-1">The network</span>
					{#if weightsView}
						<ArchDiagram
							weights={weightsView}
							{hovered}
							onhover={(h) => (hovered = h)}
							badges={false}
							inLabels={['x₁', 'x₂']}
							outLabels={['blue', 'orange']}
							outColors={['var(--accent)', 'var(--warm)']}
						/>
					{/if}

					<div class="mt-2.5 flex flex-col gap-y-2 px-1">
						{#if variant === 'guided'}
							<span
								class="flex flex-wrap items-center gap-1"
								role="group"
								aria-label="Hidden dimensions"
							>
								<span class="eyebrow mr-1 w-11">hidden</span>
								<button
									class="chip"
									class:chip-on={width === 2}
									onclick={() => void setGuidedWidth(2)}>2-D</button
								>
								<button
									class="chip"
									class:chip-on={width === 3}
									onclick={() => void setGuidedWidth(3)}>3-D</button
								>
							</span>
						{/if}
						{#if variant === 'free'}
							<span
								class="flex flex-wrap items-center gap-1"
								role="group"
								aria-label="Hidden width"
							>
								<span class="eyebrow mr-1 w-11">width</span>
								{#each [2, 3, 4, 8] as w (w)}
									<button
										class="chip"
										class:chip-on={width === w}
										onclick={() => {
											width = w;
											void rebuild();
										}}>{w}</button
									>
								{/each}
							</span>
							<span class="flex flex-wrap items-center gap-1" role="group" aria-label="Depth">
								<span class="eyebrow mr-1 w-11">depth</span>
								{#each [1, 2, 3] as dd (dd)}
									<button
										class="chip"
										class:chip-on={depth === dd}
										onclick={() => {
											depth = dd;
											void rebuild();
										}}>{dd}</button
									>
								{/each}
							</span>
						{/if}
						<span class="flex flex-wrap items-center gap-1" role="group" aria-label="Activation">
							<span class="eyebrow mr-1 w-11">bend</span>
							{#each ACTS as act (act)}
								<button
									class="chip"
									class:chip-on={activation === act}
									onclick={() => switchActivation(act)}>{act}</button
								>
							{/each}
						</span>
					</div>
				</div>
			</div>

			<!-- slim telemetry strip: the dial, the curves, the vitals -->
			<div class="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line-soft px-4 py-2">
				{#if variant === 'free'}
					<span class="w-48">
						<Slider
							label="learning rate γ"
							bind:value={logLr}
							min={-3}
							max={-1.4}
							step={0.05}
							format={(v) => (10 ** v).toFixed(3)}
						/>
					</span>
				{/if}
				<span class="flex items-center gap-1.5 whitespace-nowrap">
					<span class="eyebrow text-[9.5px]">loss</span>
					<svg width="96" height="22" class="shrink-0" aria-label="training loss" role="img">
						<path
							d={sparkPath(lossHist, 96, 22, true)}
							fill="none"
							stroke="var(--accent)"
							stroke-width="1.4"
						/>
					</svg>
				</span>
				<span class="flex items-center gap-1.5 whitespace-nowrap">
					<span class="eyebrow text-[9.5px]">val acc</span>
					<svg width="96" height="22" class="shrink-0" aria-label="held-out accuracy" role="img">
						<path
							d={sparkPath(accHist, 96, 22, false)}
							fill="none"
							stroke="var(--warm)"
							stroke-width="1.4"
						/>
					</svg>
				</span>
				<span class="num ml-auto text-[10.5px] text-ink-3">
					{archLabel} · {paramCount} params · {device} · {msPerStep.toFixed(0)} ms/step
				</span>
			</div>
		{/if}
	</div>
</Plate>

<style>
	.thumb {
		flex: 1 1 0;
		min-width: 0;
		max-width: 56px;
		aspect-ratio: 1;
		height: auto;
		padding: 2px;
		border-radius: 7px;
		border: 1px solid var(--line);
		background: var(--surface);
		cursor: pointer;
		opacity: 0.94;
		transition:
			border-color 0.1s,
			opacity 0.1s;
	}
	.thumb:hover {
		border-color: var(--ink-3);
		opacity: 1;
	}
	.thumb-on {
		border-color: var(--ink);
		opacity: 1;
	}
	.thumb svg {
		display: block;
		width: 100%;
		height: 100%;
	}

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

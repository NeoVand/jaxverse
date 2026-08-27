<script lang="ts">
	import { onDestroy } from 'svelte';
	import { Play, Pause, RotateCcw, Pencil } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { MlpEngine } from '$lib/nn/mlp-engine';
	import type { Activation, LayerWeights } from '$lib/nn/engine';
	import { progress } from '$lib/data/progress.svelte';
	import ArchDiagram, { type NodeRef } from '$lib/components/ui/ArchDiagram.svelte';
	import WeightLegend from '$lib/components/ui/WeightLegend.svelte';
	import { DATA_N, GRID_N, PRESETS, evenXs, makePerm, type TargetId } from './targets';

	// ── fixed data ────────────────────────────────────────────────────────────
	// x's never change; only the target y's do. Storage order is shuffled so the
	// engine's held-out tail is an unbiased sample of the interval.
	const xsSorted = evenXs(DATA_N);
	const perm = makePerm(DATA_N, 7);
	const xData = new Float32Array(DATA_N);
	for (let k = 0; k < DATA_N; k++) xData[k] = xsSorted[perm[k]];
	const gridXs = evenXs(GRID_N);

	// The target, in sorted order — a plain array so pen edits are reactive.
	const targetY = $state<number[]>(new Array(DATA_N).fill(0));
	let targetId = $state<TargetId>('sine');
	let hasDrawn = $state(false);

	function applyPreset(id: Exclude<TargetId, 'draw'>) {
		const preset = PRESETS.find((p) => p.id === id);
		if (!preset) return;
		for (let i = 0; i < DATA_N; i++) targetY[i] = preset.fn(xsSorted[i]);
	}
	applyPreset('sine');

	function storageY(): Float32Array {
		const y = new Float32Array(DATA_N);
		for (let k = 0; k < DATA_N; k++) y[k] = targetY[perm[k]];
		return y;
	}

	// ── architecture ──────────────────────────────────────────────────────────
	const WIDTHS = [2, 4, 8, 16, 32];
	const DEPTHS = [1, 2, 3] as const;
	const ACTIVATIONS: Activation[] = ['tanh', 'relu', 'gelu', 'silu'];
	let widthIdx = $state(3);
	let depth = $state<1 | 2 | 3>(2);
	let activation = $state<Activation>('tanh');
	const width = $derived(WIDTHS[widthIdx]);
	const archKey = $derived(`${depth}:${width}:${activation}`);

	function cfg() {
		return {
			layers: [1, ...Array(depth).fill(width), 1],
			activation,
			loss: 'mse' as const,
			lr: 8e-3,
			batchSize: 64,
			seed: 7
		};
	}

	// ── engine lifecycle ──────────────────────────────────────────────────────
	let engine: MlpEngine | null = null; // deliberately not $state
	let trainPromise: Promise<void> | null = null;
	let playing = false;
	let builtKey = '';
	let rebuildingNow = false;
	let rebuildQueued = false;
	let rebuildTimer: ReturnType<typeof setTimeout> | undefined;
	let refreshInFlight = false;
	let lastRefreshAt = 0;
	let chunksSinceRefresh = 0;
	let milestoneDone = false;

	let phase = $state<'idle' | 'loading' | 'ready' | 'training' | 'error'>('idle');
	let rebuilding = $state(false);
	let errorMsg = $state('');
	let step = $state(0);
	let lossNow = $state(NaN);
	let msNow = $state(NaN);
	let lossHist = $state<number[]>([]);
	let paramCount = $state(0);
	let device = $state('');

	let predY = $state<Float32Array | null>(null);
	// One curve per neuron: `layers[li][j]` is hidden layer li's unit j output
	// over the plot grid; `v[j]` is the final hidden layer's output weight.
	let palette = $state<{ layers: Float32Array[][]; v: number[]; total: number } | null>(null);
	let weightsView = $state<LayerWeights[] | null>(null);
	let hovered = $state<NodeRef | null>(null);

	const chunkSteps = () => 40;
	const syncEvery = () => 3;

	// Fired once by use:inview when the plate scrolls near — never on mount.
	async function boot() {
		if (phase !== 'idle') return;
		phase = 'loading';
		errorMsg = '';
		try {
			const e = new MlpEngine();
			await e.init(cfg(), { x: xData, y: storageY(), n: DATA_N });
			engine = e;
			builtKey = archKey;
			paramCount = e.paramCount;
			device = e.device;
			phase = 'ready';
			await refreshViz();
		} catch (err) {
			errorMsg = err instanceof Error ? err.message : String(err);
			phase = 'error';
		}
	}

	function retry() {
		phase = 'idle';
		void boot();
	}

	async function refreshViz() {
		const e = engine;
		if (!e || refreshInFlight) return;
		refreshInFlight = true;
		try {
			const [p, acts, ws] = await Promise.all([
				e.predict(gridXs, GRID_N, GRID_N),
				e.activations(gridXs, GRID_N, GRID_N),
				e.weights()
			]);
			if (engine !== e) return; // superseded by a rebuild
			predY = p;
			weightsView = ws;
			// The palette: every hidden neuron's output curve, layer by layer.
			// acts.layers is [hidden 1, …, hidden D, output]; the output curve is
			// already in predY, so only the hidden layers are unpacked here.
			const hiddenCount = acts.layers.length - 1;
			const layers: Float32Array[][] = [];
			for (let li = 0; li < hiddenCount; li++) {
				const flat = acts.layers[li];
				const lw = acts.widths[li];
				const shown = Math.min(lw, 16);
				const units: Float32Array[] = [];
				for (let j = 0; j < shown; j++) {
					const cj = new Float32Array(GRID_N);
					for (let i = 0; i < GRID_N; i++) cj[i] = flat[i * lw + j];
					units.push(cj);
				}
				layers.push(units);
			}
			// Final hidden layer's output weights v_j (last weight matrix is W×1).
			const out = ws[ws.length - 1];
			const v = layers[hiddenCount - 1].map((_, j) => out.w[j]);
			palette = { layers, v, total: acts.widths[hiddenCount - 1] };
		} finally {
			refreshInFlight = false;
			lastRefreshAt = performance.now();
			chunksSinceRefresh = 0;
		}
	}

	async function loop() {
		while (playing && engine) {
			const e = engine;
			let sum = 0;
			let count = 0;
			await e.train(chunkSteps(), (m) => {
				step = m.step;
				lossNow = m.loss;
				msNow = m.stepMs;
				sum += m.loss;
				count++;
			});
			if (engine !== e || count === 0) break;
			lossHist = [...lossHist.slice(-159), sum / count];
			if (!milestoneDone) {
				milestoneDone = true;
				progress.reach('neuron:trained');
			}
			chunksSinceRefresh++;
			if (chunksSinceRefresh >= syncEvery() || performance.now() - lastRefreshAt > 300)
				await refreshViz();
		}
	}

	function startTraining() {
		if (!engine || playing || rebuildingNow) return;
		playing = true;
		phase = 'training';
		trainPromise = loop();
	}

	async function pauseTraining() {
		playing = false;
		await engine?.stop().catch(() => {});
		await trainPromise;
		trainPromise = null;
		if (phase === 'training') phase = 'ready';
		await refreshViz();
	}

	function toggleTrain() {
		if (phase === 'training') void pauseTraining();
		else if (phase === 'ready') startTraining();
	}

	async function resetWeights() {
		const e = engine;
		if (!e || rebuildingNow) return;
		await e.reset(Math.floor(Math.random() * 1e9));
		step = 0;
		lossNow = NaN;
		msNow = NaN;
		lossHist = [];
		if (!playing) await refreshViz();
	}

	// Width, depth or activation changes rebuild the engine: dispose the worker,
	// boot a fresh one with the same target data, resume training if it was on.
	async function rebuild() {
		if (rebuildingNow) {
			rebuildQueued = true;
			return;
		}
		if (!engine || archKey === builtKey) return;
		rebuildingNow = true;
		rebuilding = true;
		hovered = null;
		const wasPlaying = playing;
		playing = false;
		const old = engine;
		engine = null;
		try {
			await old.stop().catch(() => {});
			await trainPromise;
			trainPromise = null;
			await old.dispose().catch(() => {});
			const e = new MlpEngine();
			await e.init(cfg(), { x: xData, y: storageY(), n: DATA_N });
			engine = e;
			builtKey = archKey;
			paramCount = e.paramCount;
			device = e.device;
			step = 0;
			lossNow = NaN;
			msNow = NaN;
			lossHist = [];
			// Pen strokes made while the worker was rebooting were held back —
			// push the current target so the new engine never trains on a stale one.
			await e.setData({ x: xData, y: storageY(), n: DATA_N }).catch(() => {});
			await refreshViz();
			if (wasPlaying) {
				playing = true;
				phase = 'training';
				trainPromise = loop();
			} else {
				phase = 'ready';
			}
		} catch (err) {
			errorMsg = err instanceof Error ? err.message : String(err);
			phase = 'error';
		} finally {
			rebuildingNow = false;
			rebuilding = false;
			if (rebuildQueued) {
				rebuildQueued = false;
				if (archKey !== builtKey) void rebuild();
			}
		}
	}

	$effect(() => {
		const key = archKey;
		if (phase === 'idle' || phase === 'loading') return;
		if (key === builtKey) return;
		clearTimeout(rebuildTimer);
		rebuildTimer = setTimeout(() => void rebuild(), 250);
	});

	onDestroy(() => {
		playing = false;
		clearTimeout(rebuildTimer);
		if (pushTimer) clearTimeout(pushTimer);
		const e = engine;
		engine = null;
		void e?.dispose();
	});

	// ── pushing target edits to the worker (throttled) ────────────────────────
	let lastPush = 0;
	let pushTimer: ReturnType<typeof setTimeout> | null = null;

	function sendData() {
		lastPush = performance.now();
		if (!engine || rebuildingNow) return;
		engine.setData({ x: xData, y: storageY(), n: DATA_N }).catch(() => {});
	}

	function queueSend(flush = false) {
		const wait = 70 - (performance.now() - lastPush);
		if (flush || wait <= 0) {
			if (pushTimer) {
				clearTimeout(pushTimer);
				pushTimer = null;
			}
			sendData();
		} else if (!pushTimer) {
			pushTimer = setTimeout(() => {
				pushTimer = null;
				sendData();
			}, wait);
		}
	}

	function selectTarget(id: TargetId) {
		targetId = id;
		if (id !== 'draw') {
			applyPreset(id);
			queueSend(true);
		}
	}

	// ── drawing: the pen defines the target, the fit chases it ────────────────
	let svgEl = $state<SVGSVGElement | null>(null);
	let stroking = false;
	let lastIdx = 0;
	let lastY = 0;

	function evtPoint(e: PointerEvent) {
		const r = svgEl!.getBoundingClientRect();
		const x = Math.max(-1, Math.min(1, pxToX(e.clientX - r.left)));
		const y = Math.max(-0.95, Math.min(0.95, pxToY(e.clientY - r.top)));
		return { idx: Math.round(((x + 1) / 2) * (DATA_N - 1)), y };
	}

	function writeSegment(i0: number, y0: number, i1: number, y1: number) {
		if (i1 === i0) {
			targetY[i1] = y1;
			return;
		}
		const dir = i1 > i0 ? 1 : -1;
		for (let i = i0; i !== i1 + dir; i += dir) {
			const t = (i - i0) / (i1 - i0);
			targetY[i] = y0 + t * (y1 - y0);
		}
	}

	function onPointerDown(e: PointerEvent) {
		if (targetId !== 'draw' || !svgEl) return;
		e.preventDefault();
		svgEl.setPointerCapture(e.pointerId);
		stroking = true;
		hasDrawn = true;
		const { idx, y } = evtPoint(e);
		targetY[idx] = y;
		lastIdx = idx;
		lastY = y;
		queueSend();
		if (phase === 'ready') startTraining();
	}

	function onPointerMove(e: PointerEvent) {
		if (!stroking) return;
		e.preventDefault();
		const { idx, y } = evtPoint(e);
		writeSegment(lastIdx, lastY, idx, y);
		lastIdx = idx;
		lastY = y;
		queueSend();
	}

	function onPointerUp() {
		if (!stroking) return;
		stroking = false;
		queueSend(true);
	}

	// ── plot geometry & paths ─────────────────────────────────────────────────
	const H = 300;
	const PADX = 16;
	const PADT = 14;
	const PADB = 22;
	const YMAX = 1.2;

	let plotW = $state(0);
	const pw = $derived(plotW || 640);
	const xPix = (x: number) => PADX + ((x + 1) / 2) * (pw - 2 * PADX);
	const yPix = (y: number) => PADT + (1 - (y + YMAX) / (2 * YMAX)) * (H - PADT - PADB);
	const pxToX = (px: number) => ((px - PADX) / (pw - 2 * PADX)) * 2 - 1;
	const pxToY = (py: number) => (1 - (py - PADT) / (H - PADT - PADB)) * 2 * YMAX - YMAX;

	function linePath(xs: ArrayLike<number>, ys: ArrayLike<number>): string {
		let d = '';
		for (let i = 0; i < xs.length; i++)
			d += `${i === 0 ? 'M' : 'L'}${xPix(xs[i]).toFixed(1)} ${yPix(ys[i]).toFixed(1)}`;
		return d;
	}

	const targetPath = $derived(linePath(xsSorted, targetY));
	const predPath = $derived(predY ? linePath(gridXs, predY) : '');

	const dotsPath = $derived.by(() => {
		const r = 1.6;
		let d = '';
		for (let i = 0; i < DATA_N; i++) {
			const cx = xPix(xsSorted[i]);
			const cy = yPix(targetY[i]);
			d += `M${(cx + r).toFixed(1)} ${cy.toFixed(1)}a${r} ${r} 0 1 0 ${-2 * r} 0a${r} ${r} 0 1 0 ${2 * r} 0`;
		}
		return d;
	});

	// Palette tiles: squat little plots that keep their aspect ratio, all sharing
	// one y-scale so heights are comparable across units.
	const TILE_W = 76;
	const TILE_H = 48;
	function tilePath(vals: Float32Array, m: number): string {
		let d = '';
		for (let i = 0; i < vals.length; i++) {
			const x = (i / (vals.length - 1)) * TILE_W;
			const y = (1 - (vals[i] + m) / (2 * m)) * TILE_H;
			d += `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(2)}`;
		}
		return d;
	}

	// Groups: one per shown hidden layer, each on its own shared y-scale. The
	// first hidden layer is skipped when deeper ones exist — its units are just
	// re-tilted copies of the raw activation, so it earns little for its space.
	// The final layer shows v-scaled contributions, the curve's thickness
	// tracking |v| exactly like the edges in the diagram. `col` names the
	// matching diagram column so hover can run both ways; `fill` pads the last
	// grid row so the hairline grid stays rectangular.
	const paletteView = $derived.by(() => {
		const p = palette;
		if (!p) return null;
		const last = p.layers.length - 1;
		let vmax = 0.1;
		for (const val of p.v) vmax = Math.max(vmax, Math.abs(val));
		const groups = p.layers
			.map((units, li) => ({ units, li }))
			.filter(({ li }) => li > 0 || li === last)
			.map(({ units, li }) => {
				const curves =
					li === last
						? units.map((h, j) => {
								const c = new Float32Array(GRID_N);
								for (let i = 0; i < GRID_N; i++) c[i] = p.v[j] * h[i];
								return c;
							})
						: units;
				let m = 0.45;
				for (const c of curves) for (const val of c) m = Math.max(m, Math.abs(val));
				m *= 1.08;
				const name = last === 0 ? 'hidden layer' : `hidden layer ${li + 1}`;
				return {
					col: li + 1,
					label: li === last ? `${name} · each × its output weight v` : `${name} · raw outputs`,
					tiles: curves.map((c, j) => ({
						d: tilePath(c, m),
						v: li === last ? p.v[j] : null,
						sw: li === last ? 0.9 + (1.7 * Math.abs(p.v[j])) / vmax : 1.4
					})),
					fill: (8 - (curves.length % 8)) % 8
				};
			});
		return {
			groups,
			shown: groups[groups.length - 1].tiles.length,
			total: p.total
		};
	});

	const sparkPath = $derived.by(() => {
		if (lossHist.length < 2) return '';
		const logs = lossHist.map((val) => Math.log10(Math.max(val, 1e-8)));
		const lo = Math.min(...logs);
		const hi = Math.max(...logs);
		const span = Math.max(hi - lo, 1e-3);
		let d = '';
		for (let i = 0; i < logs.length; i++) {
			const x = (i / (logs.length - 1)) * 100;
			const y = 1 + (1 - (logs[i] - lo) / span) * 22;
			d += `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(2)}`;
		}
		return d;
	});

	// Only neurons that own a tile spotlight the palette; the rest (input,
	// output, a skipped first layer) still light up their edges in the diagram.
	const hoverInPalette = $derived.by(() => {
		const h = hovered;
		const pv = paletteView;
		return (
			h !== null && pv !== null && pv.groups.some((g) => g.col === h.col && h.idx < g.tiles.length)
		);
	});

	const fmtLoss = (val: number) =>
		!isFinite(val) ? '—' : val >= 1e-3 ? val.toFixed(4) : val.toExponential(1);
	const fmtMs = (val: number) => (!isFinite(val) ? '—' : val.toFixed(1));
	const controlsLocked = $derived(phase === 'idle' || phase === 'loading' || rebuilding);
</script>

<Plate
	id="workshop"
	live
	title="The curve workshop"
	caption="The network itself, live — every edge is one weight, read by the legend beneath it. The plot: the dashed target, and the fit the network currently draws against it. Underneath, the palette — the curves the deeper layers output, ending in the last hidden layer's contributions: colored by the sign of each output weight, thicker as it grows, and summing exactly to the fit. Hover any node or tile; the highlight runs both ways."
>
	{#snippet status()}
		{#if phase === 'idle'}
			<span>—</span>
		{:else if phase === 'loading'}
			<span>loading…</span>
		{:else if rebuilding}
			<span>rebuilding…</span>
		{:else if phase === 'error'}
			<span style="color: var(--bad);">error</span>
		{:else}
			<span>step {step}</span>
			<span aria-hidden="true">·</span>
			<span>loss {fmtLoss(lossNow)}</span>
			<span aria-hidden="true">·</span>
			<span>{fmtMs(msNow)} ms/step</span>
		{/if}
	{/snippet}

	{#snippet actions()}
		<Btn kind="primary" onclick={toggleTrain} disabled={controlsLocked}>
			{#if phase === 'training'}
				<Pause size={13} aria-hidden="true" /> Pause
			{:else}
				<Play size={13} aria-hidden="true" /> Train
			{/if}
		</Btn>
		<Btn onclick={() => void resetWeights()} disabled={controlsLocked}>
			<RotateCcw size={13} aria-hidden="true" /> Reset
		</Btn>
	{/snippet}

	<div use:inview={boot}>
		{#if phase === 'error'}
			<div class="flex flex-wrap items-center gap-3 px-4 py-2.5">
				<span class="num text-[11.5px]" style="color: var(--bad);">
					{errorMsg || 'the training worker failed to start'}
				</span>
				<Btn onclick={retry}><RotateCcw size={13} aria-hidden="true" /> Retry</Btn>
			</div>
		{/if}

		<!-- the stage: architecture left, fit right -->
		<div class="grid sm:grid-cols-[minmax(300px,390px)_1fr]">
			<div
				class="order-2 flex flex-col justify-center border-t border-line-soft px-3 pt-2.5 pb-2 sm:order-1 sm:border-t-0 sm:border-r"
			>
				<span class="eyebrow block px-1">The network</span>
				{#if weightsView}
					<ArchDiagram weights={weightsView} {hovered} onhover={(h) => (hovered = h)} />
					<WeightLegend />
				{:else}
					<div class="flex h-[240px] items-center justify-center">
						<span class="num text-[11px] text-ink-3">
							{phase === 'error' ? 'no worker' : 'waking the training worker…'}
						</span>
					</div>
				{/if}
			</div>

			<div class="relative order-1 select-none sm:order-2" bind:clientWidth={plotW}>
				<svg
					bind:this={svgEl}
					width="100%"
					height={H}
					viewBox="0 0 {pw} {H}"
					class="block"
					class:cursor-crosshair={targetId === 'draw'}
					style:touch-action={targetId === 'draw' ? 'none' : 'pan-y'}
					role="img"
					aria-label="Target curve and the network's fit. In draw mode, drag across the plot to define your own target."
					onpointerdown={onPointerDown}
					onpointermove={onPointerMove}
					onpointerup={onPointerUp}
					onpointercancel={onPointerUp}
				>
					<line
						x1={PADX}
						x2={pw - PADX}
						y1={yPix(0)}
						y2={yPix(0)}
						stroke="var(--line)"
						stroke-width="1"
					/>
					<line
						x1={xPix(0)}
						x2={xPix(0)}
						y1={PADT}
						y2={H - PADB}
						stroke="var(--line-soft)"
						stroke-width="1"
					/>
					{#each [-1, 1] as gy (gy)}
						<line
							x1={PADX}
							x2={pw - PADX}
							y1={yPix(gy)}
							y2={yPix(gy)}
							stroke="var(--line-soft)"
							stroke-width="1"
						/>
					{/each}
					{#each [-1, 0, 1] as gx (gx)}
						<text
							x={xPix(gx)}
							y={H - 7}
							text-anchor="middle"
							class="num"
							font-size="10"
							fill="var(--ink-3)">{gx === 0 ? '0' : gx > 0 ? '+1' : '−1'}</text
						>
					{/each}

					<!-- training points -->
					<path d={dotsPath} fill="var(--ink-3)" opacity="0.35" />
					<!-- the target -->
					<path
						d={targetPath}
						fill="none"
						stroke="var(--ink-3)"
						stroke-width="1.5"
						stroke-dasharray="4 4"
					/>
					<!-- the network -->
					{#if predPath}
						<path
							d={predPath}
							fill="none"
							stroke="var(--accent)"
							stroke-width="2"
							stroke-linejoin="round"
							stroke-linecap="round"
						/>
					{/if}
				</svg>

				{#if targetId === 'draw' && !hasDrawn}
					<div class="pointer-events-none absolute inset-0 flex items-center justify-center">
						<span
							class="eyebrow rounded-full bg-surface/80 px-3 py-1.5"
							style="backdrop-filter: blur(2px);">drag across the plot to draw a target</span
						>
					</div>
				{/if}
			</div>
		</div>

		<!-- the palette -->
		<div class="border-t border-line-soft px-4 pt-3 pb-4">
			<div class="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
				<span class="eyebrow">The palette</span>
				{#if paletteView && paletteView.total > paletteView.shown}
					<span class="num text-[10px] text-ink-3"
						>showing {paletteView.shown} of {paletteView.total} units per layer</span
					>
				{/if}
			</div>
			<p class="mt-0.5 text-[11px] leading-relaxed text-ink-3">
				the curves the deeper layers output — the last hidden layer is drawn as
				<span class="num">v·{activation}(…)</span>, ultramarine where its output weight
				<span class="num">v</span> is positive, vermilion where negative, thicker as |v| grows; the fit
				above is exactly their sum. hover any tile or any node in the diagram — the highlight runs both
				ways
			</p>
			{#if paletteView}
				<div
					class="mt-2.5 grid grid-cols-4 gap-px overflow-hidden rounded-md border border-line-soft bg-line-soft sm:grid-cols-8"
				>
					{#each paletteView.groups as group (group.col)}
						<div class="col-span-full bg-surface px-2 pt-1 pb-0.5">
							<span class="eyebrow text-[9px]">{group.label}</span>
						</div>
						{#each group.tiles as tile, j (j)}
							{@const isHi = hovered !== null && hovered.col === group.col && hovered.idx === j}
							<!-- hover-only affordance: highlights the matching node in the diagram -->
							<div
								class="relative bg-surface transition-opacity duration-100"
								class:opacity-30={hoverInPalette && !isHi}
								style={isHi ? 'background: var(--surface-2);' : ''}
								role="presentation"
								onpointerenter={() => (hovered = { col: group.col, idx: j })}
								onpointerleave={() => (hovered = null)}
							>
								<svg viewBox="0 0 {TILE_W} {TILE_H}" class="block w-full" aria-hidden="true">
									<line
										x1="0"
										x2={TILE_W}
										y1={TILE_H / 2}
										y2={TILE_H / 2}
										stroke="var(--line)"
										stroke-width="1"
										vector-effect="non-scaling-stroke"
									/>
									<line
										x1={TILE_W / 2}
										x2={TILE_W / 2}
										y1="0"
										y2={TILE_H}
										stroke="var(--line-soft)"
										stroke-width="1"
										stroke-dasharray="2 3"
										vector-effect="non-scaling-stroke"
									/>
									<path
										d={tile.d}
										fill="none"
										stroke={tile.v !== null && tile.v < 0 ? 'var(--warm)' : 'var(--accent)'}
										stroke-opacity={isHi ? 1 : tile.v === null ? 0.7 : 0.85}
										stroke-width={tile.sw}
										stroke-linejoin="round"
										vector-effect="non-scaling-stroke"
									/>
								</svg>
								<span
									class="num absolute top-0.5 left-1 text-[8.5px]"
									style="color: {isHi ? 'var(--ink)' : 'var(--ink-3)'};">#{j + 1}</span
								>
								{#if tile.v !== null}
									<span
										class="num absolute right-1 bottom-0.5 text-[8.5px]"
										style="color: {tile.v >= 0 ? 'var(--accent)' : 'var(--warm)'};"
										>{tile.v >= 0 ? '+' : '−'}{Math.abs(tile.v).toFixed(1)}</span
									>
								{/if}
							</div>
						{/each}
						{#each { length: group.fill }, fi (fi)}
							<div class="bg-surface" aria-hidden="true"></div>
						{/each}
					{/each}
				</div>
			{:else}
				<div class="mt-2.5 flex h-16 items-center justify-center">
					<span class="num text-[11px] text-ink-3">the palette appears once the worker is up</span>
				</div>
			{/if}
		</div>

		<!-- one compact bench: target + architecture side by side -->
		<div class="flex flex-wrap items-end gap-x-5 gap-y-3 border-t border-line-soft px-4 py-3.5">
			<div>
				<span class="eyebrow mb-1 block">target</span>
				<div class="flex flex-wrap items-center gap-1.5">
					{#each PRESETS as p (p.id)}
						<button
							class="chip"
							class:chip-on={targetId === p.id}
							aria-pressed={targetId === p.id}
							onclick={() => selectTarget(p.id)}>{p.label}</button
						>
					{/each}
					<button
						class="chip"
						class:chip-on={targetId === 'draw'}
						aria-pressed={targetId === 'draw'}
						onclick={() => selectTarget('draw')}
					>
						<Pencil size={12} aria-hidden="true" /> draw
					</button>
				</div>
			</div>
			<div>
				<span class="eyebrow mb-1 block">hidden layers</span>
				<div class="seg" role="group" aria-label="number of hidden layers">
					{#each DEPTHS as d (d)}
						<button
							class:on={depth === d}
							aria-pressed={depth === d}
							disabled={controlsLocked}
							onclick={() => (depth = d)}>{d}</button
						>
					{/each}
				</div>
			</div>
			<div class="w-32">
				<Slider
					label="width"
					bind:value={widthIdx}
					min={0}
					max={4}
					step={1}
					tone="ink"
					disabled={controlsLocked}
					format={(i) => String(WIDTHS[i])}
				/>
			</div>
			<div>
				<span class="eyebrow mb-1 block">activation</span>
				<div class="seg" role="group" aria-label="activation function">
					{#each ACTIVATIONS as a (a)}
						<button
							class:on={activation === a}
							aria-pressed={activation === a}
							disabled={controlsLocked}
							onclick={() => (activation = a)}>{a}</button
						>
					{/each}
				</div>
			</div>
			<div class="ml-auto flex flex-col items-end gap-1 pb-0.5">
				{#if sparkPath}
					<div class="flex items-center gap-2">
						<span class="eyebrow text-[9.5px]">loss · log</span>
						<svg
							viewBox="0 0 100 24"
							preserveAspectRatio="none"
							class="h-5 w-24"
							aria-hidden="true"
						>
							<path
								d={sparkPath}
								fill="none"
								stroke="var(--accent)"
								stroke-width="1.5"
								vector-effect="non-scaling-stroke"
							/>
						</svg>
					</div>
				{/if}
				<span class="num text-[10.5px] text-ink-3" aria-live="polite">
					{paramCount.toLocaleString('en-US')} params · {device || '—'}
				</span>
			</div>
		</div>
	</div>
</Plate>

<style>
</style>

<script lang="ts">
	// Plate IV — search by drawing. The reader's own stroke goes through the
	// encoder, and the two thousand held-out digits are sorted by how near they
	// land: once in the waist's few numbers, once in the raw 784 pixels. Both
	// searches read the same centred ink, so the difference between the rows is
	// the difference between the two spaces and nothing else.
	import { onDestroy } from 'svelte';
	import { Play, Pause, Shuffle, Eraser } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import {
		DEFAULT_SIGMA,
		SIGMA_MAX,
		SIGMA_MIN,
		centered,
		stampDab,
		stampLine
	} from '$lib/data/brush';
	import { lab } from './latent-context.svelte';
	import { DIM, SIDE, TileStrip, readTokens, setupCanvas } from './common';

	interface Props {
		n: number;
		title: string;
		caption: string;
	}
	let { n, title, caption }: Props = $props();

	const PAD = 224; // pad CSS size — 8 css px per 28-grid pixel
	const SHOW = 8; // neighbours printed per row
	const SLOTS = Array.from({ length: SHOW }, (_, i) => i);
	const RANK = 10; // neighbours counted by the purity meters
	const PANEL = 48; // queries the meters average over

	type Metric = 'euclidean' | 'cosine';
	const METRICS: Metric[] = ['euclidean', 'cosine'];
	let metric = $state<Metric>('euclidean');

	// ── the query: 784 floats the reader owns ──
	const px = new Float32Array(DIM);

	/** A 3 waiting on the pad, so the plate arrives with something to search. */
	function seedDigit(): void {
		const arc = (cx: number, cy: number, r: number, a0: number, a1: number) => {
			let prev: [number, number] | null = null;
			for (let i = 0; i <= 22; i++) {
				const t = a0 + ((a1 - a0) * i) / 22;
				const p: [number, number] = [cx + r * Math.cos(t), cy + r * Math.sin(t)];
				if (prev) stampLine(px, prev[0], prev[1], p[0], p[1], DEFAULT_SIGMA);
				else stampDab(px, p[0], p[1], DEFAULT_SIGMA);
				prev = p;
			}
		};
		arc(13.5, 10, 4.6, -0.8 * Math.PI, 0.42 * Math.PI);
		arc(13.5, 17.6, 5.2, -0.46 * Math.PI, 0.78 * Math.PI);
	}
	seedDigit();

	let strokes = $state(0); // bumped per stamp → repaints the pad
	let hasInk = $state(true);
	let sigma = $state(DEFAULT_SIGMA);
	let brushAt = $state<{ x: number; y: number } | null>(null);
	/** Where the ink came from, when it came from the set — for the label line. */
	let source = $state<number | null>(null);

	let qx = $state.raw<Float32Array | null>(null); // the centred query pixels
	let qz = $state.raw<Float32Array | null>(null); // its embedding
	let qzd = $state(0);
	let recon = $state.raw<Float32Array | null>(null); // the round trip, for the eye
	let reconV = $state(0);

	/** Purity over a fixed panel of held-out queries: the share of each one's
	 * RANK nearest neighbours that carry its own label. Chance is about 10%. */
	let zPurity = $state(NaN);
	let pxPurity = $state(NaN);
	/** Pixel space never changes, so its answer is computed once per metric. */
	const pxCache: Partial<Record<Metric, number>> = {};
	let panelBusy = false;

	let raf = 0;
	const padStrip = new TileStrip();
	const reconStrip = new TileStrip();
	const zStrips = Array.from({ length: SHOW }, () => new TileStrip());
	const pxStrips = Array.from({ length: SHOW }, () => new TileStrip());
	let padCanvas: HTMLCanvasElement | undefined = $state();
	let reconCanvas: HTMLCanvasElement | undefined = $state();
	const zRow: (HTMLCanvasElement | undefined)[] = $state(SLOTS.map(() => undefined));
	const pxRow: (HTMLCanvasElement | undefined)[] = $state(SLOTS.map(() => undefined));

	// ── distances ──
	// Cosine is the metric embeddings are normally compared with; on a narrow
	// waist it throws away the radius, which is real information — hence both
	// are on offer, and both are applied to both spaces.
	function scoreAll(
		data: Float32Array,
		rows: number,
		d: number,
		q: ArrayLike<number>,
		out: Float32Array
	): void {
		if (metric === 'cosine') {
			let qn = 0;
			for (let c = 0; c < d; c++) qn += q[c] * q[c];
			qn = Math.sqrt(qn) || 1e-6;
			for (let i = 0; i < rows; i++) {
				let dot = 0;
				let nn = 0;
				const o = i * d;
				for (let c = 0; c < d; c++) {
					dot += data[o + c] * q[c];
					nn += data[o + c] * data[o + c];
				}
				out[i] = 1 - dot / (qn * (Math.sqrt(nn) || 1e-6));
			}
			return;
		}
		for (let i = 0; i < rows; i++) {
			let dd = 0;
			const o = i * d;
			for (let c = 0; c < d; c++) {
				const t = data[o + c] - q[c];
				dd += t * t;
			}
			out[i] = Math.sqrt(dd);
		}
	}

	/** The `k` rows closest to the query vector, nearest first. */
	function nearestK(
		data: Float32Array,
		rows: number,
		d: number,
		q: ArrayLike<number>,
		k: number,
		scratch: Float32Array,
		skip = -1
	): number[] {
		scoreAll(data, rows, d, q, scratch);
		const idx: number[] = [];
		const val: number[] = [];
		for (let i = 0; i < rows; i++) {
			if (i === skip) continue;
			const s = scratch[i];
			if (idx.length === k && s >= val[k - 1]) continue;
			let p = idx.length;
			while (p > 0 && val[p - 1] > s) p--;
			idx.splice(p, 0, i);
			val.splice(p, 0, s);
			if (idx.length > k) {
				idx.pop();
				val.pop();
			}
		}
		return idx;
	}

	const rows = $derived.by(() => {
		void lab.zVersion;
		return lab.mnist?.testY.length ?? 0;
	});

	const zNeighbors = $derived.by(() => {
		void lab.zVersion;
		void metric;
		const z = lab.testZ;
		const d = lab.testZd;
		const q = qz;
		// a rebuild can land between the query's encode and the set's
		if (!z || !d || !rows || !q || q.length !== d) return [] as number[];
		return nearestK(z, rows, d, q, SHOW, new Float32Array(rows));
	});

	const pxNeighbors = $derived.by(() => {
		void metric;
		const m = lab.mnist;
		const q = qx;
		if (!m || !rows || !q) return [] as number[];
		return nearestK(m.testX, rows, DIM, q, SHOW, new Float32Array(rows));
	});

	/** How many of the map's eight agree on a digit, and which one — the drawn
	 * query has no true label, so the row's own consensus is the honest read. */
	const consensus = $derived.by(() => {
		const m = lab.mnist;
		if (!m || !zNeighbors.length) return null;
		const count = new Array(10).fill(0);
		for (const i of zNeighbors) count[m.testY[i]]++;
		let best = 0;
		for (let d = 1; d < 10; d++) if (count[d] > count[best]) best = d;
		return { digit: best, share: count[best] / zNeighbors.length };
	});

	// ── the pad ──
	let drawing = false;
	let last: [number, number] | null = null;
	const brushR = $derived(sigma * 2 * (PAD / SIDE));

	function toPad(ev: PointerEvent, el: HTMLCanvasElement): { x: number; y: number } {
		const r = el.getBoundingClientRect();
		return {
			x: Math.max(0, Math.min(PAD, ((ev.clientX - r.left) / r.width) * PAD)),
			y: Math.max(0, Math.min(PAD, ((ev.clientY - r.top) / r.height) * PAD))
		};
	}

	const toGrid = (p: { x: number; y: number }): [number, number] => [
		(p.x / PAD) * SIDE - 0.5,
		(p.y / PAD) * SIDE - 0.5
	];

	function padDown(ev: PointerEvent): void {
		const el = ev.currentTarget as HTMLCanvasElement;
		drawing = true;
		try {
			el.setPointerCapture(ev.pointerId);
		} catch {
			/* capture is a nicety */
		}
		brushAt = toPad(ev, el);
		const p = toGrid(brushAt);
		stampDab(px, p[0], p[1], sigma);
		last = p;
		hasInk = true;
		source = null; // it is the reader's digit now
		strokes += 1;
		scheduleQuery();
	}

	function padMove(ev: PointerEvent): void {
		brushAt = toPad(ev, ev.currentTarget as HTMLCanvasElement);
		if (!drawing || !last) return;
		const p = toGrid(brushAt);
		stampLine(px, last[0], last[1], p[0], p[1], sigma);
		last = p;
		strokes += 1;
		scheduleQuery();
	}

	function padUp(ev: PointerEvent): void {
		drawing = false;
		last = null;
		if (ev.pointerType !== 'mouse') brushAt = null;
		scheduleQuery();
	}

	function padLeave(): void {
		if (!drawing) brushAt = null;
	}

	/** Forget the query. Also called when a read lands after the pad was wiped —
	 * the encode is 100 ms of round trip, and the reader can clear inside it. */
	function dropQuery(): void {
		qx = null;
		qz = null;
		recon = null;
		reconV += 1;
	}

	function clearPad(): void {
		px.fill(0);
		hasInk = false;
		source = null;
		strokes += 1;
		dropQuery();
	}

	/** Load a held-out digit onto the pad, ready to be searched or scribbled on. */
	function loadDigit(i: number): void {
		const m = lab.mnist;
		if (!m) return;
		px.set(m.testX.subarray(i * DIM, (i + 1) * DIM));
		hasInk = true;
		source = i;
		strokes += 1;
		scheduleQuery(true);
	}

	function shuffle(): void {
		if (!rows) return;
		loadDigit(Math.floor(Math.random() * rows));
	}

	// ── reading the pad: centre, encode, reconstruct — throttled ──
	let timer: ReturnType<typeof setTimeout> | null = null;
	let busy = false;
	let queued = false;

	function scheduleQuery(now = false): void {
		if (lab.phase !== 'ready') return;
		if (now) {
			if (timer) clearTimeout(timer);
			timer = null;
			void runQuery();
			return;
		}
		if (timer) return;
		timer = setTimeout(() => {
			timer = null;
			void runQuery();
		}, 110);
	}

	async function runQuery(): Promise<void> {
		const engine = lab.engine;
		if (lab.phase !== 'ready' || !engine) return;
		if (busy) {
			queued = true;
			return;
		}
		busy = true;
		try {
			do {
				queued = false;
				if (!hasInk) {
					dropQuery();
					break;
				}
				const { x } = centered(px);
				const { z, d } = await lab.encode(x, 1);
				const r = await lab.reconstruct(x, 1);
				if (lab.engine !== engine) return; // rebuilt mid-flight
				if (!hasInk) {
					dropQuery(); // wiped while the worker was answering
					break;
				}
				qx = x;
				qz = z.slice(0, d);
				qzd = d;
				recon = r;
				reconV += 1;
			} while (queued);
		} catch {
			/* engine disposed or rebuilding mid-flight */
		} finally {
			busy = false;
		}
	}

	/** The panel average, in slices, so a 784-dimensional scan never blocks a
	 * frame. Pixel space is static: its number is cached and reused. */
	async function measurePanel(): Promise<void> {
		if (panelBusy) return;
		const m = lab.mnist;
		const z = lab.testZ;
		const d = lab.testZd;
		if (!m || !z || !d || !rows) return;
		panelBusy = true;
		const mine = metric;
		const stride = Math.max(1, Math.floor(rows / PANEL));
		const picks: number[] = [];
		for (let i = 0; i < rows && picks.length < PANEL; i += stride) picks.push(i);

		const scratch = new Float32Array(rows);
		let hits = 0;
		for (const q of picks) {
			for (const i of nearestK(z, rows, d, z.subarray(q * d, (q + 1) * d), RANK, scratch, q)) {
				if (m.testY[i] === m.testY[q]) hits++;
			}
		}
		if (mine !== metric) {
			panelBusy = false;
			return;
		}
		zPurity = hits / (picks.length * RANK);

		const cached = pxCache[mine];
		if (cached !== undefined) {
			pxPurity = cached;
			panelBusy = false;
			return;
		}
		let pHits = 0;
		for (let s = 0; s < picks.length; s += 8) {
			for (const q of picks.slice(s, s + 8)) {
				const qv = m.testX.subarray(q * DIM, (q + 1) * DIM);
				for (const i of nearestK(m.testX, rows, DIM, qv, RANK, scratch, q)) {
					if (m.testY[i] === m.testY[q]) pHits++;
				}
			}
			await new Promise((r) => setTimeout(r, 0)); // let the page breathe
			if (mine !== metric) {
				panelBusy = false;
				return;
			}
		}
		const p = pHits / (picks.length * RANK);
		pxCache[mine] = p;
		pxPurity = p;
		panelBusy = false;
	}

	// the map moves under the query as the encoder trains: re-read both
	$effect(() => {
		void lab.zVersion;
		void metric;
		if (lab.phase !== 'ready') return;
		void lab.refreshTestLatents();
		void measurePanel();
		scheduleQuery();
	});

	// ── painters ──
	$effect(() => {
		if (lab.phase !== 'ready') return;
		const paint = () => {
			raf = requestAnimationFrame(paint);
			const m = lab.mnist;
			paintPad();
			tile(reconCanvas, reconStrip, recon, 0, reconV);
			if (!m) return;
			for (const k of SLOTS) {
				const zi = zNeighbors[k];
				const pi = pxNeighbors[k];
				tile(zRow[k], zStrips[k], m.testX, zi, zi);
				tile(pxRow[k], pxStrips[k], m.testX, pi, pi);
			}
		};
		raf = requestAnimationFrame(paint);
		return () => cancelAnimationFrame(raf);
	});

	function paintPad(): void {
		if (!padCanvas) return;
		const { ctx, W, H } = setupCanvas(padCanvas);
		if (W < 8) return;
		const tk = readTokens(padCanvas);
		ctx.fillStyle = tk.surface;
		ctx.fillRect(0, 0, W, H);
		const tiles = padStrip.ensure(px, 1, strokes, tk.surface, tk.ink);
		if (!tiles) return;
		ctx.imageSmoothingEnabled = false;
		ctx.drawImage(tiles[0], 0, 0, W, H);
	}

	function tile(
		el: HTMLCanvasElement | undefined,
		strip: TileStrip,
		data: Float32Array | null,
		i: number | undefined,
		version = i
	): void {
		if (!el) return;
		const { ctx, W, H } = setupCanvas(el);
		if (W < 8) return;
		ctx.clearRect(0, 0, W, H);
		if (!data || i === undefined || i < 0) return;
		const tk = readTokens(el);
		const tiles = strip.ensure(
			data.subarray(i * DIM, (i + 1) * DIM),
			1,
			version ?? i,
			tk.surface,
			tk.ink
		);
		if (!tiles) return;
		ctx.imageSmoothingEnabled = false;
		const s = Math.min(W, H);
		ctx.drawImage(tiles[0], (W - s) / 2, (H - s) / 2, s, s);
	}

	onDestroy(() => {
		if (timer) clearTimeout(timer);
	});

	const pct = (v: number) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '—');
	const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(4) : '—');

	const ROWS = [
		{ key: 'z', label: 'nearest in the map · z' },
		{ key: 'px', label: 'nearest in pixel space · x' }
	] as const;
</script>

<Plate {n} {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'ready'}
			<span>step {lab.step}</span>
			<span aria-hidden="true">·</span>
			<span>val {fmt(lab.valLoss)}</span>
			<span aria-hidden="true">·</span>
			<span>{qzd || lab.testZd}-d embeddings</span>
		{:else if lab.phase === 'loading'}
			<span>warming up…</span>
		{:else if lab.phase === 'error'}
			<span style="color: var(--bad);">error</span>
		{/if}
	{/snippet}

	{#snippet actions()}
		<Btn disabled={lab.phase !== 'ready'} onclick={shuffle} title="Put a held-out digit on the pad">
			<Shuffle size={12} aria-hidden="true" /> Borrow one
		</Btn>
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
						the same two thousand digits, sorted by how near they are
					</span>
				{/if}
			</div>
		{:else}
			<div
				class="grid grid-cols-1 gap-px bg-line-soft lg:grid-cols-[minmax(0,300px)_minmax(0,1fr)]"
			>
				<!-- the query: whatever the reader draws -->
				<div class="flex flex-col gap-3 bg-surface p-4">
					<div class="flex h-4 items-center justify-between gap-2">
						<span class="flex items-center gap-2">
							<span class="eyebrow">draw a query</span>
							<!-- the nib, at its true size, beside the label it belongs to -->
							<span
								class="rounded-full"
								style="width: {3 + sigma * 4}px; height: {3 +
									sigma * 4}px; background: var(--ink-3);"
								aria-hidden="true"
							></span>
						</span>
						{#if hasInk}
							<button
								class="wipe"
								onclick={clearPad}
								title="Clear the pad"
								aria-label="Clear the pad"
							>
								<Eraser size={12} aria-hidden="true" />
							</button>
						{/if}
					</div>

					<div class="flex items-start gap-2.5">
						<div class="vwrap">
							<input
								class="vrange"
								type="range"
								min={SIGMA_MIN}
								max={SIGMA_MAX}
								step="0.05"
								bind:value={sigma}
								aria-label="brush size"
								title="Brush size"
								style="--p: {((sigma - SIGMA_MIN) / (SIGMA_MAX - SIGMA_MIN)) * 100}%;"
							/>
						</div>

						<div class="relative h-56 w-56">
							<canvas
								bind:this={padCanvas}
								class="h-56 w-56 cursor-none touch-none rounded-md border border-line"
								aria-label="drawing pad, 28 by 28 pixels — draw a digit to search with"
								onpointerdown={padDown}
								onpointermove={padMove}
								onpointerup={padUp}
								onpointercancel={padUp}
								onpointerleave={padLeave}
							></canvas>
							{#if brushAt}
								<div
									class="pointer-events-none absolute rounded-full border"
									style="left: {brushAt.x - brushR}px; top: {brushAt.y - brushR}px; width: {brushR *
										2}px; height: {brushR * 2}px; border-color: var(--ink-3);"
									aria-hidden="true"
								></div>
							{/if}
						</div>
					</div>

					<!-- the round trip, as the map understood it -->
					<div class="flex items-center gap-3 border-t border-line-soft pt-3">
						<canvas
							bind:this={reconCanvas}
							class="h-14 w-14 shrink-0 rounded border border-line-soft"
							aria-label="the query, rebuilt from its embedding"
						></canvas>
						<div class="flex min-w-0 flex-col gap-0.5">
							<span class="eyebrow">what the waist kept</span>
							<span class="text-[11.5px] text-ink-3">
								{#if !hasInk}
									an empty pad has nothing to encode
								{:else}
									your stroke, through {qzd || lab.testZd} numbers and back
								{/if}
							</span>
						</div>
					</div>

					<div class="flex flex-wrap items-center gap-x-3 gap-y-2">
						<span class="eyebrow">near means</span>
						<span class="flex flex-wrap items-center gap-1.5" role="group" aria-label="Metric">
							{#each METRICS as m (m)}
								<button
									class="chip"
									class:chip-on={metric === m}
									aria-pressed={metric === m}
									onclick={() => (metric = m)}
								>
									{m}
								</button>
							{/each}
						</span>
					</div>
				</div>

				<!-- the two searches over the same two thousand digits -->
				<div class="flex flex-col gap-3.5 bg-surface p-4">
					<div class="flex flex-1 flex-col justify-center gap-4">
						{#each ROWS as row (row.key)}
							{@const zSide = row.key === 'z'}
							{@const ids = zSide ? zNeighbors : pxNeighbors}
							{@const refs = zSide ? zRow : pxRow}
							<div class="flex flex-col gap-1.5">
								<div class="flex items-baseline justify-between gap-3">
									<span class="eyebrow" style={zSide ? 'color: var(--accent);' : ''}
										>{row.label}</span
									>
									<span class="num text-[10px] text-ink-3">
										{zSide ? `${qzd || lab.testZd} numbers` : '784 numbers'}
									</span>
								</div>
								<div class="grid grid-cols-8 gap-1.5">
									{#each SLOTS as k (k)}
										{@const i = ids[k]}
										<button
											class="tile"
											class:tile-set={i !== undefined}
											disabled={i === undefined}
											title={i === undefined
												? ''
												: `a ${lab.mnist?.testY[i]} — click to draw on it`}
											onclick={() => i !== undefined && loadDigit(i)}
										>
											<canvas
												bind:this={refs[k]}
												class="block aspect-square w-full"
												aria-label={i === undefined ? 'empty' : `a ${lab.mnist?.testY[i]}`}
											></canvas>
											<span class="num tag">{i === undefined ? '' : lab.mnist?.testY[i]}</span>
										</button>
									{/each}
								</div>
							</div>
						{/each}

						<!-- what the map made of the stroke, in one line -->
						<div class="flex items-baseline gap-2 font-serif text-[13.5px] text-ink-2 italic">
							{#if !hasInk}
								Draw something on the pad — anything digit-shaped — and both rows will fill.
							{:else if consensus}
								The map put your stroke among the {consensus.digit}s
								<span class="num text-[11.5px] text-ink-3 not-italic">
									{Math.round(consensus.share * 100)}% of its eight{source !== null
										? ` · borrowed #${source}`
										: ''}
								</span>
							{/if}
						</div>
					</div>

					<!-- the verdict, over a whole panel of held-out queries -->
					<div class="mt-auto flex flex-col gap-2 border-t border-line-soft pt-3">
						<div class="flex flex-wrap items-baseline justify-between gap-x-3">
							<span class="eyebrow">neighbours of the same digit</span>
							<span class="num text-[10px] text-ink-3">
								top {RANK} · {PANEL} held-out queries · chance 10%
							</span>
						</div>
						{#each [{ key: 'z', label: 'in the map', v: zPurity, color: 'var(--accent)' }, { key: 'px', label: 'in pixels', v: pxPurity, color: 'var(--ink-3)' }] as m (m.key)}
							<div class="flex items-center gap-3">
								<span class="w-[68px] shrink-0 text-[11.5px] text-ink-2">{m.label}</span>
								<span class="meter">
									<i
										style="width: {Number.isFinite(m.v)
											? Math.round(m.v * 100)
											: 0}%; background: {m.color};"
									></i>
								</span>
								<span class="num w-[42px] shrink-0 text-right text-[12.5px] text-ink"
									>{pct(m.v)}</span
								>
							</div>
						{/each}
					</div>
				</div>
			</div>
		{/if}
	</div>
</Plate>

<style>
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
	.chip-on:hover {
		color: var(--paper);
		background: color-mix(in srgb, var(--ink) 88%, var(--paper));
		border-color: color-mix(in srgb, var(--ink) 88%, var(--paper));
	}

	/* a neighbour: the digit, with its label tucked in the corner */
	.tile {
		position: relative;
		display: block;
		width: 100%;
		border: 1px solid var(--line-soft);
		border-radius: 4px;
		overflow: hidden;
		background: var(--surface);
		transition:
			transform 100ms ease,
			border-color 100ms ease;
	}
	.tile-set:hover {
		transform: translateY(-1px);
		border-color: var(--ink-3);
	}
	.tag {
		position: absolute;
		right: 2px;
		bottom: 1px;
		font-size: 9px;
		line-height: 1;
		color: var(--ink-3);
	}

	.meter {
		position: relative;
		display: block;
		flex: 1;
		height: 6px;
		border-radius: 3px;
		background: var(--line-soft);
		overflow: hidden;
	}
	.meter i {
		position: absolute;
		inset: 0 auto 0 0;
		display: block;
		border-radius: 3px;
		transition: width 220ms ease;
	}

	/* the clear button: a quiet square beside the pad's label */
	.wipe {
		display: grid;
		place-items: center;
		width: 20px;
		height: 20px;
		border-radius: 5px;
		background: var(--surface);
		border: 1px solid var(--line);
		color: var(--ink-2);
		cursor: pointer;
	}
	.wipe:hover {
		color: var(--ink);
		border-color: var(--ink-3);
	}

	/* a vertical range input, by rotation — the layout box stays 16 × 224 */
	.vwrap {
		position: relative;
		width: 16px;
		height: 224px;
	}
	.vrange {
		appearance: none;
		position: absolute;
		top: 50%;
		left: 50%;
		width: 224px;
		height: 16px;
		transform: translate(-50%, -50%) rotate(-90deg);
		background: transparent;
		cursor: pointer;
		padding: 0;
		border: none;
	}
	.vrange::-webkit-slider-runnable-track {
		height: 2px;
		border-radius: 1px;
		background: linear-gradient(to right, var(--line) var(--p), var(--line-soft) var(--p));
	}
	.vrange::-moz-range-track {
		height: 2px;
		border-radius: 1px;
		background: linear-gradient(to right, var(--line) var(--p), var(--line-soft) var(--p));
	}
	.vrange::-webkit-slider-thumb {
		appearance: none;
		margin-top: -4px;
		height: 10px;
		width: 10px;
		border-radius: 50%;
		background: var(--surface);
		border: 1.5px solid var(--ink-3);
		transition: transform 100ms ease;
	}
	.vrange::-moz-range-thumb {
		height: 10px;
		width: 10px;
		border-radius: 50%;
		background: var(--surface);
		border: 1.5px solid var(--ink-3);
		transition: transform 100ms ease;
	}
	.vrange:hover::-webkit-slider-thumb {
		transform: scale(1.2);
		border-color: var(--ink-2);
	}
	.vrange:focus-visible {
		outline: none;
	}
	.vrange:focus-visible::-webkit-slider-thumb {
		box-shadow: 0 0 0 4px color-mix(in srgb, var(--ink-3) 25%, transparent);
	}
</style>

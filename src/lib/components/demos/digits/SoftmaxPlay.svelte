<script lang="ts">
	// The softmax, by hand — a compact port of the SoftMax Explainer: a bank of
	// scores drawn from a gaussian, the belief they become, and the shape of
	// each as a sideways histogram. Pure arithmetic; no engine, no data.
	// Drag inside the score chart to bend one score by hand.
	import { RotateCcw, ArrowDownWideNarrow } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { readTokens, hexRgb, type Rgb } from './common';
	import { themePulse, watchTheme } from './digits-context.svelte';

	interface Props {
		title: string;
		caption?: string;
	}
	let { title, caption }: Props = $props();

	// ten scores by default — one per digit class, the case this chapter needs
	let count = $state(10);
	let mean = $state(0);
	let sigma = $state(1.2);
	let logT = $state(0); // temperature on a log scale, τ = 10^logT
	let sorted = $state(false);
	let seed = $state(1);
	let dragged = $state<Record<number, number>>({}); // hand-bent scores, by index

	watchTheme();

	const T = $derived(10 ** logT);

	/** Box–Muller from a seeded rng, so a resample is a deliberate act. */
	function gaussians(k: number, s: number): number[] {
		let x = s * 2654435761 + 12345;
		const rand = () => {
			x = (x + 0x6d2b79f5) | 0;
			let t = Math.imul(x ^ (x >>> 15), 1 | x);
			t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
			return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
		};
		return Array.from({ length: k }, () => {
			const u = Math.max(rand(), 1e-9);
			return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
		});
	}

	// one long pool, sliced — moving the count slider never re-rolls the dice
	const POOL = 512;
	const pool = $derived(gaussians(POOL, seed));
	const raw = $derived.by(() => {
		const r = pool.slice(0, count).map((z) => mean + z * sigma);
		for (const [i, v] of Object.entries(dragged)) if (+i < count) r[+i] = v;
		return r;
	});
	/** Display order → index into `raw`; sorting is a view, never a re-labeling. */
	const order = $derived(
		sorted ? [...raw.keys()].sort((a, b) => raw[b] - raw[a]) : [...raw.keys()]
	);
	const scores = $derived(order.map((i) => raw[i]));
	const probs = $derived.by(() => {
		const z = scores.map((v) => v / T);
		const mx = Math.max(...z);
		const e = z.map((v) => Math.exp(v - mx));
		const sum = e.reduce((a, b) => a + b, 0);
		return e.map((v) => v / sum);
	});
	const topProb = $derived(probs.length ? Math.max(...probs) : 0);
	const entropy = $derived(-probs.reduce((a, p) => a + (p > 1e-12 ? p * Math.log(p) : 0), 0));
	// how flat the belief is: 1 = perfectly uniform, 0 = all on one score
	const flatness = $derived(count > 1 ? entropy / Math.log(count) : 0);

	function resample() {
		seed += 1;
		dragged = {};
	}

	// ── painting ──────────────────────────────────────────────────────────────
	interface Geom {
		ctx: CanvasRenderingContext2D;
		W: number;
		H: number;
	}
	function setup(el: HTMLCanvasElement): Geom | null {
		const dpr = Math.min(devicePixelRatio || 1, 2);
		const W = el.clientWidth;
		const H = el.clientHeight;
		if (!W || !H) return null;
		if (el.width !== Math.round(W * dpr) || el.height !== Math.round(H * dpr)) {
			el.width = Math.round(W * dpr);
			el.height = Math.round(H * dpr);
		}
		const ctx = el.getContext('2d');
		if (!ctx) return null;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, W, H);
		return { ctx, W, H };
	}

	const PADL = 34; // room for the axis readout
	const PADV = 8;

	/** Attachment that repaints on state changes and on layout resize. */
	function painter(draw: (el: HTMLCanvasElement) => void) {
		return (el: HTMLCanvasElement) => {
			draw(el);
			if (typeof ResizeObserver === 'undefined') return;
			const ro = new ResizeObserver(() => draw(el));
			ro.observe(el);
			return () => ro.disconnect();
		};
	}

	function rgba(c: Rgb, a: number) {
		return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
	}

	function axisLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, c: string) {
		ctx.fillStyle = c;
		ctx.font = '9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
		ctx.textAlign = 'right';
		ctx.textBaseline = 'middle';
		ctx.fillText(text, x, y);
	}

	/** The scores: bars rising and falling from a zero rule. */
	function paintScores(el: HTMLCanvasElement) {
		void themePulse.tick;
		const vals = scores;
		const g = setup(el);
		if (!g) return;
		const { ctx, W, H } = g;
		const tk = readTokens(el);
		// one rule across the chapter: warm is positive, blue is negative
		const pos = hexRgb(tk.warm);
		const neg = hexRgb(tk.accent);
		const lim = Math.max(1e-6, ...vals.map(Math.abs));
		const y0 = PADV + ((lim - 0) / (2 * lim)) * (H - 2 * PADV);

		ctx.strokeStyle = tk.line;
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(PADL, Math.round(y0) + 0.5);
		ctx.lineTo(W, Math.round(y0) + 0.5);
		ctx.stroke();
		axisLabel(ctx, `+${lim.toFixed(1)}`, PADL - 6, PADV + 3, tk.ink3);
		axisLabel(ctx, '0', PADL - 6, y0, tk.ink3);
		axisLabel(ctx, `−${lim.toFixed(1)}`, PADL - 6, H - PADV - 3, tk.ink3);

		const span = W - PADL;
		const slot = span / vals.length;
		const bw = Math.max(1, Math.min(26, slot * 0.62));
		vals.forEach((v, i) => {
			const cx = PADL + slot * (i + 0.5);
			const h = (Math.abs(v) / (2 * lim)) * (H - 2 * PADV);
			ctx.fillStyle = rgba(v >= 0 ? pos : neg, 0.82);
			ctx.fillRect(cx - bw / 2, v >= 0 ? y0 - h : y0, bw, Math.max(0.7, h));
		});
	}

	/** The belief: bars from a floor, auto-scaled to the winner. */
	function paintProbs(el: HTMLCanvasElement) {
		void themePulse.tick;
		const vals = probs;
		const g = setup(el);
		if (!g) return;
		const { ctx, W, H } = g;
		const tk = readTokens(el);
		const c = hexRgb(tk.cats[2]);
		const lim = Math.max(1e-6, ...vals);
		const base = H - PADV;

		ctx.strokeStyle = tk.line;
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(PADL, Math.round(base) + 0.5);
		ctx.lineTo(W, Math.round(base) + 0.5);
		ctx.stroke();
		axisLabel(ctx, `${(lim * 100).toFixed(0)}%`, PADL - 6, PADV + 3, tk.ink3);
		axisLabel(ctx, '0', PADL - 6, base, tk.ink3);

		const span = W - PADL;
		const slot = span / vals.length;
		const bw = Math.max(1, Math.min(26, slot * 0.62));
		vals.forEach((p, i) => {
			const cx = PADL + slot * (i + 0.5);
			const h = (p / lim) * (H - 2 * PADV);
			ctx.fillStyle = rgba(c, p === lim ? 0.95 : 0.62);
			ctx.fillRect(cx - bw / 2, base - h, bw, Math.max(0.7, h));
		});
	}

	/** Sideways histogram sharing the bar chart's vertical axis. */
	function histogram(vals: number[], lo: number, hi: number, bins: number): number[] {
		const out = new Array(bins).fill(0);
		if (hi - lo < 1e-12) return out;
		for (const v of vals) {
			const k = Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)));
			out[k] += 1;
		}
		return out;
	}

	const bins = $derived(Math.min(28, Math.max(6, Math.ceil(Math.sqrt(count)) * 2)));

	function paintScoreHist(el: HTMLCanvasElement) {
		void themePulse.tick;
		const vals = scores;
		const g = setup(el);
		if (!g) return;
		const { ctx, W, H } = g;
		const tk = readTokens(el);
		const pos = hexRgb(tk.warm);
		const neg = hexRgb(tk.accent);
		const lim = Math.max(1e-6, ...vals.map(Math.abs));
		const hist = histogram(vals, -lim, lim, bins);
		const mx = Math.max(1, ...hist);
		const rowH = (H - 2 * PADV) / bins;
		hist.forEach((k, i) => {
			// bin 0 is the most negative — rows grow downward from +lim
			const y = PADV + (bins - 1 - i) * rowH;
			const mid = -lim + ((i + 0.5) / bins) * 2 * lim;
			ctx.fillStyle = rgba(mid >= 0 ? pos : neg, 0.55);
			ctx.fillRect(0, y + 0.4, (k / mx) * W, Math.max(0.8, rowH - 0.8));
		});
	}

	function paintProbHist(el: HTMLCanvasElement) {
		void themePulse.tick;
		const vals = probs;
		const g = setup(el);
		if (!g) return;
		const { ctx, W, H } = g;
		const tk = readTokens(el);
		const c = hexRgb(tk.cats[2]);
		const lim = Math.max(1e-6, ...vals);
		const hist = histogram(vals, 0, lim, bins);
		const mx = Math.max(1, ...hist);
		const rowH = (H - 2 * PADV) / bins;
		hist.forEach((k, i) => {
			const y = PADV + (bins - 1 - i) * rowH;
			ctx.fillStyle = rgba(c, 0.55);
			ctx.fillRect(0, y + 0.4, (k / mx) * W, Math.max(0.8, rowH - 0.8));
		});
	}

	// ── bending a score by hand ───────────────────────────────────────────────
	let bending = false;

	function bendAt(ev: PointerEvent) {
		const el = ev.currentTarget as HTMLCanvasElement;
		const r = el.getBoundingClientRect();
		const span = r.width - PADL;
		const slot = Math.floor(((ev.clientX - r.left - PADL) / span) * count);
		if (slot < 0 || slot >= count) return;
		const lim = Math.max(1e-6, ...scores.map(Math.abs));
		const u = (ev.clientY - r.top - PADV) / (r.height - 2 * PADV);
		const v = lim - Math.min(1, Math.max(0, u)) * 2 * lim;
		dragged = { ...dragged, [order[slot]]: Math.round(v * 10) / 10 };
	}

	function bendDown(ev: PointerEvent) {
		if (count > 64) return; // too dense to aim at one bar
		bending = true;
		try {
			(ev.currentTarget as HTMLCanvasElement).setPointerCapture(ev.pointerId);
		} catch {
			/* pointer already released — the drag still works, just uncaptured */
		}
		bendAt(ev);
	}
	function bendMove(ev: PointerEvent) {
		if (bending) bendAt(ev);
	}
	function bendUp() {
		bending = false;
	}
</script>

<Plate {title} {caption}>
	{#snippet status()}
		<span>top {(topProb * 100).toFixed(1)}%</span>
		<span aria-hidden="true">·</span>
		<span>flatness {flatness.toFixed(2)}</span>
		<span aria-hidden="true">·</span>
		<span>Σp = 1.00</span>
	{/snippet}

	{#snippet actions()}
		<Btn
			onclick={() => (sorted = !sorted)}
			kind={sorted ? 'primary' : 'ghost'}
			title="Show the scores in descending order"
		>
			<ArrowDownWideNarrow size={12} aria-hidden="true" /> Sort
		</Btn>
		<Btn onclick={resample} title="Draw a fresh set of scores">
			<RotateCcw size={12} aria-hidden="true" /> Resample
		</Btn>
	{/snippet}

	<div class="grid grid-cols-1 gap-x-6 gap-y-4 p-4 sm:p-5 md:grid-cols-[168px_1fr]">
		<!-- the dials, spread to stand exactly as tall as the charts -->
		<div class="flex flex-col justify-between gap-3 py-0.5">
			<!-- the greek rides in the mono readout: the eyebrow uppercases, and
			     uppercase μ σ τ read as M Σ T -->
			<Slider
				label="how many"
				bind:value={count}
				min={2}
				max={512}
				step={1}
				format={(v) => `n = ${v}`}
				tone="ink"
			/>
			<Slider
				label="mean"
				bind:value={mean}
				min={-2}
				max={2}
				step={0.05}
				format={(v) => `μ = ${v.toFixed(2)}`}
			/>
			<Slider
				label="spread"
				bind:value={sigma}
				min={0}
				max={3}
				step={0.05}
				format={(v) => `σ = ${v.toFixed(2)}`}
			/>
			<Slider
				label="temperature"
				bind:value={logT}
				min={-1}
				max={0.7}
				step={0.02}
				format={() => `τ = ${T.toFixed(2)}`}
				tone="warm"
			/>
		</div>

		<!-- the two charts, each with the shape of its own numbers beside it -->
		<div class="flex flex-col gap-3">
			<div>
				<div class="mb-1 flex items-baseline justify-between gap-3">
					<span class="eyebrow" style="color: var(--warm);">the scores · logits z</span>
					{#if count <= 64}
						<span class="text-[10.5px] text-ink-3 italic">drag a bar to bend it</span>
					{/if}
				</div>
				<div class="grid grid-cols-[1fr_72px] gap-1.5">
					<canvas
						class="block h-[122px] w-full touch-none"
						class:cursor-ns-resize={count <= 64}
						aria-label="the scores, as bars above and below zero"
						onpointerdown={bendDown}
						onpointermove={bendMove}
						onpointerup={bendUp}
						onpointercancel={bendUp}
						{@attach painter(paintScores)}
					></canvas>
					<canvas
						class="block h-[122px] w-full"
						aria-label="distribution of the scores"
						{@attach painter(paintScoreHist)}
					></canvas>
				</div>
			</div>

			<div>
				<div class="mb-1 flex items-baseline justify-between gap-3">
					<span class="eyebrow" style="color: var(--cat-2);">the belief · softmax(z / τ)</span>
					<span class="eyebrow text-ink-3">their shape →</span>
				</div>
				<div class="grid grid-cols-[1fr_72px] gap-1.5">
					<canvas
						class="block h-[122px] w-full"
						aria-label="the softmax probabilities, as bars"
						{@attach painter(paintProbs)}
					></canvas>
					<canvas
						class="block h-[122px] w-full"
						aria-label="distribution of the probabilities"
						{@attach painter(paintProbHist)}
					></canvas>
				</div>
			</div>
		</div>
	</div>
</Plate>

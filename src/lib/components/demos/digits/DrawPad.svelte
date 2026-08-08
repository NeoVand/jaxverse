<script lang="ts">
	// Draw your own. Borrows the engine the classifier plate trained (through
	// digits-context). Every stroke is painted straight into the 784-float
	// buffer the model reads, re-centered by mass before predicting — MNIST
	// digits are centered, and the model has never seen one that isn't.
	// The two charts on the right mirror the softmax plate above: raw scores
	// over a zero rule, then the belief they become.
	import { onDestroy } from 'svelte';
	import { Eraser } from 'lucide-svelte';
	import { lab, themePulse, watchTheme } from './digits-context.svelte';
	import { SIGMA_MAX, SIGMA_MIN } from '$lib/data/brush';
	import { plateAnchor, plateLabel } from '$lib/data/plates';
	import {
		DEFAULT_SIGMA,
		DIM,
		SIDE,
		argmax,
		blit,
		centerShift,
		divergingImage,
		hexRgb,
		inkImage,
		readTokens,
		shiftImage,
		softmax,
		stampDab,
		stampLine,
		type Rgb
	} from './common';

	const PAD = 224; // pad CSS size — 8 css px per 28-grid pixel

	const px = new Float32Array(DIM); // the drawing itself — model input

	/** A 3 waiting on the pad, so the plate arrives with something to read.
	 * Two right-facing arcs, stamped with the same brush the reader gets. */
	function seedDigit() {
		const arc = (cx: number, cy: number, r: number, a0: number, a1: number) => {
			const n = 22;
			let prev: [number, number] | null = null;
			for (let i = 0; i <= n; i++) {
				const t = a0 + ((a1 - a0) * i) / n;
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
	let logits = $state<number[] | null>(null);
	let probs = $state<number[] | null>(null);
	let predictedClass = $state<number | null>(null);
	let saliency: Float32Array | null = null; // signed ∂score/∂pixel, un-centered
	let salVersion = $state(0);
	/** Brush-head preview position in pad CSS px; null = pointer off the pad. */
	let brushAt = $state<{ x: number; y: number } | null>(null);

	watchTheme();

	const winnerPct = $derived(
		probs && predictedClass !== null ? Math.round(probs[predictedClass] * 100) : null
	);
	// the visible extent of the dab (core + soft edge), in pad CSS px
	const brushR = $derived(sigma * 2 * (PAD / SIDE));

	// ── the brush ──
	let drawing = false;
	let last: [number, number] | null = null;

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

	function padDown(ev: PointerEvent) {
		const el = ev.currentTarget as HTMLCanvasElement;
		drawing = true;
		el.setPointerCapture(ev.pointerId);
		brushAt = toPad(ev, el);
		const p = toGrid(brushAt);
		stampDab(px, p[0], p[1], sigma);
		last = p;
		hasInk = true;
		strokes += 1;
		schedulePredict();
	}

	function padMove(ev: PointerEvent) {
		brushAt = toPad(ev, ev.currentTarget as HTMLCanvasElement);
		if (!drawing || !last) return;
		const p = toGrid(brushAt);
		stampLine(px, last[0], last[1], p[0], p[1], sigma);
		last = p;
		strokes += 1;
		schedulePredict();
	}

	function padUp(ev: PointerEvent) {
		drawing = false;
		last = null;
		if (ev.pointerType !== 'mouse') brushAt = null; // no hover after a touch lifts
		schedulePredict();
	}

	function padLeave() {
		if (!drawing) brushAt = null;
	}

	function clearPad() {
		px.fill(0);
		hasInk = false;
		strokes += 1;
		logits = null;
		probs = null;
		predictedClass = null;
		saliency = null;
		salVersion += 1;
	}

	// ── prediction, throttled to ~100 ms while the pointer moves ──
	let predictTimer: ReturnType<typeof setTimeout> | null = null;
	let busy = false;
	let queued = false;

	function schedulePredict() {
		if (lab.phase !== 'ready' || predictTimer) return;
		predictTimer = setTimeout(() => {
			predictTimer = null;
			void runPredict();
		}, 100);
	}

	async function runPredict() {
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
					logits = null;
					probs = null;
					predictedClass = null;
					saliency = null;
					salVersion += 1;
					break;
				}
				const { dx, dy } = centerShift(px);
				const centered = shiftImage(px, dx, dy);
				const z = await engine.predict(centered, 1, 16);
				const cls = argmax(z, 0, 10);
				const g = await engine.inputGrad(centered, cls);
				if (lab.engine !== engine) return;
				logits = Array.from({ length: 10 }, (_, d) => z[d]);
				probs = softmax(z, 0, 10);
				predictedClass = cls;
				// shift the evidence back so it overlays the stroke as drawn
				saliency = shiftImage(g, -dx, -dy);
				salVersion += 1;
			} while (queued);
		} catch {
			/* engine disposed mid-flight */
		} finally {
			busy = false;
		}
	}

	// keep the verdict honest while the classifier plate trains on, and read the seeded
	// digit as soon as there is an engine to read it with
	$effect(() => {
		void lab.version;
		void lab.phase;
		if (hasInk) schedulePredict();
	});

	// ── the two images ────────────────────────────────────────────────────────
	function paintPad(el: HTMLCanvasElement) {
		void strokes;
		void themePulse.tick;
		blit(el, [inkImage(px, 0, hexRgb(readTokens(el).ink))], 224);
	}

	/** A soft halo around the ink: dilate the stroke, then blur the edge. */
	function inkHalo(src: Float32Array, grow = 3, soft = 2): Float32Array {
		const dil = new Float32Array(DIM);
		for (let y = 0; y < SIDE; y++) {
			for (let x = 0; x < SIDE; x++) {
				let m = 0;
				for (let j = -grow; j <= grow; j++) {
					const yy = y + j;
					if (yy < 0 || yy >= SIDE) continue;
					for (let i = -grow; i <= grow; i++) {
						const xx = x + i;
						if (xx < 0 || xx >= SIDE) continue;
						m = Math.max(m, src[yy * SIDE + xx]);
					}
				}
				dil[y * SIDE + x] = m;
			}
		}
		const out = new Float32Array(DIM);
		const win = (2 * soft + 1) ** 2;
		for (let y = 0; y < SIDE; y++) {
			for (let x = 0; x < SIDE; x++) {
				let s = 0;
				for (let j = -soft; j <= soft; j++) {
					const yy = Math.min(SIDE - 1, Math.max(0, y + j));
					for (let i = -soft; i <= soft; i++) {
						const xx = Math.min(SIDE - 1, Math.max(0, x + i));
						s += dil[yy * SIDE + xx];
					}
				}
				out[y * SIDE + x] = Math.min(1, (s / win) * 1.6);
			}
		}
		return out;
	}

	function paintEvidence(el: HTMLCanvasElement) {
		void salVersion;
		void themePulse.tick;
		const tk = readTokens(el);
		const layers = [inkImage(px, 0, hexRgb(tk.ink), 0.28)];
		if (saliency) {
			// The gradient is dense over all 784 pixels, but its far field belongs
			// to the model's re-centered frame — it slides sideways every time the
			// stroke's centre of mass moves. Keep only the neighbourhood of the ink,
			// which is what the caption promises and what actually holds still.
			const halo = inkHalo(px);
			const masked = new Float32Array(DIM);
			for (let i = 0; i < DIM; i++) masked[i] = saliency[i] * halo[i];
			layers.push(divergingImage(masked, 0, hexRgb(tk.accent), hexRgb(tk.warm), 0.9));
		}
		blit(el, layers, 224);
	}

	// ── the two charts ────────────────────────────────────────────────────────
	const PADL = 30; // room for the axis readout
	const PADV = 7;
	const PADB = 14; // the digit axis, under the belief chart

	function setup(el: HTMLCanvasElement) {
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

	const rgba = (c: Rgb, a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
	const MONO = '9.5px ui-monospace, SFMono-Regular, Menlo, monospace';

	function axisLabel(ctx: CanvasRenderingContext2D, text: string, y: number, c: string) {
		ctx.fillStyle = c;
		ctx.font = MONO;
		ctx.textAlign = 'right';
		ctx.textBaseline = 'middle';
		ctx.fillText(text, PADL - 6, y);
	}

	/** The raw scores, rising and falling from a zero rule. */
	function paintLogits(el: HTMLCanvasElement) {
		void themePulse.tick;
		const vals = logits;
		const g = setup(el);
		if (!g) return;
		const { ctx, W, H } = g;
		const tk = readTokens(el);
		// the book-wide rule for signed values: ultramarine up, vermilion down
		const pos = hexRgb(tk.accent);
		const neg = hexRgb(tk.warm);
		const lim = Math.max(1e-6, ...(vals ?? [1]).map(Math.abs));
		const y0 = H / 2;

		ctx.strokeStyle = tk.line;
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(PADL, Math.round(y0) + 0.5);
		ctx.lineTo(W, Math.round(y0) + 0.5);
		ctx.stroke();
		if (vals) {
			const mag = lim >= 10 ? lim.toFixed(0) : lim.toFixed(1);
			axisLabel(ctx, `+${mag}`, PADV + 3, tk.ink3);
			axisLabel(ctx, `−${mag}`, H - PADV - 3, tk.ink3);
		}
		axisLabel(ctx, '0', y0, tk.ink3);
		if (!vals) return;

		const slot = (W - PADL) / 10;
		const bw = Math.min(22, slot * 0.6);
		vals.forEach((v, d) => {
			const cx = PADL + slot * (d + 0.5);
			const h = (Math.abs(v) / lim) * (H / 2 - PADV);
			ctx.fillStyle = rgba(v >= 0 ? pos : neg, d === predictedClass ? 0.95 : 0.55);
			ctx.fillRect(cx - bw / 2, v >= 0 ? y0 - h : y0, bw, Math.max(0.8, h));
		});
	}

	/** The belief: bars from a floor, with the ten classes named beneath. */
	function paintProbs(el: HTMLCanvasElement) {
		void themePulse.tick;
		const vals = probs;
		const g = setup(el);
		if (!g) return;
		const { ctx, W, H } = g;
		const tk = readTokens(el);
		const c = hexRgb(tk.cats[2]);
		const base = H - PADB;
		const top = PADV;

		ctx.strokeStyle = tk.line;
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(PADL, Math.round(base) + 0.5);
		ctx.lineTo(W, Math.round(base) + 0.5);
		ctx.stroke();
		axisLabel(ctx, '100%', top + 3, tk.ink3);
		axisLabel(ctx, '0', base, tk.ink3);

		const slot = (W - PADL) / 10;
		const bw = Math.min(22, slot * 0.6);
		for (let d = 0; d < 10; d++) {
			const cx = PADL + slot * (d + 0.5);
			const win = d === predictedClass;
			if (vals) {
				const h = vals[d] * (base - top);
				ctx.fillStyle = rgba(c, win ? 0.95 : 0.5);
				ctx.fillRect(cx - bw / 2, base - h, bw, Math.max(0.8, h));
			}
			ctx.fillStyle = win ? tk.ink : tk.ink3;
			ctx.font = win
				? '600 10.5px ui-monospace, SFMono-Regular, Menlo, monospace'
				: '10px ui-monospace, SFMono-Regular, Menlo, monospace';
			ctx.textAlign = 'center';
			ctx.textBaseline = 'alphabetic';
			ctx.fillText(String(d), cx, H - 2.5);
		}
	}

	onDestroy(() => {
		if (predictTimer) clearTimeout(predictTimer);
	});
</script>

{#if lab.phase !== 'ready'}
	<div class="px-6 py-12 text-center">
		<p class="m-0 font-serif text-[15px] text-ink-2 italic">
			The pad borrows the classifier's weights —
			<a
				href="#{plateAnchor('classifier')}"
				class="underline underline-offset-2"
				style="text-decoration-color: var(--accent);"
				>train the classifier in {plateLabel('digits', 'classifier')}</a
			> first, then come back and draw.
		</p>
	</div>
{:else}
	<div class="flex flex-wrap items-start gap-x-7 gap-y-6 p-4 sm:p-5">
		<!-- the pad, its brush, and the same square of evidence beside it -->
		<div class="flex items-start gap-3">
			<!-- brush size: a dot the size of the nib, on a track as tall as the pad -->
			<div>
				<div class="mb-1.5 flex h-4 items-center justify-center">
					<span
						class="rounded-full"
						style="width: {3 + sigma * 4}px; height: {3 + sigma * 4}px; background: var(--ink-3);"
						aria-hidden="true"
					></span>
				</div>
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
			</div>

			<div class="group">
				<div class="mb-1.5 flex h-4 items-baseline">
					<span class="eyebrow">draw a digit</span>
				</div>
				<div class="relative h-56 w-56">
					<canvas
						class="h-56 w-56 cursor-none touch-none rounded-md border border-line"
						aria-label="drawing pad, 28 by 28 pixels — draw a digit with your pointer"
						onpointerdown={padDown}
						onpointermove={padMove}
						onpointerup={padUp}
						onpointercancel={padUp}
						onpointerleave={padLeave}
						{@attach paintPad}
					></canvas>
					{#if brushAt}
						<div
							class="pointer-events-none absolute rounded-full border"
							style="left: {brushAt.x - brushR}px; top: {brushAt.y - brushR}px; width: {brushR *
								2}px; height: {brushR * 2}px; border-color: var(--ink-3);"
							aria-hidden="true"
						></div>
					{/if}
					{#if hasInk}
						<button
							class="wipe absolute top-2 right-2 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
							onclick={clearPad}
							title="Clear the pad"
							aria-label="Clear the pad"
						>
							<Eraser size={13} aria-hidden="true" />
						</button>
					{/if}
				</div>
			</div>

			<div>
				<div class="mb-1.5 flex h-4 items-baseline gap-2.5">
					<span class="eyebrow">the evidence</span>
					<span class="flex items-center gap-1 text-[10.5px] text-ink-3">
						<i class="dot" style="background: var(--accent);"></i>for
						<i class="dot ml-1" style="background: var(--warm);"></i>against
					</span>
				</div>
				<canvas
					class="h-56 w-56 rounded-md border border-line-soft"
					aria-label="evidence map: gradient of the winning score with respect to each pixel"
					{@attach paintEvidence}
				></canvas>
			</div>
		</div>

		<!-- the scores it produces, and the belief they become -->
		<div class="flex min-w-[300px] flex-1 flex-col gap-3.5">
			<div>
				<div class="mb-1 flex h-4 items-baseline justify-between gap-3">
					<span class="eyebrow" style="color: var(--warm);">the scores · logits z</span>
					<span class="text-[10.5px] text-ink-3">one per class, any size</span>
				</div>
				<canvas
					class="block h-[92px] w-full"
					aria-label="the ten raw scores, as bars above and below zero"
					{@attach painter(paintLogits)}
				></canvas>
			</div>

			<div>
				<div class="mb-1 flex h-4 items-baseline justify-between gap-3">
					<span class="eyebrow" style="color: var(--cat-2);">the belief · softmax(z)</span>
					<span class="font-serif text-[13px] text-ink-2 italic">
						{#if predictedClass !== null && winnerPct !== null}
							it reads a {predictedClass} — {winnerPct}% sure
						{:else}
							draw, and the bars will move
						{/if}
					</span>
				</div>
				<canvas
					class="block h-[100px] w-full"
					aria-label="the softmax probabilities over the ten classes, as bars"
					{@attach painter(paintProbs)}
				></canvas>
			</div>
		</div>
	</div>
{/if}

<style>
	/* a vertical range input, by rotation — the layout box stays 16 × 224 */
	.vwrap {
		position: relative;
		width: 16px;
		height: 224px;
	}
	.vrange {
		-webkit-appearance: none;
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
		-webkit-appearance: none;
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
		box-shadow: var(--focus-ring);
	}

	/* the clear button: a quiet square that only appears over the pad */
	.wipe {
		display: grid;
		place-items: center;
		width: 24px;
		height: 24px;
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

	.dot {
		display: inline-block;
		width: 7px;
		height: 7px;
		border-radius: 50%;
	}
</style>

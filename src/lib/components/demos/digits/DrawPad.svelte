<script lang="ts">
	// Plate III — draw your own. Borrows the engine Plate II trained (through
	// digits-context). Every stroke is painted straight into the 784-float
	// buffer the model reads, re-centered by mass before predicting — MNIST
	// digits are centered, and the model has never seen one that isn't.
	import { onDestroy } from 'svelte';
	import { Eraser } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { lab, themePulse, watchTheme } from './digits-context.svelte';
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
		stampLine
	} from './common';

	const PAD = 224; // pad CSS size — 8 css px per 28-grid pixel

	const px = new Float32Array(DIM); // the drawing itself — model input
	let strokes = $state(0); // bumped per stamp → repaints the pad
	let hasInk = $state(false);
	let sigma = $state(DEFAULT_SIGMA);
	let probs = $state<number[] | null>(null);
	let predictedClass = $state<number | null>(null);
	let saliency: Float32Array | null = null; // signed ∂score/∂pixel, un-centered
	let salVersion = $state(0);
	/** Brush-head preview position in pad CSS px; null = pointer off the pad. */
	let brushAt = $state<{ x: number; y: number } | null>(null);

	watchTheme();

	const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
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
					probs = null;
					predictedClass = null;
					saliency = null;
					salVersion += 1;
					break;
				}
				const { dx, dy } = centerShift(px);
				const centered = shiftImage(px, dx, dy);
				const logits = await engine.predict(centered, 1, 16);
				const cls = argmax(logits, 0, 10);
				const g = await engine.inputGrad(centered, cls);
				if (lab.engine !== engine) return;
				probs = softmax(logits, 0, 10);
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

	// keep the verdict honest while Plate II trains on: re-read the model
	$effect(() => {
		void lab.version;
		if (hasInk) schedulePredict();
	});

	// ── painting (attachments re-run when strokes / theme / saliency change) ──
	function paintPad(el: HTMLCanvasElement) {
		void strokes;
		void themePulse.tick;
		blit(el, [inkImage(px, 0, hexRgb(readTokens(el).ink))], 224);
	}

	function paintEvidence(el: HTMLCanvasElement) {
		void salVersion;
		void themePulse.tick;
		const tk = readTokens(el);
		const layers = [inkImage(px, 0, hexRgb(tk.ink), 0.28)];
		if (saliency) {
			layers.push(divergingImage(saliency, 0, hexRgb(tk.warm), hexRgb(tk.accent), 0.85));
		}
		blit(el, layers, 112);
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
				href="#plate-classifier"
				class="underline underline-offset-2"
				style="text-decoration-color: var(--accent);">train the classifier in Plate II</a
			> first, then come back and draw.
		</p>
	</div>
{:else}
	<div class="flex flex-wrap items-start gap-x-8 gap-y-6 p-4 sm:p-5">
		<!-- the pad -->
		<div class="flex w-56 flex-col gap-2">
			<span class="eyebrow">draw a digit · 28 × 28</span>
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
			</div>
			<div class="flex items-center gap-3">
				<Btn onclick={clearPad} disabled={!hasInk}>
					<Eraser size={12} aria-hidden="true" /> Clear
				</Btn>
				<span class="min-w-0 flex-1">
					<Slider
						label="brush σ"
						bind:value={sigma}
						min={0.5}
						max={2.5}
						step={0.05}
						format={(v) => v.toFixed(2)}
						tone="ink"
					/>
				</span>
			</div>
			{#if lab.step === 0}
				<span class="text-[11.5px] text-ink-3 italic">untrained — train Plate II first</span>
			{/if}
		</div>

		<!-- the evidence -->
		<div class="flex w-28 flex-col gap-2">
			<span class="eyebrow">evidence</span>
			<canvas
				class="h-28 w-28 rounded-md border border-line-soft"
				aria-label="evidence map: gradient of the winning score with respect to each pixel"
				{@attach paintEvidence}
			></canvas>
			<p class="m-0 text-[11px] leading-snug text-ink-3">
				<span style="color: var(--warm);">warm</span> pixels argue for the verdict,
				<span style="color: var(--accent);">blue</span> against it
			</p>
		</div>

		<!-- the belief -->
		<div class="flex min-w-60 flex-1 flex-col gap-2">
			<span class="eyebrow">its belief · softmax over ten classes</span>
			<div class="flex items-end gap-1.5">
				{#each DIGITS as d (d)}
					{@const p = probs ? probs[d] : 0}
					{@const winner = predictedClass === d && probs !== null}
					<div class="flex min-w-0 flex-1 flex-col items-center gap-1">
						<div class="relative h-28 w-full">
							{#if winner}
								<span
									class="num absolute inset-x-0 text-center text-[10px]"
									style="bottom: calc({Math.min(
										86,
										Math.max(2, p * 100)
									)}% + 2px); color: var(--ink-2);"
								>
									{Math.round(p * 100)}%
								</span>
							{/if}
							<div
								class="absolute inset-x-0 bottom-0 rounded-t-[3px] transition-[height] duration-150"
								style="height: {Math.max(2, p * 100)}%; background: var(--cat-{d}); opacity: {winner
									? 1
									: 0.4};"
							></div>
						</div>
						<span
							class="num text-[12px]"
							style="color: {winner ? 'var(--ink)' : 'var(--ink-3)'}; font-weight: {winner
								? 620
								: 400};">{d}</span
						>
					</div>
				{/each}
			</div>
			<p class="m-0 h-5 font-serif text-[13.5px] text-ink-2 italic">
				{#if predictedClass !== null && winnerPct !== null}
					it reads a {predictedClass} — {winnerPct}% sure
				{:else}
					draw on the pad and the bars will move
				{/if}
			</p>
		</div>
	</div>
{/if}

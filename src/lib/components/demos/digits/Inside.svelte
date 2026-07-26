<script lang="ts">
	// Plate IV — inside the machine. Two instruments on the shared engine,
	// deliberately the same height: an explorer over every learned layer (each
	// unit traced back into a 28 × 28 image) and a confusion matrix over the
	// full 2,000-row test set, whose cells open a popover of real mistakes.
	// Refreshes are throttled and only run while the plate is on screen, so the
	// training loop in Plate II keeps its rhythm.
	import { onDestroy } from 'svelte';
	import type { LayerWeights } from '$lib/nn/engine';
	import { lab, themePulse, watchTheme } from './digits-context.svelte';
	import { DIM, argmax, blit, divergingImage, hexRgb, inkImage, readTokens } from './common';

	const MAX_TILES = 36; // 6 × 6 — stands as tall as the matrix beside it
	const HID_COLS = 6;
	const OUT_COLS = 4; // ten class templates, larger, with two cells to spare
	const TOP_K = 24; // test digits averaged for the "what excites it" view
	const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
	const POP_W = 240; // popover width, fixed so it can be clamped exactly

	type Cell = { r: number; c: number; x: number; top: number; bottom: number };
	type View = 'weights' | 'excites';

	// Raw, not proxied: both are replaced wholesale on every read and their
	// arrays are walked in million-iteration loops, where a deep state proxy
	// costs three orders of magnitude.
	let ws = $state.raw<LayerWeights[] | null>(null); // every layer's weight matrix
	let acts = $state.raw<{ layers: Float32Array[]; widths: number[] } | null>(null);
	let layer = $state(0); // which one the explorer is showing
	let view = $state<View>('weights');
	let conf = $state<Int32Array | null>(null); // 100 counts, true × predicted
	let examples: number[][] = []; // up to 6 test indices per cell
	let hover = $state<Cell | null>(null);
	let pinned = $state<Cell | null>(null); // click/tap sticks
	let wrapW = $state(0);
	let visible = $state(false);
	let busy = false;
	let lastRun = 0;
	let timer: ReturnType<typeof setTimeout> | null = null;

	watchTheme();

	const testN = $derived(lab.phase === 'ready' && lab.mnist ? lab.mnist.testY.length : 0);

	// ── the explorer ──────────────────────────────────────────────────────────
	// A unit in the first layer owns 784 incoming weights, so it simply *is* an
	// image. A unit deeper in cannot be looked at directly, but the chain of
	// weight matrices behind it can be multiplied out — W₁W₂…W_L — which carries
	// it back to pixel space. That product is the unit's linear shadow: honest
	// about which ink it accumulates, silent about the bends between. The other
	// view skips the algebra and asks the test set instead — the digits that
	// drive the unit hardest, averaged.
	const tmpl = $derived.by(() => {
		const all = ws;
		if (view !== 'weights' || !all || all.length === 0) return null;
		const last = Math.min(layer, all.length - 1);
		let p = all[0].w;
		let cols = all[0].outDim;
		for (let k = 1; k <= last; k++) {
			const bw = all[k].w;
			const bc = all[k].outDim;
			const out = new Float32Array(DIM * bc);
			for (let i = 0; i < DIM; i++) {
				const rowP = i * cols;
				const rowO = i * bc;
				for (let j = 0; j < cols; j++) {
					const a = p[rowP + j];
					if (a === 0) continue;
					const rowB = j * bc;
					for (let o = 0; o < bc; o++) out[rowO + o] += a * bw[rowB + o];
				}
			}
			p = out;
			cols = bc;
		}
		return { m: p, cols };
	});

	/** Mean image of the TOP_K test digits that drive unit `k` of `l` hardest. */
	function excitesImage(l: number, k: number): Float32Array | null {
		const a = acts;
		const m = lab.mnist;
		if (!a || !m || !a.layers[l]) return null;
		const h = a.layers[l];
		const w = a.widths[l];
		const n = Math.min(m.testY.length, Math.floor(h.length / w));
		const best: number[] = []; // row indices, activation descending
		const vals: number[] = [];
		for (let i = 0; i < n; i++) {
			const v = h[i * w + k];
			if (best.length === TOP_K && v <= vals[TOP_K - 1]) continue;
			let at = vals.length;
			while (at > 0 && vals[at - 1] < v) at--;
			vals.splice(at, 0, v);
			best.splice(at, 0, i);
			if (best.length > TOP_K) {
				vals.pop();
				best.pop();
			}
		}
		const avg = new Float32Array(DIM);
		if (!best.length) return avg;
		const x = m.testX;
		for (const i of best) {
			const off = i * DIM;
			for (let px = 0; px < DIM; px++) avg[px] += x[off + px];
		}
		let mx = 1e-6;
		for (let px = 0; px < DIM; px++) {
			avg[px] /= best.length;
			mx = Math.max(mx, avg[px]);
		}
		for (let px = 0; px < DIM; px++) avg[px] = Math.min(1, avg[px] / mx);
		return avg;
	}

	const nLayers = $derived(ws ? ws.length : 0);
	const layerIds = $derived([...Array(nLayers).keys()]);
	const isReadout = $derived(nLayers > 0 && layer === nLayers - 1);
	const gridCols = $derived(isReadout ? OUT_COLS : HID_COLS);
	const unitCount = $derived(ws && ws[layer] ? ws[layer].outDim : 0);
	const tileIds = $derived.by(() => {
		const n = isReadout ? 10 : Math.min(MAX_TILES, unitCount);
		return [...Array(n).keys()];
	});

	// ── the matrix ────────────────────────────────────────────────────────────
	const maxDiag = $derived.by(() => {
		if (!conf) return 0;
		let m = 0;
		for (let d = 0; d < 10; d++) m = Math.max(m, conf[d * 11]);
		return m;
	});
	const maxOff = $derived.by(() => {
		if (!conf) return 0;
		let m = 0;
		for (let r = 0; r < 10; r++) {
			for (let c = 0; c < 10; c++) if (r !== c) m = Math.max(m, conf[r * 10 + c]);
		}
		return m;
	});
	const sel = $derived(hover ?? pinned);
	const selExamples = $derived(sel && conf ? (examples[sel.r * 10 + sel.c] ?? []) : []);
	const popAbove = $derived(sel ? sel.r >= 5 : false);
	const popX = $derived(
		sel ? Math.max(POP_W / 2, Math.min(Math.max(POP_W, wrapW) - POP_W / 2, sel.x)) : 0
	);

	// refresh when the weights move — but only on screen, at most every ~1.5 s
	$effect(() => {
		void lab.version;
		if (lab.phase !== 'ready' || !visible || timer) return;
		const wait = Math.max(0, 1500 - (performance.now() - lastRun));
		timer = setTimeout(() => {
			timer = null;
			void refresh();
		}, wait);
	});

	// a rebuilt model has a different number of layers — never point past the end
	$effect(() => {
		if (nLayers > 0 && layer > nLayers - 1) layer = nLayers - 1;
	});

	function observeVisible(el: HTMLElement) {
		if (typeof IntersectionObserver === 'undefined') return;
		const io = new IntersectionObserver(
			(es) => {
				visible = es[0]?.isIntersecting ?? false;
			},
			{ rootMargin: '250px' }
		);
		io.observe(el);
		return () => io.disconnect();
	}

	let queued = false;
	async function refresh() {
		const engine = lab.engine;
		const m = lab.mnist;
		if (!engine || !m) return;
		if (busy) {
			queued = true; // a newer version arrived mid-read — go again after
			return;
		}
		busy = true;
		try {
			do {
				queued = false;
				const nTest = m.testY.length;
				// activations() ends with the output layer, so this one pass feeds
				// the explorer and the matrix both
				const [layers, run] = await Promise.all([
					engine.weights(),
					engine.activations(m.testX, nTest, 1024)
				]);
				if (lab.engine !== engine) return;
				ws = layers;
				acts = run;
				const logits = run.layers[run.layers.length - 1];
				const counts = new Int32Array(100);
				const ex: number[][] = Array.from({ length: 100 }, () => []);
				for (let i = 0; i < nTest; i++) {
					const cell = m.testY[i] * 10 + argmax(logits, i * 10, 10);
					counts[cell] += 1;
					if (ex[cell].length < 6) ex[cell].push(i);
				}
				examples = ex;
				conf = counts;
				lastRun = performance.now();
			} while (queued);
		} catch {
			/* engine disposed mid-flight */
		} finally {
			busy = false;
		}
	}

	function cellStyle(r: number, c: number, count: number): string {
		// a neutral wash, so the empty cells still read as a grid in both themes
		if (!count) return 'background: color-mix(in srgb, var(--ink-3) 9%, transparent);';
		const diag = r === c;
		const m = diag ? maxDiag : maxOff;
		const a = m > 0 ? count / m : 0;
		const pct = Math.round(10 + 68 * Math.min(1, a));
		return `background: color-mix(in srgb, ${diag ? 'var(--good)' : 'var(--bad)'} ${pct}%, var(--surface));`;
	}

	/** Where the popover should sit, in coordinates of the matrix wrapper. */
	function geometry(el: HTMLElement, r: number, c: number): Cell {
		const wrap = el.closest('.matrix') as HTMLElement | null;
		const a = el.getBoundingClientRect();
		const b = wrap?.getBoundingClientRect();
		return {
			r,
			c,
			x: b ? a.left - b.left + a.width / 2 : 0,
			top: b ? a.top - b.top : 0,
			bottom: b ? a.bottom - b.top : 0
		};
	}

	function open(ev: Event, r: number, c: number) {
		hover = geometry(ev.currentTarget as HTMLElement, r, c);
	}

	function pin(ev: Event, r: number, c: number) {
		pinned =
			pinned && pinned.r === r && pinned.c === c
				? null
				: geometry(ev.currentTarget as HTMLElement, r, c);
	}

	function unitTile(k: number) {
		return (el: HTMLCanvasElement) => {
			void themePulse.tick;
			const tk = readTokens(el);
			if (view === 'excites') {
				const avg = excitesImage(layer, k);
				if (avg) blit(el, [inkImage(avg, 0, hexRgb(tk.ink))], 64);
				return;
			}
			const t = tmpl;
			if (!t) return;
			const src = t.m;
			const cols = t.cols;
			const vals = new Float32Array(DIM);
			for (let i = 0; i < DIM; i++) vals[i] = src[i * cols + k];
			blit(el, [divergingImage(vals, 0, hexRgb(tk.warm), hexRgb(tk.accent), 0.95)], 64);
		};
	}

	function exampleTile(idx: number) {
		return (el: HTMLCanvasElement) => {
			void themePulse.tick;
			const m = lab.mnist;
			if (!m) return;
			blit(el, [inkImage(m.testX, idx * DIM, hexRgb(readTokens(el).ink))], 34);
		};
	}

	onDestroy(() => {
		if (timer) clearTimeout(timer);
	});
</script>

<div {@attach observeVisible}>
	{#if lab.phase !== 'ready'}
		<div class="px-6 py-12 text-center">
			<p class="m-0 font-serif text-[15px] text-ink-2 italic">
				There is nothing inside yet —
				<a
					href="#plate-classifier"
					class="underline underline-offset-2"
					style="text-decoration-color: var(--accent);">train the classifier in Plate II</a
				> and this plate will open the machine up.
			</p>
		</div>
	{:else}
		<div class="grid grid-cols-1 gap-x-10 gap-y-8 p-4 sm:p-5 md:grid-cols-2">
			<!-- every layer, traced back into images -->
			<section>
				<div class="mb-1.5 flex h-4 items-baseline justify-between gap-3">
					<span class="eyebrow">
						{#if isReadout}
							the readout · ten classes
						{:else if layer === 0}
							first layer · {tileIds.length} of {unitCount} units
						{:else}
							layer {layer + 1} · {tileIds.length} of {unitCount} units
						{/if}
					</span>
					<span class="flex shrink-0 items-center gap-1 text-[10.5px] text-ink-3">
						{#if view === 'weights'}
							<i class="dot" style="background: var(--warm);"></i>rewards ink
							<i class="dot ml-1" style="background: var(--accent);"></i>penalizes
						{:else}
							<i class="dot" style="background: var(--ink-2);"></i>top {TOP_K} test digits, averaged
						{/if}
					</span>
				</div>

				<div class="mb-2 flex h-6 items-center justify-between gap-3">
					<span class="flex items-center gap-1" role="group" aria-label="Which layer to look at">
						<span class="eyebrow mr-0.5">layer</span>
						{#each layerIds as k (k)}
							<button
								class="chip"
								class:chip-on={layer === k}
								aria-pressed={layer === k}
								onclick={() => (layer = k)}
							>
								{k === nLayers - 1 ? 'out' : k + 1}
							</button>
						{/each}
					</span>
					<span class="flex shrink-0 items-center gap-1" role="group" aria-label="What to draw">
						<button
							class="chip"
							class:chip-on={view === 'weights'}
							aria-pressed={view === 'weights'}
							title="The unit's own weights, carried back to pixel space"
							onclick={() => (view = 'weights')}
						>
							weights
						</button>
						<button
							class="chip"
							class:chip-on={view === 'excites'}
							aria-pressed={view === 'excites'}
							title="The test digits that drive the unit hardest, averaged"
							onclick={() => (view = 'excites')}
						>
							excites it
						</button>
					</span>
				</div>

				<div class="grid gap-2" style="grid-template-columns: repeat({gridCols}, minmax(0, 1fr));">
					{#each tileIds as k (k)}
						<div>
							{#if isReadout}
								<div class="num mb-1 text-center text-[11.5px]" style="color: var(--cat-{k});">
									{k}
								</div>
							{/if}
							<canvas
								class="aspect-square w-full rounded-[3px] border border-line-soft"
								aria-label={view === 'excites'
									? `the ${TOP_K} test digits that most excite ${isReadout ? `the score for ${k}` : `unit ${k}`}, averaged`
									: isReadout
										? `the readout's template for the digit ${k}, as a 28 by 28 image`
										: `unit ${k} of layer ${layer + 1}, traced back to a 28 by 28 image`}
								{@attach unitTile(k)}
							></canvas>
						</div>
					{/each}
					{#if isReadout}
						<div class="col-span-2 flex items-end">
							<p class="m-0 font-serif text-[12.5px] leading-snug text-ink-2 italic">
								{#if view === 'excites'}
									the digits it calls each name — averaged, so the shape it settles on for a class
									is the shape you see
								{:else}
									the machine's idea of each digit — the ink each class pays for, summed over every
									path through the layers
								{/if}
							</p>
						</div>
					{/if}
				</div>
			</section>

			<!-- the confusion matrix on the held-out rows -->
			<section>
				<div class="mb-1.5 flex h-4 items-baseline justify-between gap-3">
					<span class="eyebrow">confusion · all {testN.toLocaleString('en-US')} test digits</span>
					<span class="flex shrink-0 items-center gap-1 text-[10.5px] text-ink-3">
						<i class="dot" style="background: var(--good);"></i>right
						<i class="dot ml-1" style="background: var(--bad);"></i>wrong
					</span>
				</div>

				<div class="mb-2 flex h-6 items-center justify-between gap-3">
					<span class="num text-[10px] text-ink-3">rows: truth · columns: its reading</span>
					<span class="shrink-0 text-[10.5px] text-ink-3">hover a cell for the digits</span>
				</div>

				<!-- mouseleave only clears the hover preview; the cells themselves are focusable buttons -->
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div class="matrix relative" bind:clientWidth={wrapW}>
					<div
						class="grid gap-[2px]"
						style="grid-template-columns: 16px repeat(10, minmax(0, 1fr));"
						onmouseleave={() => (hover = null)}
					>
						<span></span>
						{#each DIGITS as c (c)}
							<span class="num pb-[3px] text-center text-[10.5px]" style="color: var(--cat-{c});"
								>{c}</span
							>
						{/each}
						{#each DIGITS as r (r)}
							<span
								class="num self-center pr-1 text-right text-[10.5px]"
								style="color: var(--cat-{r});">{r}</span
							>
							{#each DIGITS as c (c)}
								{@const count = conf ? conf[r * 10 + c] : 0}
								<button
									type="button"
									class="cell aspect-square w-full rounded-[2px]"
									class:cell-on={sel?.r === r && sel?.c === c}
									style={cellStyle(r, c, count)}
									aria-label="true {r} read as {c}: {count} times"
									onmouseenter={(ev) => open(ev, r, c)}
									onfocus={(ev) => open(ev, r, c)}
									onclick={(ev) => pin(ev, r, c)}
								></button>
							{/each}
						{/each}
					</div>

					{#if sel && conf}
						{@const n = conf[sel.r * 10 + sel.c]}
						<div
							class="pop pointer-events-none absolute z-10"
							style="width: {POP_W}px; left: {popX}px; top: {popAbove
								? sel.top - 8
								: sel.bottom + 8}px; transform: translate(-50%, {popAbove ? '-100%' : '0'});"
						>
							<div class="num text-[10.5px] text-ink-2">
								<span style="color: var(--cat-{sel.r});">{sel.r}</span>
								→ read as
								<span style="color: var(--cat-{sel.c});">{sel.c}</span>
								· {n}
								{n === 1 ? 'time' : 'times'}
							</div>
							{#if selExamples.length}
								<div class="mt-1.5 flex gap-1">
									{#each selExamples as idx (idx)}
										<canvas
											class="h-[34px] w-[34px] rounded-[3px] border border-line-soft"
											aria-hidden="true"
											{@attach exampleTile(idx)}
										></canvas>
									{/each}
								</div>
							{:else}
								<div class="mt-0.5 font-serif text-[12.5px] text-ink-3 italic">
									it never makes this mistake
								</div>
							{/if}
						</div>
					{/if}
				</div>
			</section>
		</div>
	{/if}
</div>

<style>
	.cell {
		border: none;
		padding: 0;
		cursor: default;
		transition: box-shadow 80ms ease;
	}
	.cell-on {
		box-shadow: inset 0 0 0 1.5px var(--ink-2);
	}

	.pop {
		padding: 8px;
		border-radius: 6px;
		background: var(--surface);
		border: 1px solid var(--line);
		box-shadow: 0 8px 24px color-mix(in srgb, var(--ink) 14%, transparent);
	}

	.dot {
		display: inline-block;
		width: 7px;
		height: 7px;
		border-radius: 50%;
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

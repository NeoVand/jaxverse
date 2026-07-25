<script lang="ts">
	// Plate IV — inside the machine. Two honest instruments on the shared
	// engine: the first layer's incoming weights reshaped back into images,
	// and a confusion matrix over the full 2,000-row test set. Refreshes are
	// throttled and only run while the plate is on screen, so the training
	// loop in Plate II keeps its rhythm.
	import { onDestroy } from 'svelte';
	import { lab, themePulse, watchTheme } from './digits-context.svelte';
	import { DIM, argmax, blit, divergingImage, hexRgb, inkImage, readTokens } from './common';

	const UNITS = 24; // first-layer units shown
	const UNIT_IDS = [...Array(UNITS).keys()];
	const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

	let w1 = $state<Float32Array | null>(null); // 784 × 128, row-major inDim × outDim
	let w1Cols = 128;
	let conf = $state<Int32Array | null>(null); // 100 counts, true × predicted
	let examples: number[][] = []; // up to 6 test indices per cell — read on select
	let hover = $state<{ r: number; c: number } | null>(null);
	let pinned = $state<{ r: number; c: number } | null>(null); // click/tap sticks
	let visible = $state(false);
	let busy = false;
	let lastRun = 0;
	let timer: ReturnType<typeof setTimeout> | null = null;

	watchTheme();

	const testN = $derived(lab.phase === 'ready' && lab.mnist ? lab.mnist.testY.length : 0);
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
				const [ws, logits] = await Promise.all([
					engine.weights(),
					engine.predict(m.testX, nTest, 1024)
				]);
				if (lab.engine !== engine) return;
				w1Cols = ws[0].outDim;
				w1 = ws[0].w;
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
		if (!count) return 'background: color-mix(in srgb, var(--surface-2) 65%, transparent);';
		const diag = r === c;
		const m = diag ? maxDiag : maxOff;
		const a = m > 0 ? count / m : 0;
		const pct = Math.round(10 + 68 * Math.min(1, a));
		return `background: color-mix(in srgb, ${diag ? 'var(--good)' : 'var(--bad)'} ${pct}%, var(--surface));`;
	}

	function weightTile(k: number) {
		return (el: HTMLCanvasElement) => {
			void themePulse.tick;
			const w = w1;
			if (!w) return;
			const tk = readTokens(el);
			const vals = new Float32Array(DIM);
			for (let i = 0; i < DIM; i++) vals[i] = w[i * w1Cols + k];
			blit(el, [divergingImage(vals, 0, hexRgb(tk.warm), hexRgb(tk.accent), 0.95)], 56);
		};
	}

	function exampleTile(idx: number) {
		return (el: HTMLCanvasElement) => {
			void themePulse.tick;
			const m = lab.mnist;
			if (!m) return;
			blit(el, [inkImage(m.testX, idx * DIM, hexRgb(readTokens(el).ink))], 56);
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
			<!-- first-layer weights, reshaped back into images -->
			<section>
				<span class="eyebrow">first-layer weights · 24 of 128 units</span>
				<div class="mt-3 grid grid-cols-6 gap-2" style="max-width: 24rem;">
					{#each UNIT_IDS as k (k)}
						<canvas
							class="aspect-square w-full rounded-[3px] border border-line-soft"
							aria-label="incoming weights of hidden unit {k}, drawn as a 28 by 28 image"
							{@attach weightTile(k)}
						></canvas>
					{/each}
				</div>
				<p class="mt-3 mb-0 font-serif text-[13px] text-ink-2 italic">
					the machine's first questions — each unit asks how much the image resembles this pattern: <span
						style="color: var(--warm);">warm</span
					>
					rewards ink,
					<span style="color: var(--accent);">blue</span> penalizes it
				</p>
			</section>

			<!-- the confusion matrix on the held-out rows -->
			<section>
				<span class="eyebrow">confusion · all {testN.toLocaleString('en-US')} test digits</span>
				<div class="num mt-3 mb-1 text-[10px] text-ink-3">rows: truth · columns: its reading</div>
				<!-- mouseleave only clears the hover preview; the cells themselves are focusable buttons -->
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					class="grid gap-[2px]"
					style="grid-template-columns: auto repeat(10, minmax(0, 1fr)); max-width: 22rem;"
					onmouseleave={() => (hover = null)}
				>
					<span></span>
					{#each DIGITS as c (c)}
						<span class="num pb-0.5 text-center text-[10.5px]" style="color: var(--cat-{c});"
							>{c}</span
						>
					{/each}
					{#each DIGITS as r (r)}
						<span
							class="num self-center pr-1.5 text-right text-[10.5px]"
							style="color: var(--cat-{r});">{r}</span
						>
						{#each DIGITS as c (c)}
							{@const count = conf ? conf[r * 10 + c] : 0}
							<button
								type="button"
								class="cell h-[24px] w-full rounded-[2px]"
								class:cell-on={sel?.r === r && sel?.c === c}
								style={cellStyle(r, c, count)}
								aria-label="true {r} read as {c}: {count} times"
								onmouseenter={() => (hover = { r, c })}
								onfocus={() => (hover = { r, c })}
								onclick={() =>
									(pinned = pinned && pinned.r === r && pinned.c === c ? null : { r, c })}
							></button>
						{/each}
					{/each}
				</div>

				<!-- fixed detail panel — fills on hover, sticks on click/tap; never overflows -->
				<div class="mt-3 min-h-[86px] rounded-md border border-line-soft p-2.5">
					{#if sel && conf}
						{@const n = conf[sel.r * 10 + sel.c]}
						<div class="num text-[11px] text-ink-2">
							true {sel.r} → read as {sel.c} · {n}
							{n === 1 ? 'time' : 'times'} of {testN.toLocaleString('en-US')}
						</div>
						<div class="mt-2 flex flex-wrap items-center gap-1.5">
							{#each selExamples as idx (idx)}
								<canvas
									class="h-10 w-10 rounded-[3px] border border-line-soft"
									aria-label="a {sel.r} the model read as a {sel.c}"
									{@attach exampleTile(idx)}
								></canvas>
							{/each}
							{#if selExamples.length === 0}
								<span class="font-serif text-[12.5px] text-ink-3 italic"
									>it never makes this mistake</span
								>
							{/if}
						</div>
					{:else}
						<div class="font-serif text-[12.5px] text-ink-3 italic">
							hover or tap a cell to open it here — try 4 read as 9, and 3 read as 5
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
</style>

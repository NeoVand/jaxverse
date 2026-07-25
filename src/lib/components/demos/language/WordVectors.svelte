<script lang="ts">
	// Plate I — words become vectors. Skip-gram with negative sampling, trained
	// live on the story corpus, on the main thread (embeddings.ts). The stage
	// is a PCA shadow of the 16-D space that reorganizes as training runs;
	// the rail lists cosine neighbors; the analogy row does a − b + c for real.
	import { onDestroy } from 'svelte';
	import { Pause, Play } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { loadCorpus } from '$lib/data/corpus';
	import { buildWordCorpus, SkipGram, fitPca2, project, type WordCorpus } from './embeddings';

	type Phase = 'idle' | 'loading' | 'running' | 'paused' | 'done' | 'error';
	let phase = $state<Phase>('idle');
	let errorMsg = $state('');
	let tick = $state(0); // bumps once per training chunk
	let pairsSeen = $state(0);
	let lossNow = $state(NaN);
	let rate = $state(0); // pairs/s including display pacing
	let pts = $state<Float32Array>(new Float32Array(0)); // [2·V] projected coords
	let vocab = $state<string[]>([]);
	let selected = $state<number | null>(null);
	let selA = $state('boy');
	let selB = $state('he');
	let selC = $state('she');

	let words: WordCorpus | null = null; // deliberately not $state
	let sg: SkipGram | null = null;
	let gen = 0;
	let lastFit = 0;
	let lastBeat = 0;

	// ≈ 60k pairs/s — structure appears in a few seconds
	const PACING = { chunk: 4000, pace: 66 };

	async function boot() {
		if (phase !== 'idle' && phase !== 'error') return;
		phase = 'loading';
		errorMsg = '';
		const myGen = ++gen;
		try {
			const corpus = await loadCorpus(); // shared cache with every other plate
			if (myGen !== gen) return;
			const text = Array.from(corpus.tokens, (t) => corpus.vocab[t]).join('');
			words = buildWordCorpus(text, 220);
			sg = new SkipGram(words, { dim: 16, seed: 7 });
			vocab = words.vocab;
			selected = Math.max(0, words.vocab.indexOf('dog'));
			refit(true);
			phase = 'running';
			void run();
		} catch (err) {
			if (myGen !== gen) return;
			errorMsg = err instanceof Error ? err.message : String(err);
			phase = 'error';
		}
	}

	async function run() {
		const s = sg;
		if (!s) return;
		const myGen = gen;
		lastBeat = performance.now();
		while (myGen === gen && s.pairsSeen < s.budget) {
			const { chunk, pace } = PACING;
			s.trainPairs(Math.min(chunk, s.budget - s.pairsSeen));
			const now = performance.now();
			const inst = chunk / Math.max(1e-3, (now - lastBeat) / 1000);
			rate = rate ? rate * 0.8 + inst * 0.2 : inst;
			lastBeat = now;
			pairsSeen = s.pairsSeen;
			lossNow = s.lossEma;
			tick++;
			refit();
			await new Promise((r) => setTimeout(r, pace));
		}
		if (myGen !== gen) return;
		refit(true);
		phase = 'done';
	}

	function pauseTraining() {
		if (phase !== 'running') return;
		gen++;
		refit(true);
		phase = 'paused';
	}
	function resumeTraining() {
		if (phase !== 'paused' || !sg) return;
		phase = 'running';
		void run();
	}

	onDestroy(() => {
		gen++;
	});

	/** PCA refit, throttled — 220×16 power iteration costs ~2 ms. */
	function refit(force = false) {
		const s = sg;
		const w = words;
		if (!s || !w) return;
		const now = performance.now();
		if (!force && now - lastFit < 120) return;
		lastFit = now;
		const V = w.vocab.length;
		const vecs: number[][] = [];
		for (let i = 0; i < V; i++) vecs.push([...s.vector(i)]);
		const pca = fitPca2(vecs);
		const out = new Float32Array(2 * V);
		for (let i = 0; i < V; i++) {
			const [x, y] = project(vecs[i], pca);
			out[2 * i] = x;
			out[2 * i + 1] = y;
		}
		pts = out;
	}

	// ── stage geometry ──
	const H = 420;
	let stageW = $state(0);
	const W = $derived(stageW || 640);
	const placed = $derived.by(() => {
		const V = vocab.length;
		if (V === 0 || pts.length !== 2 * V) return [];
		let minX = Infinity;
		let maxX = -Infinity;
		let minY = Infinity;
		let maxY = -Infinity;
		for (let i = 0; i < V; i++) {
			minX = Math.min(minX, pts[2 * i]);
			maxX = Math.max(maxX, pts[2 * i]);
			minY = Math.min(minY, pts[2 * i + 1]);
			maxY = Math.max(maxY, pts[2 * i + 1]);
		}
		const pad = 36;
		// uniform scale: the map keeps its geometry, only fits the frame
		const s = Math.min(
			(W - 2 * pad) / Math.max(1e-6, maxX - minX),
			(H - 2 * pad) / Math.max(1e-6, maxY - minY)
		);
		const cx = (minX + maxX) / 2;
		const cy = (minY + maxY) / 2;
		return Array.from({ length: V }, (_, i) => ({
			x: W / 2 + (pts[2 * i] - cx) * s,
			y: H / 2 - (pts[2 * i + 1] - cy) * s
		}));
	});
	/** Greedy collision thinning in frequency order; the selected word always shows. */
	const shown = $derived.by(() => {
		const out = new Array<boolean>(placed.length).fill(false);
		const boxes: Array<{ x0: number; x1: number; y0: number; y1: number }> = [];
		const tryPlace = (i: number): boolean => {
			const w = vocab[i].length * 6.4 + 8;
			const h = 14;
			const b = {
				x0: placed[i].x - w / 2,
				x1: placed[i].x + w / 2,
				y0: placed[i].y - h / 2,
				y1: placed[i].y + h / 2
			};
			for (const o of boxes)
				if (b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0) return false;
			boxes.push(b);
			return true;
		};
		if (selected !== null && placed[selected]) {
			out[selected] = true;
			tryPlace(selected);
		}
		for (let i = 0; i < placed.length; i++) {
			if (out[i]) continue;
			out[i] = tryPlace(i);
		}
		return out;
	});

	// ── readouts ──
	const neighborRows = $derived.by(() => {
		void tick;
		if (!sg || selected === null) return [];
		return sg.neighbors(selected, 8).map((n) => ({ id: n.id, word: vocab[n.id], sim: n.sim }));
	});
	const neighborSet = $derived(new Set(neighborRows.map((n) => n.id)));
	const alphaVocab = $derived([...vocab].sort());
	const analogyRows = $derived.by(() => {
		void tick;
		if (!sg) return null;
		const ia = vocab.indexOf(selA);
		const ib = vocab.indexOf(selB);
		const ic = vocab.indexOf(selC);
		if (ia < 0 || ib < 0 || ic < 0) return null;
		return sg.analogy(ia, ib, ic, 3).map((r) => ({ word: vocab[r.id], sim: r.sim }));
	});

	const fmtK = (n: number) => (n >= 1000 ? `${Math.round(n / 1000)}k` : String(Math.round(n)));
</script>

<Plate
	n={1}
	title="Words become vectors"
	caption="A PCA shadow of a 16-dimensional space, redrawn as skip-gram training runs — each word learns to predict its neighbors, and words used alike drift together: animals with animals, names with names, he beside she. Every vector is trained here, in your tab, on this book's story corpus."
>
	{#snippet status()}
		{#if phase === 'idle' || phase === 'loading'}
			<span>{phase === 'idle' ? 'waiting' : 'loading corpus…'}</span>
		{:else if phase === 'error'}
			<span style="color: var(--bad);">error</span>
		{:else}
			<span>{fmtK(pairsSeen)} pairs</span>
			<span aria-hidden="true">·</span>
			<span>loss {Number.isFinite(lossNow) ? lossNow.toFixed(3) : '—'}</span>
			<span aria-hidden="true">·</span>
			<span>{fmtK(rate)} pairs/s</span>
			{#if phase === 'done'}
				<span aria-hidden="true">·</span>
				<span>done</span>
			{/if}
		{/if}
	{/snippet}

	<div use:inview={() => void boot()}>
		{#if phase === 'idle' || phase === 'loading' || phase === 'error'}
			<div class="flex min-h-56 flex-col items-center justify-center gap-3 px-6 py-8">
				{#if phase === 'error'}
					<p class="num max-w-md text-center text-[12px]" style="color: var(--bad);">
						{errorMsg || 'the corpus failed to load'}
					</p>
					<Btn onclick={() => void boot()}>Retry</Btn>
				{:else}
					<span class="num text-[12px] text-ink-3">
						{phase === 'idle'
							? 'the embedding lab loads when you reach it'
							: 'fetching the story corpus (about 1.5 MB)…'}
					</span>
				{/if}
			</div>
		{:else}
			<!-- transport -->
			<div
				class="flex flex-wrap items-center gap-x-4 gap-y-2.5 border-b border-line-soft px-4 py-3"
			>
				{#if phase === 'running'}
					<Btn kind="primary" onclick={pauseTraining}>
						<Pause size={13} aria-hidden="true" /> Pause
					</Btn>
				{:else if phase === 'paused'}
					<Btn kind="primary" onclick={resumeTraining}>
						<Play size={13} aria-hidden="true" /> Resume
					</Btn>
				{/if}
				<span class="num text-[11px] text-ink-3">
					skip-gram · 16 dims · top 220 words · {phase === 'done'
						? 'budget reached — the geometry has settled'
						: 'training on your CPU, right now'}
				</span>
			</div>

			<div class="grid grid-cols-1 gap-px bg-line-soft md:grid-cols-[minmax(0,1fr)_15rem]">
				<!-- the map -->
				<div class="bg-surface" bind:clientWidth={stageW}>
					<svg
						width="100%"
						height={H}
						viewBox="0 0 {W} {H}"
						class="block"
						role="img"
						aria-label="Two-dimensional PCA projection of the word vectors; nearby labels are words the model treats alike. Use the inspect menu to explore neighbors."
					>
						{#each placed as p, i (i)}
							{#if shown[i]}
								<text
									class="lbl"
									class:lbl-sel={selected === i}
									class:lbl-nb={selected !== i && neighborSet.has(i)}
									style="transform: translate({p.x}px, {p.y}px);"
									text-anchor="middle"
									onpointerenter={() => (selected = i)}
									onclick={() => (selected = i)}
									role="presentation">{vocab[i]}</text
								>
							{:else}
								<circle
									cx={p.x}
									cy={p.y}
									r="1.5"
									fill="var(--ink-3)"
									opacity="0.45"
									role="presentation"
								>
									<title>{vocab[i]}</title>
								</circle>
							{/if}
						{/each}
					</svg>
				</div>

				<!-- the rail: nearest neighbors by cosine -->
				<div class="bg-surface p-3.5">
					<label class="block">
						<span class="eyebrow mb-1 block">inspect a word</span>
						<select
							class="input"
							value={selected}
							onchange={(e) => (selected = Number(e.currentTarget.value))}
						>
							{#each alphaVocab as w (w)}
								<option value={vocab.indexOf(w)}>{w}</option>
							{/each}
						</select>
					</label>
					<div class="mt-3 flex flex-col gap-1.5">
						{#if selected !== null}
							<span class="eyebrow">nearest by cosine</span>
							{#each neighborRows as n (n.id)}
								<button class="nb-row" onclick={() => (selected = n.id)} title="inspect {n.word}">
									<span class="num w-16 truncate text-left text-[11.5px] text-ink">{n.word}</span>
									<span class="h-[3px] flex-1 overflow-hidden rounded-full bg-line-soft">
										<span
											class="block h-full"
											style="width: {(Math.max(0, n.sim) * 100).toFixed(
												1
											)}%; background: var(--accent);"
										></span>
									</span>
									<span class="num w-8 text-right text-[10px] text-ink-3">{n.sim.toFixed(2)}</span>
								</button>
							{/each}
						{/if}
					</div>
					<p class="mt-3 text-[10.5px] leading-snug text-ink-3">
						hover any label on the map, or pick from the menu — the list re-ranks live as training
						moves the vectors.
					</p>
				</div>
			</div>

			<!-- the analogy row: a − b + c ≈ ? -->
			<div
				class="flex flex-wrap items-center gap-x-2.5 gap-y-2 border-t border-line-soft px-4 py-3"
			>
				<span class="eyebrow mr-1">vector arithmetic</span>
				<select class="input input-sm" bind:value={selA} aria-label="a">
					{#each alphaVocab as w (w)}<option value={w}>{w}</option>{/each}
				</select>
				<span class="num text-[12px] text-ink-2">−</span>
				<select class="input input-sm" bind:value={selB} aria-label="minus b">
					{#each alphaVocab as w (w)}<option value={w}>{w}</option>{/each}
				</select>
				<span class="num text-[12px] text-ink-2">+</span>
				<select class="input input-sm" bind:value={selC} aria-label="plus c">
					{#each alphaVocab as w (w)}<option value={w}>{w}</option>{/each}
				</select>
				<span class="num text-[12px] text-ink-2">≈</span>
				{#if analogyRows}
					{#each analogyRows as r, i (r.word)}
						<span
							class="num rounded-md border px-2 py-0.5 text-[12px]"
							style="border-color: {i === 0
								? 'color-mix(in srgb, var(--accent) 55%, var(--line))'
								: 'var(--line-soft)'}; color: {i === 0 ? 'var(--accent)' : 'var(--ink-2)'};"
							>{r.word} <span class="text-[10px] text-ink-3">{r.sim.toFixed(2)}</span></span
						>
					{/each}
				{/if}
				<span class="ml-auto text-[10.5px] text-ink-3">
					a small corpus and 16 dimensions — the arithmetic is approximate; at web scale it becomes
					uncanny
				</span>
			</div>
		{/if}
	</div>
</Plate>

<style>
	.lbl {
		font-family: var(--font-serif);
		font-size: 11px;
		fill: var(--ink);
		cursor: pointer;
		transition: transform 260ms linear;
	}
	.lbl-sel {
		fill: var(--accent);
		font-size: 12.5px;
		font-weight: 600;
	}
	.lbl-nb {
		fill: color-mix(in srgb, var(--accent) 70%, var(--ink));
	}
	.nb-row {
		display: flex;
		align-items: center;
		gap: 6px;
		width: 100%;
		padding: 1px 2px;
		border-radius: 4px;
	}
	.nb-row:hover {
		background: var(--surface-2);
	}
	.input {
		width: 100%;
		height: 30px;
		font-family: var(--font-mono);
		font-size: 12.5px;
		color: var(--ink);
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: 6px;
		padding: 2px 8px;
	}
	.input:focus {
		outline: none;
		border-color: var(--accent);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 14%, transparent);
	}
	.input-sm {
		width: auto;
		min-width: 5.5rem;
	}
</style>

<script lang="ts">
	// Plate I — words become vectors. Skip-gram with negative sampling, trained
	// live on the story corpus, on the main thread (embeddings.ts). The stage
	// is a PCA shadow of the 16-D space that reorganizes as training runs —
	// flat, or a slowly turning 3-D cloud you can grab and spin; the rail
	// lists cosine neighbors; the analogy row does a − b + c for real.
	import { onDestroy } from 'svelte';
	import { browser } from '$app/environment';
	import { Pause, Play, RotateCcw } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { loadCorpus } from '$lib/data/corpus';
	import { buildWordCorpus, SkipGram, fitPca3, project, type WordCorpus } from './embeddings';

	type Phase = 'idle' | 'loading' | 'running' | 'paused' | 'done' | 'error';
	let phase = $state<Phase>('idle');
	let errorMsg = $state('');
	let tick = $state(0); // bumps once per training chunk
	let pairsSeen = $state(0);
	let lossNow = $state(NaN);
	let rate = $state(0); // pairs/s including display pacing
	let pts = $state<Float32Array>(new Float32Array(0)); // [3·V] projected coords
	let vocab = $state<string[]>([]);
	let selected = $state<number | null>(null);
	let query = $state('dog'); // the inspect box — typed, not picked
	let selA = $state('boy');
	let selB = $state('he');
	let selC = $state('she');

	// typed words: normalize, look up, and go red when the corpus has no such
	// word — the reader edits until the border relaxes
	const norm = (s: string) => s.trim().toLowerCase();
	const vocabIndex = $derived(new Map(vocab.map((w, i) => [w, i])));
	const isUnknown = (s: string) => norm(s) !== '' && vocab.length > 0 && !vocabIndex.has(norm(s));
	/** One entry point for every way of picking a word, so the box follows. */
	function pick(i: number) {
		selected = i;
		query = vocab[i] ?? query;
	}
	function onQuery() {
		const i = vocabIndex.get(norm(query));
		if (i !== undefined) selected = i;
	}

	let words: WordCorpus | null = null; // deliberately not $state
	let sg: SkipGram | null = null;
	let gen = 0;
	let lastFit = 0;
	let lastBeat = 0;

	// ≈ 60k pairs/s — structure appears in a few seconds
	const PACING = { chunk: 4000, pace: 66 };
	/** Everything trains at VOCAB words; the density slider decides how many
	 * of them (in frequency order) the stage draws. */
	const VOCAB = 600;

	// ── the view: a flat shadow, or a cloud you can spin ──
	let view = $state<'2d' | '3d'>('3d');
	let shownCount = $state(320);
	let dragging = $state(false);
	let hoverable = $state(false); // a label is under the cursor (2-d affordance)
	// camera state is plain — only the canvas reads it, every frame
	let yaw = 0.5;
	let pitch = 0.3;
	let zoom = 1;
	let dragLast: [number, number] | null = null;
	let downAt: [number, number] | null = null;
	const reduced = browser && matchMedia('(prefers-reduced-motion: reduce)').matches;
	const VIEWS = [
		{ id: '2d' as const, label: '2-d' },
		{ id: '3d' as const, label: '3-d' }
	];

	/** Label boxes of the last painted frame, in CSS px — the hit-test set. */
	let labelBoxes: Array<{ i: number; x0: number; y0: number; x1: number; y1: number }> = [];
	function hitLabel(ev: PointerEvent): number {
		if (!canvasEl) return -1;
		const r = canvasEl.getBoundingClientRect();
		const x = ev.clientX - r.left;
		const y = ev.clientY - r.top;
		for (let k = labelBoxes.length - 1; k >= 0; k--) {
			const b = labelBoxes[k];
			if (x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1) return b.i;
		}
		return -1;
	}

	function mapPtrDown(ev: PointerEvent) {
		downAt = [ev.clientX, ev.clientY];
		if (view !== '3d') return;
		(ev.currentTarget as HTMLCanvasElement).setPointerCapture(ev.pointerId);
		dragging = true;
		dragLast = [ev.clientX, ev.clientY];
	}
	function mapPtrMove(ev: PointerEvent) {
		if (dragging && dragLast) {
			yaw += (ev.clientX - dragLast[0]) * 0.008;
			pitch = Math.max(-1.3, Math.min(1.3, pitch + (ev.clientY - dragLast[1]) * 0.006));
			dragLast = [ev.clientX, ev.clientY];
			return;
		}
		// hover-select only on the still map; on the turning cloud labels sweep
		// under a resting cursor and would churn the selection, so there it
		// takes a click (handled in ptrUp)
		const i = hitLabel(ev);
		hoverable = i !== -1;
		if (view === '2d' && i !== -1) pick(i);
	}
	function mapPtrUp(ev: PointerEvent) {
		const wasDrag =
			downAt && Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 5 && view === '3d';
		dragging = false;
		dragLast = null;
		downAt = null;
		if (wasDrag) return;
		const i = hitLabel(ev);
		if (i !== -1) pick(i);
	}
	function mapPtrLeave() {
		dragging = false;
		dragLast = null;
		hoverable = false;
	}
	/** Wheel-to-zoom on the cloud; an action because Svelte's wheel handlers
	 * are passive and could not preventDefault the page scroll. */
	function wheelZoom(node: HTMLCanvasElement) {
		const h = (ev: WheelEvent) => {
			if (view !== '3d') return;
			ev.preventDefault();
			zoom = Math.max(0.6, Math.min(3, zoom * Math.exp(-ev.deltaY * 0.0012)));
		};
		node.addEventListener('wheel', h, { passive: false });
		return { destroy: () => node.removeEventListener('wheel', h) };
	}

	async function boot() {
		if (phase !== 'idle' && phase !== 'error') return;
		phase = 'loading';
		errorMsg = '';
		const myGen = ++gen;
		try {
			const corpus = await loadCorpus(); // shared cache with every other plate
			if (myGen !== gen) return;
			const text = Array.from(corpus.tokens, (t) => corpus.vocab[t]).join('');
			words = buildWordCorpus(text, VOCAB);
			// a larger vocabulary needs a little more schedule for its rare tail
			sg = new SkipGram(words, { dim: 16, seed: 7, budget: 2_000_000 });
			vocab = words.vocab;
			pick(Math.max(0, words.vocab.indexOf('dog')));
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
	/** Fresh vectors, same corpus and seed — watch the space organize again.
	 * The stage's easing does the rest: the map collapses back into the random
	 * cloud and re-forms instead of teleporting. */
	function resetTraining() {
		if (!words) return;
		gen++;
		sg = new SkipGram(words, { dim: 16, seed: 7, budget: 2_000_000 });
		pairsSeen = 0;
		lossNow = NaN;
		rate = 0;
		tick++;
		refit(true);
		phase = 'running';
		void run();
	}

	onDestroy(() => {
		gen++;
	});

	/** PCA refit, throttled — 600×16 power iteration costs a few ms. */
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
		const pca = fitPca3(vecs);
		const out = new Float32Array(3 * V);
		for (let i = 0; i < V; i++) {
			const [x, y, z] = project(vecs[i], pca);
			out[3 * i] = x;
			out[3 * i + 1] = y;
			out[3 * i + 2] = z;
		}
		pts = out;
	}

	// ── the stage: a canvas painter ──
	// SVG choked here — six hundred words × (dot + label) × 60 fps of attribute
	// diffing dropped to ~18 fps. The canvas draws the same scene in ~1 ms and
	// makes the anti-flicker tricks cheap: label alphas lerp for a cross-fade,
	// incumbents keep their room in the thinning, points ease between refits.
	const H = 420;
	let canvasEl = $state<HTMLCanvasElement | null>(null);
	/** How many words (in frequency order) the stage draws; ids ARE freq ranks. */
	const visCount = $derived(Math.min(shownCount, vocab.length));

	// paint-loop state, all plain: nothing here should wake Svelte
	let smooth: Float32Array | null = null; // eased copy of pts
	let alphas: Float32Array | null = null; // per-label fade
	let prevShown: Uint8Array | null = null; // thinning hysteresis
	let px: Float32Array = new Float32Array(0);
	let py: Float32Array = new Float32Array(0);
	let po: Float32Array = new Float32Array(0);

	$effect(() => {
		const cv = canvasEl;
		if (!cv) return;
		const ctx = cv.getContext('2d');
		if (!ctx) return;
		let raf = 0;
		let last = performance.now();
		const step = (now: number) => {
			raf = requestAnimationFrame(step);
			const dt = Math.min(50, now - last);
			last = now;
			// the cloud turns slowly on its own; a drag takes over, reduced motion opts out
			if (view === '3d' && !reduced && !dragging) yaw += dt * 0.00012;
			paint(ctx, cv, dt);
		};
		raf = requestAnimationFrame(step);
		return () => cancelAnimationFrame(raf);
	});

	function paint(ctx: CanvasRenderingContext2D, cv: HTMLCanvasElement, dt: number): void {
		const V = vocab.length;
		const dpr = Math.min(2, (browser && devicePixelRatio) || 1);
		const W = cv.clientWidth || 640;
		if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
			cv.width = Math.round(W * dpr);
			cv.height = Math.round(H * dpr);
		}
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, W, H);
		if (V === 0 || pts.length !== 3 * V) return;

		// ease toward the latest PCA refit, so the map drifts instead of jumping
		if (!smooth || smooth.length !== pts.length) smooth = Float32Array.from(pts);
		const kf = Math.min(1, dt / 220);
		for (let j = 0; j < pts.length; j++) smooth[j] += (pts[j] - smooth[j]) * kf;
		if (!alphas || alphas.length !== V) {
			alphas = new Float32Array(V);
			prevShown = new Uint8Array(V);
		}
		if (px.length !== V) {
			px = new Float32Array(V);
			py = new Float32Array(V);
			po = new Float32Array(V);
		}

		const vis = visCount;
		const sel = selected;
		const isVis = (i: number) => i < vis || i === sel;
		const pad = 36;

		// ── place every visible word ──
		if (view === '2d') {
			let minX = Infinity;
			let maxX = -Infinity;
			let minY = Infinity;
			let maxY = -Infinity;
			for (let i = 0; i < V; i++) {
				if (!isVis(i)) continue;
				minX = Math.min(minX, smooth[3 * i]);
				maxX = Math.max(maxX, smooth[3 * i]);
				minY = Math.min(minY, smooth[3 * i + 1]);
				maxY = Math.max(maxY, smooth[3 * i + 1]);
			}
			// uniform scale: the map keeps its geometry, only fits the frame
			const s = Math.min(
				(W - 2 * pad) / Math.max(1e-6, maxX - minX),
				(H - 2 * pad) / Math.max(1e-6, maxY - minY)
			);
			const cx = (minX + maxX) / 2;
			const cy = (minY + maxY) / 2;
			for (let i = 0; i < V; i++) {
				px[i] = W / 2 + (smooth[3 * i] - cx) * s;
				py[i] = H / 2 - (smooth[3 * i + 1] - cy) * s;
				po[i] = 1;
			}
		} else {
			// 3-d: centre on the visible cloud, fit by a high quantile of its
			// radius — rotation-invariant (no breathing) and outlier-proof
			let cx = 0;
			let cy = 0;
			let cz = 0;
			let m = 0;
			for (let i = 0; i < V; i++) {
				if (!isVis(i)) continue;
				cx += smooth[3 * i];
				cy += smooth[3 * i + 1];
				cz += smooth[3 * i + 2];
				m++;
			}
			cx /= Math.max(1, m);
			cy /= Math.max(1, m);
			cz /= Math.max(1, m);
			const radii: number[] = [];
			for (let i = 0; i < V; i++) {
				if (!isVis(i)) continue;
				radii.push(Math.hypot(smooth[3 * i] - cx, smooth[3 * i + 1] - cy, smooth[3 * i + 2] - cz));
			}
			radii.sort((a, b) => a - b);
			const maxR = Math.max(1e-6, radii[Math.floor(radii.length * 0.92)] ?? 1e-6);
			const s = ((Math.min(W, H) / 2 - pad) / maxR) * zoom;
			const cyaw = Math.cos(yaw);
			const syaw = Math.sin(yaw);
			const cp = Math.cos(pitch);
			const sp = Math.sin(pitch);
			for (let i = 0; i < V; i++) {
				const dx = smooth[3 * i] - cx;
				const dy = smooth[3 * i + 1] - cy;
				const dz = smooth[3 * i + 2] - cz;
				const x1 = cyaw * dx + syaw * dz;
				const z1 = -syaw * dx + cyaw * dz;
				const y2 = cp * dy - sp * z1;
				const z2 = sp * dy + cp * z1;
				px[i] = W / 2 + x1 * s;
				py[i] = H / 2 - y2 * s;
				// depth as opacity — the only cue an orthographic cloud needs
				po[i] = Math.max(0.18, Math.min(1, 0.35 + 0.65 * ((z2 / maxR + 1) / 2)));
			}
		}

		// ── greedy collision thinning in frequency order; the selected word
		// always shows. Incumbents get first claim: a label already on screen
		// keeps its room and newcomers only fill what is left — without this the
		// turning cloud re-elects its labels every frame and they flash ──
		const shownNow = new Uint8Array(V);
		const boxes: number[] = []; // x0,x1,y0,y1 quads, flat
		const tryPlace = (i: number): boolean => {
			const bw = vocab[i].length * 6.4 + 8;
			const x0 = px[i] - bw / 2;
			const x1 = px[i] + bw / 2;
			const y0 = py[i] - 7;
			const y1 = py[i] + 7;
			for (let b = 0; b < boxes.length; b += 4)
				if (x0 < boxes[b + 1] && x1 > boxes[b] && y0 < boxes[b + 3] && y1 > boxes[b + 2])
					return false;
			boxes.push(x0, x1, y0, y1);
			return true;
		};
		if (sel !== null && sel < V) {
			shownNow[sel] = 1;
			tryPlace(sel);
		}
		for (let pass = 0; pass < 2; pass++) {
			for (let i = 0; i < V; i++) {
				if (shownNow[i] || !isVis(i)) continue;
				if (pass === 0 ? !prevShown![i] : prevShown![i]) continue;
				if (tryPlace(i)) shownNow[i] = 1;
			}
		}
		prevShown = shownNow;

		// labels cross-fade instead of popping
		const ka = Math.min(1, dt / 260);
		for (let i = 0; i < V; i++) {
			const target = isVis(i) ? shownNow[i] : 0;
			alphas[i] += (target - alphas[i]) * ka;
			if (alphas[i] < 0.015) alphas[i] = target ? alphas[i] : 0;
		}

		// ── draw ──
		const cs = getComputedStyle(cv);
		const ink = cs.getPropertyValue('--ink').trim() || '#222';
		const ink3 = cs.getPropertyValue('--ink-3').trim() || '#999';
		const accent = cs.getPropertyValue('--accent').trim() || '#65d';
		const serif = cs.getPropertyValue('--font-serif').trim() || 'serif';
		const nbSet = neighborSet;

		// dots — every visible word has one; it recedes while its label is up
		ctx.fillStyle = ink3;
		for (let i = 0; i < V; i++) {
			if (!isVis(i)) continue;
			const a = 0.45 * po[i] * (1 - 0.85 * alphas[i]);
			if (a < 0.02) continue;
			ctx.globalAlpha = a;
			ctx.fillRect(px[i] - 1.2, py[i] - 1.2, 2.4, 2.4);
		}

		// labels, back to front so near words overpaint far ones
		const order: number[] = [];
		for (let i = 0; i < V; i++) if (alphas[i] > 0.02 && isVis(i)) order.push(i);
		order.sort((a, b) => po[a] - po[b]);
		ctx.textAlign = 'center';
		labelBoxes = [];
		for (const i of order) {
			const isSel = i === sel;
			ctx.globalAlpha = isSel ? 1 : alphas[i] * po[i];
			ctx.font = isSel ? `600 12.5px ${serif}` : `11px ${serif}`;
			ctx.fillStyle = isSel || nbSet.has(i) ? accent : ink;
			ctx.fillText(vocab[i], px[i], py[i] + 3.5);
			if (alphas[i] > 0.5 || isSel) {
				const bw = vocab[i].length * 6.4 + 8;
				labelBoxes.push({
					i,
					x0: px[i] - bw / 2,
					x1: px[i] + bw / 2,
					y0: py[i] - 7,
					y1: py[i] + 7
				});
			}
		}
		ctx.globalAlpha = 1;
	}

	// ── readouts ──
	const neighborRows = $derived.by(() => {
		void tick;
		if (!sg || selected === null) return [];
		return sg.neighbors(selected, 8).map((n) => ({ id: n.id, word: vocab[n.id], sim: n.sim }));
	});
	const neighborSet = $derived(new Set(neighborRows.map((n) => n.id)));
	// the selected word's actual vector — the 16 numbers the 2-D map is a shadow
	// of. Steals the explorer's "a token is a column of numbers" idea to make
	// concrete what "words become vectors" means, and to fill the rail.
	const selVec = $derived.by(() => {
		void tick;
		if (!sg || selected === null) return null;
		return Array.from(sg.vector(selected));
	});
	const selVecMax = $derived(selVec ? Math.max(1e-6, ...selVec.map((v) => Math.abs(v))) : 1);
	const analogyRows = $derived.by(() => {
		void tick;
		if (!sg) return null;
		const ia = vocabIndex.get(norm(selA));
		const ib = vocabIndex.get(norm(selB));
		const ic = vocabIndex.get(norm(selC));
		if (ia === undefined || ib === undefined || ic === undefined) return null;
		return sg.analogy(ia, ib, ic, 3).map((r) => ({ word: vocab[r.id], sim: r.sim }));
	});

	const fmtK = (n: number) => (n >= 1000 ? `${Math.round(n / 1000)}k` : String(Math.round(n)));

	/** Analogies that actually land on this corpus (probed offline at the same
	 * seed and budget). No king−man+woman here: children's stories mention
	 * "king" nine times in 292k words, too rare to earn a usable vector. */
	const PRESETS: Array<[string, string, string]> = [
		['boy', 'he', 'she'],
		['dad', 'he', 'she'],
		['mommy', 'mom', 'dad']
	];
</script>

<Plate
	n={1}
	title="Words become vectors"
	caption="A PCA shadow of a 16-dimensional space — the top 600 words of this book's story corpus, redrawn as skip-gram training runs on your CPU. Words used alike drift together: animals with animals, names with names, he beside she. The cloud turns on its own; drag to steer it, or flatten it to the 2-D shadow, and use the slider to decide how much of the vocabulary the stage draws. Hover any label or pick from the menu to re-rank the neighbour list, which is computed in the full space by cosine similarity."
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
				<span>settled</span>
			{/if}
		{/if}
	{/snippet}

	{#snippet actions()}
		{#if phase === 'running'}
			<Btn onclick={pauseTraining}>
				<Pause size={12} aria-hidden="true" /> Pause
			</Btn>
		{:else if phase === 'paused'}
			<Btn onclick={resumeTraining}>
				<Play size={12} aria-hidden="true" /> Resume
			</Btn>
		{/if}
		{#if phase === 'running' || phase === 'paused' || phase === 'done'}
			<Btn onclick={resetTraining}>
				<RotateCcw size={12} aria-hidden="true" /> Reset
			</Btn>
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
			<div class="grid grid-cols-1 gap-px bg-line-soft md:grid-cols-[minmax(0,1fr)_15rem]">
				<!-- the map -->
				<div class="relative bg-surface">
					<div class="absolute top-2.5 right-3 z-10 flex items-center gap-1.5">
						{#if view === '3d'}
							<span class="num mr-1 text-[10px] text-ink-3">drag to turn · wheel to zoom</span>
						{/if}
						{#each VIEWS as v (v.id)}
							<button
								class="chip"
								class:chip-on={view === v.id}
								aria-pressed={view === v.id}
								onclick={() => (view = v.id)}>{v.label}</button
							>
						{/each}
					</div>
					<div class="absolute bottom-2.5 left-3 z-10 flex items-center gap-2">
						<input
							class="dens"
							type="range"
							min="120"
							max={vocab.length || VOCAB}
							step="20"
							bind:value={shownCount}
							aria-label="how many words the map shows"
						/>
						<span class="num text-[10px] text-ink-3">{visCount} words</span>
					</div>
					<canvas
						bind:this={canvasEl}
						class="block w-full touch-none"
						style="height: {H}px;"
						class:cursor-grab={view === '3d' && !dragging && !hoverable}
						class:cursor-grabbing={dragging}
						class:cursor-pointer={!dragging && hoverable}
						aria-label="PCA projection of the word vectors, flat or as a rotatable three-dimensional cloud; nearby labels are words the model treats alike. Use the inspect menu to explore neighbors."
						use:wheelZoom
						onpointerdown={mapPtrDown}
						onpointermove={mapPtrMove}
						onpointerup={mapPtrUp}
						onpointercancel={mapPtrLeave}
						onpointerleave={mapPtrLeave}
					></canvas>
				</div>

				<!-- the rail: nearest neighbors by cosine -->
				<div class="flex h-full flex-col bg-surface p-3.5">
					<label class="block">
						<span class="eyebrow mb-1 block">inspect a word</span>
						<input
							class="input"
							class:input-bad={isUnknown(query)}
							type="text"
							spellcheck="false"
							autocomplete="off"
							placeholder="type a word…"
							bind:value={query}
							oninput={onQuery}
						/>
					</label>
					<div class="mt-3 flex flex-col gap-1.5">
						{#if selected !== null}
							<span class="eyebrow">nearest by cosine</span>
							{#each neighborRows as n (n.id)}
								<button class="nb-row" onclick={() => pick(n.id)} title="inspect {n.word}">
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

					<!-- the word as its actual 16 numbers; the map above is a flat shadow of this -->
					{#if selVec && selected !== null}
						<div class="mt-auto border-t border-line-soft pt-3">
							<span class="eyebrow mb-1.5 block">
								<span style="color: var(--accent);">{vocab[selected]}</span> · its 16 numbers
							</span>
							<svg
								width="100%"
								height="66"
								viewBox="0 0 208 66"
								preserveAspectRatio="none"
								class="block"
								role="img"
								aria-label="The selected word's raw 16-dimensional vector as signed bars around a zero line; this is the space the two-dimensional map projects down from."
							>
								<line x1="0" y1="33" x2="208" y2="33" stroke="var(--line)" stroke-width="1" />
								{#each selVec as v, i (i)}
									{@const h = (Math.abs(v) / selVecMax) * 28}
									<rect
										x={i * 13 + 3}
										y={v >= 0 ? 33 - h : 33}
										width="7"
										height={Math.max(1, h)}
										rx="1"
										fill={v >= 0 ? 'var(--warm)' : 'var(--accent)'}
										opacity="0.85"
									/>
								{/each}
							</svg>
							<span class="num mt-1 block text-[10px] text-ink-3">
								{view === '3d'
									? 'the turning cloud is a shadow of these'
									: 'the flat map is a shadow of these'}
							</span>
						</div>
					{/if}
				</div>
			</div>

			<!-- the analogy row: a − b + c ≈ ? -->
			<div class="border-t border-line-soft px-4 py-3">
				<div class="flex flex-wrap items-center gap-x-2.5 gap-y-2">
					<span class="eyebrow mr-1">vector arithmetic</span>
					<input
						class="input input-sm"
						class:input-bad={isUnknown(selA)}
						type="text"
						spellcheck="false"
						autocomplete="off"
						bind:value={selA}
						aria-label="a"
					/>
					<span class="num text-[12px] text-ink-2">−</span>
					<input
						class="input input-sm"
						class:input-bad={isUnknown(selB)}
						type="text"
						spellcheck="false"
						autocomplete="off"
						bind:value={selB}
						aria-label="minus b"
					/>
					<span class="num text-[12px] text-ink-2">+</span>
					<input
						class="input input-sm"
						class:input-bad={isUnknown(selC)}
						type="text"
						spellcheck="false"
						autocomplete="off"
						bind:value={selC}
						aria-label="plus c"
					/>
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
				</div>
				<!-- presets live on their own line: the result chips above change
				     width as training re-ranks them, and sharing a row made the
				     presets jump around -->
				<div class="mt-2.5 flex flex-wrap items-center gap-x-2 gap-y-2">
					<span class="num text-[10px] text-ink-3">try</span>
					{#each PRESETS as [a, b, c] (a + b + c)}
						<button
							class="chip"
							onclick={() => {
								selA = a;
								selB = b;
								selC = c;
							}}>{a}−{b}+{c}</button
						>
					{/each}
				</div>
			</div>
		{/if}
	</div>
</Plate>

<style>
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
		width: 6.5rem;
	}
	/* an unknown word goes red; the reader edits until the border relaxes */
	.input-bad {
		border-color: var(--bad);
		color: var(--bad);
	}
	.input-bad:focus {
		border-color: var(--bad);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--bad) 14%, transparent);
	}
	.chip {
		font-family: var(--font-sans);
		font-size: 10.5px;
		font-weight: 520;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--ink-2);
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: 999px;
		padding: 3px 10px;
		transition:
			color 100ms ease,
			border-color 100ms ease;
	}
	.chip:hover {
		border-color: var(--ink-3);
		color: var(--ink);
	}
	.chip-on {
		color: var(--accent);
		background: var(--accent-soft);
		border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
	}
	/* how many words to draw — a hairline control that sits on the map
	   without competing with it */
	.dens {
		appearance: none;
		width: 104px;
		height: 12px;
		background: transparent;
		cursor: pointer;
	}
	.dens::-webkit-slider-runnable-track {
		height: 2px;
		border-radius: 1px;
		background: var(--line);
	}
	.dens::-webkit-slider-thumb {
		appearance: none;
		width: 9px;
		height: 9px;
		margin-top: -3.5px;
		border-radius: 50%;
		background: var(--ink-3);
		transition: background 100ms ease;
	}
	.dens:hover::-webkit-slider-thumb {
		background: var(--ink-2);
	}
	.dens::-moz-range-track {
		height: 2px;
		border-radius: 1px;
		background: var(--line);
	}
	.dens::-moz-range-thumb {
		width: 9px;
		height: 9px;
		border: none;
		border-radius: 50%;
		background: var(--ink-3);
	}
</style>

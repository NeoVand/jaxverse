<script lang="ts">
	// Plate II — the classifier. Train and test ride in one buffer: 8,000 train
	// rows first, then the 2,000 test rows as the held-out tail (valFraction
	// 0.2), so the worker's eval() is honest test-set accuracy and the test
	// rows are never sampled by a training batch. Boots on scroll (inview);
	// the first button the reader sees is Train.
	import { onDestroy } from 'svelte';
	import { Pause, Play, RotateCcw, Shuffle } from 'lucide-svelte';
	import { MlpEngine } from '$lib/nn/mlp-engine';
	import { loadMnist } from '$lib/data/mnist';
	import { progress } from '$lib/data/progress.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { lab, themePulse, watchTheme } from './digits-context.svelte';
	import { DIM, argmax, blit, hexRgb, inkImage, readTokens, sparkPath } from './common';

	const STRIP = 12;
	const LAYERS = [784, 128, 64, 10];
	// the model card's layer-stack glyph: bar height ∝ √width, fits 32px
	const GLYPH = LAYERS.map((w, i) => {
		const h = Math.round(Math.sqrt(w) * 10.7) / 10;
		return { x: 1 + i * 12, y: (32 - h) / 2, h };
	});

	let training = $state(false);
	let errorMsg = $state('');
	let sampleIdx = $state<number[]>([]);
	let guesses = $state<number[]>([]);
	let lossHist = $state<number[]>([]);
	let accHist = $state<number[]>([]);
	let gen = 0; // boot generation — cancels stale loops

	watchTheme();

	const mnist = $derived(lab.phase === 'ready' ? lab.mnist : null);
	const accLabel = $derived(
		Number.isFinite(lab.testAcc) ? `${(lab.testAcc * 100).toFixed(1)}%` : '—'
	);

	function pickSamples(): number[] {
		const m = lab.mnist;
		if (!m) return [];
		const n = m.testY.length;
		const out: number[] = [];
		while (out.length < Math.min(STRIP, n)) {
			const t = Math.floor(Math.random() * n);
			if (!out.includes(t)) out.push(t);
		}
		return out;
	}

	async function boot() {
		if (lab.phase === 'loading' || lab.phase === 'ready') return;
		lab.phase = 'loading';
		errorMsg = '';
		const myGen = ++gen;
		let engine: MlpEngine | null = null;
		try {
			const m = await loadMnist();
			if (myGen !== gen) return;
			engine = new MlpEngine();
			const nTrain = m.trainY.length;
			const nTest = m.testY.length;
			const n = nTrain + nTest;
			const x = new Float32Array(n * DIM);
			x.set(m.trainX, 0);
			x.set(m.testX, nTrain * DIM);
			const y = new Int32Array(n);
			y.set(m.trainY, 0);
			y.set(m.testY, nTrain);
			await engine.init(
				{
					layers: LAYERS,
					activation: 'relu',
					loss: 'xent',
					lr: 3e-3,
					batchSize: 128,
					seed: 7,
					valFraction: nTest / n
				},
				{ x, y, n }
			);
			if (myGen !== gen) {
				void engine.dispose();
				return;
			}
			lab.engine = engine;
			lab.mnist = m;
			lab.device = engine.device;
			lab.paramCount = engine.paramCount;
			lab.step = 0;
			lab.phase = 'ready';
			sampleIdx = pickSamples();
			await refreshGuesses();
		} catch (err) {
			void engine?.dispose();
			if (myGen !== gen) return;
			lab.phase = 'error';
			errorMsg = err instanceof Error ? err.message : String(err);
		}
	}

	function setTraining(on: boolean) {
		if (on && !training && lab.phase === 'ready') {
			training = true;
			void trainLoop();
		} else if (!on && training) {
			training = false;
			void lab.engine?.stop(); // ends the in-flight chunk early
		}
	}

	async function trainLoop() {
		const myGen = gen;
		const engine = lab.engine;
		while (training && engine && myGen === gen) {
			const chunk = 25;
			const syncEvery = 4;
			try {
				await engine.train(
					chunk,
					(m) => {
						lab.step = m.step;
						lab.loss = m.loss;
						lab.msPerStep = lab.msPerStep ? lab.msPerStep * 0.7 + m.stepMs * 0.3 : m.stepMs;
						lossHist = [...lossHist.slice(-239), m.loss];
					},
					syncEvery
				);
				if (myGen !== gen || !lab.engine) return;
				const ev = await engine.eval();
				if (myGen !== gen) return;
				lab.testAcc = ev.accuracy ?? NaN;
				accHist = [...accHist.slice(-239), lab.testAcc];
				if (lab.testAcc >= 0.9) progress.reach('digits:trained');
				await refreshGuesses();
				lab.version += 1;
			} catch {
				return; // engine disposed mid-flight
			}
		}
	}

	async function refreshGuesses() {
		const engine = lab.engine;
		const m = lab.mnist;
		const ids = sampleIdx;
		if (!engine || !m || ids.length === 0) return;
		const xs = new Float32Array(ids.length * DIM);
		ids.forEach((t, i) => xs.set(m.testX.subarray(t * DIM, (t + 1) * DIM), i * DIM));
		try {
			const logits = await engine.predict(xs, ids.length, 16);
			if (ids !== sampleIdx || lab.engine !== engine) return;
			guesses = ids.map((_, i) => argmax(logits, i * 10, 10));
		} catch {
			/* engine disposed mid-flight */
		}
	}

	function reshuffle() {
		sampleIdx = pickSamples();
		guesses = [];
		void refreshGuesses();
	}

	async function resetWeights() {
		const engine = lab.engine;
		if (!engine) return;
		const wasTraining = training;
		training = false;
		try {
			await engine.stop(); // wind down the in-flight chunk first
			await engine.reset(Math.floor(Math.random() * 1e9));
		} catch {
			return;
		}
		lab.step = 0;
		lab.loss = NaN;
		lab.testAcc = NaN;
		lab.msPerStep = 0;
		lossHist = [];
		accHist = [];
		await refreshGuesses(); // never throws — it guards internally
		lab.version += 1;
		if (wasTraining) setTraining(true);
	}

	function drawTile(t: number) {
		return (el: HTMLCanvasElement) => {
			void themePulse.tick; // redraw when the theme flips
			const m = lab.mnist;
			if (!m) return;
			blit(el, [inkImage(m.testX, t * DIM, hexRgb(readTokens(el).ink))], 84);
		};
	}

	onDestroy(() => {
		gen += 1;
		training = false;
		void lab.engine?.dispose();
		lab.clear();
	});
</script>

<div class="flex flex-col" use:inview={boot}>
	<!-- controls -->
	<div class="flex flex-wrap items-center gap-x-4 gap-y-3 border-b border-line-soft px-4 py-3">
		{#if lab.phase === 'error'}
			<span class="text-[12.5px] text-bad">{errorMsg}</span>
			<Btn onclick={boot}>Retry</Btn>
		{:else if lab.phase !== 'ready'}
			<span class="eyebrow">fetching MNIST (≈1.7 MB, cached) · warming up the worker…</span>
			<span class="text-[12.5px] text-ink-3">trains on your device — nothing leaves this page</span>
		{:else}
			<Btn kind="primary" onclick={() => setTraining(!training)}>
				{#if training}
					<Pause size={12} aria-hidden="true" /> Pause
				{:else}
					<Play size={12} aria-hidden="true" /> Train
				{/if}
			</Btn>
			<Btn onclick={resetWeights} title="Fresh random weights">
				<RotateCcw size={12} aria-hidden="true" /> Reset
			</Btn>
		{/if}
	</div>

	{#if lab.phase === 'ready' && mnist}
		<div class="grid grid-cols-1 gap-x-8 gap-y-6 p-4 sm:p-5 md:grid-cols-[1fr_224px]">
			<!-- stage left: twelve held-out digits and the model's reading of each -->
			<div>
				<div class="mb-3 flex flex-wrap items-center justify-between gap-2">
					<span class="eyebrow">twelve test digits · its guess under each</span>
					<Btn onclick={reshuffle}>
						<Shuffle size={12} aria-hidden="true" /> Reshuffle
					</Btn>
				</div>
				<div class="grid grid-cols-4 gap-x-3 gap-y-4 sm:grid-cols-6">
					{#each sampleIdx as t, i (t)}
						{@const truth = mnist.testY[t]}
						{@const guess = guesses[i]}
						{@const wrong = guess !== undefined && guess !== truth}
						<figure class="m-0 flex flex-col items-center gap-1">
							<canvas
								class="h-21 w-21 rounded-[3px] border border-line-soft"
								aria-label="a handwritten {truth}"
								{@attach drawTile(t)}
							></canvas>
							<figcaption
								class="num text-[13px]"
								style="color: {guess === undefined
									? 'var(--ink-3)'
									: wrong
										? 'var(--bad)'
										: 'var(--ink-2)'};"
							>
								{guess === undefined ? '·' : guess}<span class="sr-only">
									— {wrong ? `wrong, it is a ${truth}` : 'correct'}</span
								>
							</figcaption>
						</figure>
					{/each}
				</div>
			</div>

			<!-- stage right: the model and its verdict -->
			<div class="flex flex-col gap-5">
				<div>
					<span class="eyebrow">the model</span>
					<div class="mt-1.5 flex items-center gap-3">
						<svg width="46" height="32" viewBox="0 0 46 32" aria-hidden="true" class="shrink-0">
							{#each GLYPH as g (g.x)}
								<rect
									x={g.x}
									y={g.y}
									width="6"
									height={g.h}
									rx="1"
									fill="none"
									stroke="var(--ink-3)"
									stroke-width="1"
								/>
							{/each}
						</svg>
						<div class="num text-[11px] leading-[1.6] text-ink-2">
							784 → 128 → 64 → 10<br />
							relu · softmax<br />
							{lab.paramCount.toLocaleString('en-US')} parameters · {lab.device}
						</div>
					</div>
				</div>
				<div>
					<span class="eyebrow">test accuracy</span>
					<div
						class="num text-[2.6rem] leading-[1.15]"
						style="color: {Number.isFinite(lab.testAcc) && lab.testAcc >= 0.9
							? 'var(--good)'
							: 'var(--ink)'};"
					>
						{accLabel}
					</div>
					<span class="text-[11.5px] text-ink-3">on digits it never trained on</span>
				</div>
				<div>
					<span class="eyebrow" style="color: var(--accent);">loss · train</span>
					<svg
						viewBox="0 0 200 34"
						preserveAspectRatio="none"
						class="mt-1 block h-[34px] w-full"
						role="img"
						aria-label="training loss over steps"
					>
						<path
							d={sparkPath(lossHist, 200, 34, true)}
							fill="none"
							stroke="var(--accent)"
							stroke-width="1.4"
							vector-effect="non-scaling-stroke"
						/>
					</svg>
				</div>
				<div>
					<span class="eyebrow" style="color: var(--warm);">accuracy · test</span>
					<svg
						viewBox="0 0 200 34"
						preserveAspectRatio="none"
						class="mt-1 block h-[34px] w-full"
						role="img"
						aria-label="test accuracy over steps"
					>
						<path
							d={sparkPath(accHist, 200, 34, false)}
							fill="none"
							stroke="var(--warm)"
							stroke-width="1.4"
							vector-effect="non-scaling-stroke"
						/>
					</svg>
				</div>
				{#if lab.step === 0}
					<p class="m-0 font-serif text-[13px] text-ink-3 italic">
						untrained — its guesses are chance. Press Train.
					</p>
				{/if}
			</div>
		</div>
	{:else}
		<div class="h-40" aria-hidden="true"></div>
	{/if}
</div>

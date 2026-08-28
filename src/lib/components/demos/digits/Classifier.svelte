<script lang="ts">
	// The classifier. Train and test ride in one buffer: 8,000 train
	// rows first, then the 2,000 test rows as the held-out tail (valFraction
	// 0.2), so the worker's eval() is honest test-set accuracy and the test
	// rows are never sampled by a training batch. Boots on scroll (inview);
	// the first button the reader sees is Train.
	import { onDestroy } from 'svelte';
	import {
		ArrowDownWideNarrow,
		ArrowUpNarrowWide,
		Pause,
		Play,
		RotateCcw,
		Shuffle
	} from 'lucide-svelte';
	import { MlpEngine } from '$lib/nn/mlp-engine';
	import type { Activation } from '$lib/nn/engine';
	import { loadMnist, type MnistData } from '$lib/data/mnist';
	import { progress } from '$lib/data/progress.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { lab, themePulse, watchTheme } from './digits-context.svelte';
	import { DIM, argmax, blit, hexRgb, inkImage, readTokens, softmax } from './common';
	import { sparkPath } from '$lib/viz/spark';

	interface Props {
		title: string;
		caption?: string;
	}
	let { title, caption }: Props = $props();

	const STRIP = 16; // two rows of eight, each with room for its own belief
	const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
	const DEPTHS = [1, 2, 3];
	const WIDTHS = [32, 64, 128, 256];
	const ACTS: Activation[] = ['tanh', 'relu', 'gelu', 'silu'];
	const PROBE = 1500; // training rows sampled for the train-accuracy reading

	let training = $state(false);
	let errorMsg = $state('');
	let pickMode = $state<'random' | 'easiest' | 'hardest'>('random');
	let sampleIdx = $state<number[]>([]);
	let guesses = $state<number[]>([]);
	/** Full softmax over the ten classes for each shown digit. */
	let dists = $state<number[][]>([]);
	let lossHist = $state<number[]>([]);
	let accHist = $state<number[]>([]);
	let gen = 0; // boot generation — cancels stale loops
	/** Identifies the current run, so a pause-then-play mid-run cannot leave
	 * two loops driving one worker. */
	let runToken = 0;
	/** Steps per worker run; it yields between sync points, so `stop` still
	 * lands promptly and measurement RPCs interleave. */
	const RUN_STEPS = 2000;
	/** Steps between sync points — each costs a stall to read the loss back
	 * and buys one sample of the curve. */
	const SYNC_EVERY = 10;
	/** How often the readings are taken. Accuracy is a percentage; five times
	 * a second is already more than a reader can follow. */
	const REFRESH_MS = 200;
	/** …and how often the expensive one is. */
	const PROBE_MS = 1500;
	/** Metrics land here first — see onMetrics. */
	let lossRing: number[] = [];
	let accRing: number[] = [];
	let liveStep = 0;
	let liveLoss = NaN;
	let liveMs = 0;
	let lossSum = 0;
	let lossCount = 0;

	watchTheme();

	const mnist = $derived(lab.phase === 'ready' ? lab.mnist : null);
	const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—');
	const gapLabel = $derived(
		Number.isFinite(lab.trainAcc) && Number.isFinite(lab.testAcc)
			? `${((lab.trainAcc - lab.testAcc) * 100).toFixed(1)} pts`
			: '—'
	);
	const layers = $derived([DIM, ...Array(lab.depth).fill(lab.width), 10]);
	const archLabel = $derived(layers.join(' → '));

	function pickSamples(): number[] {
		const m = lab.mnist;
		if (!m) return [];
		const total = m.testY.length;
		const out: number[] = [];
		while (out.length < Math.min(STRIP, total)) {
			const t = Math.floor(Math.random() * total);
			if (!out.includes(t)) out.push(t);
		}
		return out;
	}

	/** Rank the whole test set by the probability it gives the true answer, and
	 * show either end of it — the digits it is sure of, or the ones it fails. */
	async function rankSamples(mode: 'easiest' | 'hardest') {
		const engine = lab.engine;
		const m = lab.mnist;
		if (!engine || !m) return;
		const total = m.testY.length;
		try {
			const z = await engine.predict(m.testX, total, 512);
			if (lab.engine !== engine) return;
			const truthP = new Float64Array(total);
			for (let i = 0; i < total; i++) truthP[i] = softmax(z, i * 10, 10)[m.testY[i]];
			const order = Array.from({ length: total }, (_, i) => i).sort((a, b) =>
				mode === 'hardest' ? truthP[a] - truthP[b] : truthP[b] - truthP[a]
			);
			pickMode = mode;
			sampleIdx = order.slice(0, STRIP);
			guesses = [];
			dists = [];
			await refreshGuesses();
		} catch {
			/* engine disposed mid-flight */
		}
	}

	/** A fixed slice of the training rows, so the train/test gap is comparable
	 * from step to step and across architectures. */
	let probe: { x: Float32Array; y: Int32Array; n: number } | null = null;
	let testProbe: { x: Float32Array; y: Int32Array; n: number } | null = null;
	/** An evenly-strided sample of `xs`, so the reading covers the whole set
	 * rather than whichever rows happen to sit at the front of it. */
	function buildProbe(xs: Float32Array, ys: Int32Array, want: number) {
		const total = ys.length;
		const k = Math.min(want, total);
		const stride = Math.max(1, Math.floor(total / k));
		const x = new Float32Array(k * DIM);
		const y = new Int32Array(k);
		for (let i = 0; i < k; i++) {
			const t = i * stride;
			x.set(xs.subarray(t * DIM, (t + 1) * DIM), i * DIM);
			y[i] = ys[t];
		}
		return { x, y, n: k };
	}

	/** Accuracy of the current weights on a fixed set of rows. */
	async function accuracyOn(p: { x: Float32Array; y: Int32Array; n: number }): Promise<number> {
		const engine = lab.engine;
		if (!engine) return NaN;
		const logits = await engine.predict(p.x, p.n, 1024);
		if (lab.engine !== engine) return NaN;
		let hit = 0;
		for (let i = 0; i < p.n; i++) if (argmax(logits, i * 10, 10) === p.y[i]) hit += 1;
		return hit / p.n;
	}

	/** Train rows first, test rows as the held-out tail — built once per dataset. */
	let rows: { x: Float32Array; y: Int32Array; n: number; valFraction: number } | null = null;
	function buildRows(m: MnistData) {
		const nTrain = m.trainY.length;
		const nTest = m.testY.length;
		const total = nTrain + nTest;
		const x = new Float32Array(total * DIM);
		x.set(m.trainX, 0);
		x.set(m.testX, nTrain * DIM);
		const y = new Int32Array(total);
		y.set(m.trainY, 0);
		y.set(m.testY, nTrain);
		return { x, y, n: total, valFraction: nTest / total };
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
			rows = buildRows(m);
			probe = buildProbe(m.trainX, m.trainY, PROBE);
			testProbe = buildProbe(m.testX, m.testY, PROBE);
			engine = new MlpEngine();
			await engine.init(
				{
					layers,
					activation: lab.activation,
					loss: 'xent',
					lr: 3e-3,
					batchSize: 128,
					seed: 7,
					valFraction: rows.valFraction
				},
				{ x: rows.x, y: rows.y, n: rows.n }
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

	/** A new architecture is a new engine — the data is already decoded. */
	async function rebuild() {
		const m = lab.mnist;
		if (lab.phase !== 'ready' || !m || !rows) return;
		const wasTraining = training;
		training = false;
		const old = lab.engine;
		const myGen = ++gen;
		let engine: MlpEngine | null = null;
		try {
			await old?.stop();
			engine = new MlpEngine();
			await engine.init(
				{
					layers,
					activation: lab.activation,
					loss: 'xent',
					lr: 3e-3,
					batchSize: 128,
					seed: 7,
					valFraction: rows.valFraction
				},
				{ x: rows.x, y: rows.y, n: rows.n }
			);
			if (myGen !== gen) {
				void engine.dispose();
				return;
			}
			void old?.dispose();
			lab.engine = engine;
			lab.device = engine.device;
			lab.paramCount = engine.paramCount;
			lab.step = 0;
			lab.loss = NaN;
			lab.testAcc = NaN;
			lab.trainAcc = NaN;
			lab.msPerStep = 0;
			lossHist = [];
			accHist = [];
			lossRing = [];
			accRing = [];
			liveStep = 0;
			liveLoss = NaN;
			liveMs = 0;
			lossSum = 0;
			lossCount = 0;
			await refreshGuesses();
			lab.version += 1;
			if (wasTraining) setTraining(true);
		} catch (err) {
			void engine?.dispose();
			if (myGen !== gen) return;
			lab.phase = 'error';
			errorMsg = err instanceof Error ? err.message : String(err);
		}
	}

	function setDepth(d: number) {
		if (lab.depth === d) return;
		lab.depth = d;
		void rebuild();
	}
	function setWidth(w: number) {
		if (lab.width === w) return;
		lab.width = w;
		void rebuild();
	}
	function setActivation(a: Activation) {
		if (lab.activation === a) return;
		lab.activation = a;
		void rebuild();
	}

	function setTraining(on: boolean) {
		if (on && !training && lab.phase === 'ready') {
			training = true;
			const token = ++runToken;
			void trainLoop(token);
			void measureLoop(token);
		} else if (!on && training) {
			training = false;
			runToken++;
			void lab.engine?.stop(); // ends the in-flight run early
			publish(); // whatever the last steps measured, shown rather than dropped
		}
	}

	/**
	 * Train, and measure on a different clock.
	 *
	 * This was a ping-pong — twenty-five steps, then a test-set eval, then a
	 * 1,500-row probe of the training set, then the sixteen shown digits, then
	 * repeat — and every one of those was awaited in turn, so the worker
	 * stalled four times per chunk and the whole cycle ran as fast as training
	 * allowed. On this network that is a couple of thousand extra forward
	 * passes for every three thousand rows actually trained on, spent to
	 * refresh percentages that move too slowly for anyone to read at that
	 * rate, and it made the entire page feel heavy while the plate ran.
	 *
	 * The worker now gets a long run and is left alone; it yields between sync
	 * points, so the measurement RPCs interleave with training instead of
	 * taking turns with it. `measureLoop` reads on its own clock, and the most
	 * expensive reading — the training-set probe, which exists to show a
	 * generalization gap that moves over minutes — gets a much slower one.
	 */
	async function trainLoop(token: number) {
		const myGen = gen;
		const engine = lab.engine;
		while (training && engine && myGen === gen && token === runToken) {
			try {
				await engine.train(RUN_STEPS, onMetrics, SYNC_EVERY);
			} catch {
				return; // engine disposed mid-flight
			}
			if (myGen !== gen) return;
		}
	}

	/** Cheap by design: this fires many times a second and must not touch
	 * anything that re-renders. `publish` moves it into $state on the slower
	 * clock. */
	function onMetrics(m: { step: number; loss: number; stepMs: number }) {
		liveStep = m.step;
		liveLoss = m.loss;
		liveMs = liveMs ? liveMs * 0.7 + m.stepMs * 0.3 : m.stepMs;
		lossSum += m.loss;
		lossCount += 1;
	}

	function publish() {
		lab.step = liveStep;
		lab.loss = liveLoss;
		lab.msPerStep = liveMs;
		// One point per reading, and it is the MEAN of the steps since the last
		// one rather than whichever single step happened to be synced. Pushing
		// raw sync losses filled the 240-point window in seven seconds with a
		// band of minibatch noise; a 200 ms average holds forty-odd seconds of
		// run and has a shape.
		if (lossCount) {
			lossRing.push(lossSum / lossCount);
			lossSum = 0;
			lossCount = 0;
			if (lossRing.length > 240) lossRing.splice(0, lossRing.length - 240);
		}
		lossHist = lossRing.slice();
		accHist = accRing.slice();
	}

	const pause = (ms: number) => new Promise((r) => setTimeout(r, ms));

	async function measureLoop(token: number) {
		const myGen = gen;
		let probeAt = 0;
		while (training && myGen === gen && token === runToken) {
			await pause(REFRESH_MS);
			const engine = lab.engine;
			if (!training || myGen !== gen || token !== runToken || !engine) return;
			try {
				// cheap, and the only thing the eye actually tracks: the loss
				// curve and the sixteen digits on show
				await refreshGuesses();
				// The two accuracy readings are three thousand forward passes
				// between them and show numbers that move over minutes. They do
				// not belong on the same clock as everything else.
				const now = performance.now();
				if (now - probeAt >= PROBE_MS) {
					probeAt = now;
					await refreshAccuracies();
				}
				if (myGen !== gen || token !== runToken) return;
				publish();
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
			const logits = await engine.predict(xs, ids.length, 32);
			if (ids !== sampleIdx || lab.engine !== engine) return;
			guesses = ids.map((_, i) => argmax(logits, i * 10, 10));
			dists = ids.map((_, i) => softmax(logits, i * 10, 10));
		} catch {
			/* engine disposed mid-flight */
		}
	}

	/**
	 * Both accuracies, on the same number of rows, in one pass.
	 *
	 * The headline of this plate is the GAP between them, and the gap used to
	 * compare 1,500 sampled training rows against the worker's eval — which
	 * caps itself at 512, and always the first 512 of the test tail. On a
	 * 512-row sample an accuracy near 95% carries about a point of sampling
	 * noise, so the difference of the two readings jittered by about as much
	 * as the effect it was there to show, and it was shown to a tenth of a
	 * point. Matched sample sizes make the two sides comparable and the gap
	 * worth reading.
	 */
	async function refreshAccuracies() {
		if (!probe || !testProbe) return;
		try {
			const [tr, te] = await Promise.all([accuracyOn(probe), accuracyOn(testProbe)]);
			if (!Number.isNaN(tr)) lab.trainAcc = tr;
			if (!Number.isNaN(te)) {
				lab.testAcc = te;
				accRing.push(te);
				if (accRing.length > 240) accRing.splice(0, accRing.length - 240);
				if (te >= 0.9) progress.reach('digits:trained');
			}
		} catch {
			/* engine disposed mid-flight */
		}
	}

	function reshuffle() {
		pickMode = 'random';
		sampleIdx = pickSamples();
		guesses = [];
		dists = [];
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
		lab.trainAcc = NaN;
		lab.msPerStep = 0;
		lossHist = [];
		accHist = [];
		lossRing = [];
		accRing = [];
		liveStep = 0;
		liveLoss = NaN;
		liveMs = 0;
		lossSum = 0;
		lossCount = 0;
		await refreshGuesses(); // never throws — it guards internally
		lab.version += 1;
		if (wasTraining) setTraining(true);
	}

	function drawTile(t: number) {
		return (el: HTMLCanvasElement) => {
			void themePulse.tick; // redraw when the theme flips
			const m = lab.mnist;
			if (!m) return;
			blit(el, [inkImage(m.testX, t * DIM, hexRgb(readTokens(el).ink))], 64);
		};
	}

	onDestroy(() => {
		gen += 1;
		training = false;
		void lab.engine?.dispose();
		lab.clear();
	});
</script>

<Plate id="classifier" live {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'ready'}
			<span>step {lab.step}</span>
			<span aria-hidden="true">·</span>
			<span>loss {Number.isFinite(lab.loss) ? lab.loss.toFixed(3) : '—'}</span>
			<span aria-hidden="true">·</span>
			<span>{lab.msPerStep.toFixed(0)} ms/step</span>
		{:else if lab.phase === 'loading'}
			<span>fetching MNIST…</span>
		{:else if lab.phase === 'error'}
			<span style="color: var(--bad);">error</span>
		{/if}
	{/snippet}

	{#snippet actions()}
		<Btn onclick={() => setTraining(!training)} disabled={lab.phase !== 'ready'}>
			{#if training}
				<Pause size={12} aria-hidden="true" /> Pause
			{:else}
				<Play size={12} aria-hidden="true" /> Train
			{/if}
		</Btn>
		<Btn onclick={resetWeights} disabled={lab.phase !== 'ready'} title="Fresh random weights">
			<RotateCcw size={12} aria-hidden="true" /> Reset
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={boot}>
		{#if lab.phase === 'error'}
			<div class="flex flex-wrap items-center gap-3 px-4 py-4">
				<span class="text-[12.5px] text-bad">{errorMsg}</span>
				<Btn onclick={boot}>Retry</Btn>
			</div>
		{:else if lab.phase !== 'ready' || !mnist}
			<div class="flex h-[280px] flex-col items-center justify-center gap-1">
				<span class="eyebrow">fetching MNIST (≈1.7 MB, cached) · warming up the worker…</span>
				<span class="text-[12.5px] text-ink-3">
					it trains right here, on your machine — nothing leaves this page
				</span>
			</div>
		{:else}
			<div
				class="grid grid-cols-1 gap-x-7 gap-y-6 p-4 sm:p-5 lg:grid-cols-[1fr_minmax(250px,282px)]"
			>
				<!-- stage left: two dozen held-out digits and the model's reading of each -->
				<div>
					<!-- the label takes a fixed share of the row, so switching modes never
					     reflows the buttons out from under the pointer -->
					<div class="mb-2.5 flex flex-wrap items-center justify-between gap-2">
						<span class="eyebrow min-w-[240px] flex-1 truncate">
							{#if pickMode === 'hardest'}
								{STRIP} test digits · where it fails hardest
							{:else if pickMode === 'easiest'}
								{STRIP} test digits · where it is surest
							{:else}
								{STRIP} test digits · a random draw
							{/if}
						</span>
						<div class="flex shrink-0 items-center gap-2">
							<Btn
								onclick={reshuffle}
								pressed={pickMode === 'random'}
								title="A fresh sixteen, drawn at random from the test set"
							>
								<Shuffle size={12} aria-hidden="true" /> Random
							</Btn>
							<Btn
								onclick={() => rankSamples('easiest')}
								pressed={pickMode === 'easiest'}
								title="The test digits it gives the highest probability to the right answer"
							>
								<ArrowUpNarrowWide size={12} aria-hidden="true" /> Surest
							</Btn>
							<Btn
								onclick={() => rankSamples('hardest')}
								pressed={pickMode === 'hardest'}
								title="The test digits it gives the lowest probability to the right answer"
							>
								<ArrowDownWideNarrow size={12} aria-hidden="true" /> Hardest
							</Btn>
						</div>
					</div>
					<div class="grid grid-cols-4 gap-x-2.5 gap-y-3 sm:grid-cols-8">
						{#each sampleIdx as t, i (t)}
							{@const truth = mnist.testY[t]}
							{@const guess = guesses[i]}
							{@const p = dists[i]}
							{@const right = guess === truth}
							{@const tone =
								guess === undefined ? 'var(--ink-3)' : right ? 'var(--good)' : 'var(--bad)'}
							{@const surprise = p ? -Math.log(Math.max(p[truth], 1e-9)) : NaN}
							<figure
								class="m-0 flex flex-col items-center gap-[3px]"
								title={guess === undefined
									? 'not read yet'
									: `reads ${guess} — ${right ? 'correct' : `wrong, it is a ${truth}`}; surprise −log p(${truth}) = ${surprise.toFixed(2)}`}
							>
								<canvas
									class="aspect-square w-full rounded-[3px] border border-line-soft"
									aria-label="a handwritten {truth}"
									{@attach drawTile(t)}
								></canvas>
								<!-- its whole belief, not just the winner -->
								<svg
									viewBox="0 0 40 24"
									preserveAspectRatio="none"
									class="block h-10 w-full"
									aria-hidden="true"
								>
									<rect x="0" y="23.4" width="40" height="0.6" fill="var(--line)" />
									{#each DIGITS as d (d)}
										{@const h = p ? Math.max(0.7, p[d] * 23.4) : 0.7}
										<rect
											x={d * 4 + 0.4}
											y={23.4 - h}
											width="3.2"
											height={h}
											fill={d === guess ? tone : d === truth ? 'var(--ink-2)' : 'var(--ink-3)'}
											opacity={d === guess ? 0.95 : d === truth ? 0.7 : 0.4}
										/>
									{/each}
								</svg>
								<figcaption class="flex w-full items-baseline justify-center gap-1.5 leading-none">
									<span class="num text-[15px]" style="font-weight: 560; color: {tone};">
										{guess === undefined ? '·' : guess}
									</span>
									<span class="num text-[10px] text-ink-3">
										{p ? surprise.toFixed(2) : '—'}
									</span>
									<span class="sr-only">
										{guess === undefined
											? 'not read yet'
											: right
												? 'correct'
												: `wrong, it is a ${truth}`}
									</span>
								</figcaption>
							</figure>
						{/each}
					</div>
				</div>

				<!-- stage right: the model you can change, and its verdict -->
				<div class="flex flex-col gap-4">
					<div>
						<span class="eyebrow">the model</span>
						<div class="mt-2 flex flex-col gap-y-2">
							<span class="flex flex-wrap items-center gap-1" role="group" aria-label="Depth">
								<span class="eyebrow mr-1 w-11">depth</span>
								{#each DEPTHS as d (d)}
									<button class="chip" class:chip-on={lab.depth === d} onclick={() => setDepth(d)}
										>{d}</button
									>
								{/each}
							</span>
							<span
								class="flex flex-wrap items-center gap-1"
								role="group"
								aria-label="Hidden width"
							>
								<span class="eyebrow mr-1 w-11">width</span>
								{#each WIDTHS as w (w)}
									<button class="chip" class:chip-on={lab.width === w} onclick={() => setWidth(w)}
										>{w}</button
									>
								{/each}
							</span>
							<span class="flex flex-wrap items-center gap-1" role="group" aria-label="Activation">
								<span class="eyebrow mr-1 w-11">bend</span>
								{#each ACTS as act (act)}
									<button
										class="chip"
										class:chip-on={lab.activation === act}
										onclick={() => setActivation(act)}>{act}</button
									>
								{/each}
							</span>
						</div>
						<div class="num mt-2 text-[10.5px] text-ink-3">{archLabel}</div>
					</div>
					<!-- the two numbers that tell different stories, and their shortfall -->
					<div class="flex flex-col gap-3">
						<div>
							<div class="flex items-baseline justify-between gap-2">
								<span class="eyebrow">accuracy · train</span>
								<span class="text-[10.5px] text-ink-3">{PROBE.toLocaleString('en-US')} rows</span>
							</div>
							<div class="num text-[2rem] leading-[1.1] text-ink-2">{pct(lab.trainAcc)}</div>
							<div class="meter mt-1">
								<span
									style="width: {Number.isFinite(lab.trainAcc)
										? lab.trainAcc * 100
										: 0}%; background: var(--ink-3);"
								></span>
							</div>
						</div>
						<div>
							<div class="flex items-baseline justify-between gap-2">
								<span class="eyebrow">accuracy · test</span>
								<span class="text-[10.5px] text-ink-3"
									>{PROBE.toLocaleString('en-US')} rows · held out</span
								>
							</div>
							<div
								class="num text-[2.6rem] leading-[1.1]"
								style="color: {Number.isFinite(lab.testAcc) && lab.testAcc >= 0.9
									? 'var(--good)'
									: 'var(--ink)'};"
							>
								{pct(lab.testAcc)}
							</div>
							<div class="meter mt-1">
								<span
									style="width: {Number.isFinite(lab.testAcc)
										? lab.testAcc * 100
										: 0}%; background: {Number.isFinite(lab.testAcc) && lab.testAcc >= 0.9
										? 'var(--good)'
										: 'var(--ink-2)'};"
								></span>
							</div>
						</div>
						<div
							class="flex items-baseline justify-between gap-2 border-t border-line-soft pt-2 text-[11.5px]"
						>
							<span class="text-ink-3">what it memorized</span>
							<span class="num text-ink-2">{gapLabel}</span>
						</div>
					</div>
				</div>
			</div>

			<!-- slim telemetry strip: the two curves that tell different stories -->
			<div class="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line-soft px-4 py-2">
				<span class="flex min-w-40 flex-1 items-center gap-1.5">
					<span class="eyebrow shrink-0 text-[9.5px]" style="color: var(--accent);">
						loss · train
					</span>
					<svg
						viewBox="0 0 200 22"
						preserveAspectRatio="none"
						class="block h-[22px] w-full"
						role="img"
						aria-label="training loss over steps"
					>
						<path
							d={sparkPath(lossHist, 200, 22, { log: true })}
							fill="none"
							stroke="var(--accent)"
							stroke-width="1.4"
							vector-effect="non-scaling-stroke"
						/>
					</svg>
				</span>
				<span class="flex min-w-40 flex-1 items-center gap-1.5">
					<span class="eyebrow shrink-0 text-[9.5px]" style="color: var(--warm);">
						accuracy · test
					</span>
					<svg
						viewBox="0 0 200 22"
						preserveAspectRatio="none"
						class="block h-[22px] w-full"
						role="img"
						aria-label="test accuracy over steps"
					>
						<path
							d={sparkPath(accHist, 200, 22)}
							fill="none"
							stroke="var(--warm)"
							stroke-width="1.4"
							vector-effect="non-scaling-stroke"
						/>
					</svg>
				</span>
				<span class="num text-[10.5px] whitespace-nowrap text-ink-3">
					{lab.paramCount.toLocaleString('en-US')} params · {lab.device}
				</span>
			</div>
		{/if}
	</div>
</Plate>

<style>
	.meter {
		height: 3px;
		border-radius: 2px;
		background: var(--line-soft);
		overflow: hidden;
	}
	.meter span {
		display: block;
		height: 100%;
		border-radius: 2px;
		transition: width 200ms ease;
	}
</style>

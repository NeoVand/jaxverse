<script lang="ts">
	// Supervised fine-tuning, honestly: the SAME weights, a better corpus (2,381
	// greedy-material games), the same update rule. The headline is taste — the
	// capture rate of Rook's own games — and the plate charges for it out loud:
	// held-out loss on the ABANDONED random-play corpus, drifting up.
	//
	// The plate is furnished before you touch it. A baseline measurement runs as
	// it scrolls into view, so the board, the gauges and the chart's two
	// reference lines are all there from the first frame; switching the diet
	// starts a series inside a figure that already exists. Nothing in here
	// appears, disappears or resizes between evaluations — the tree is identical
	// in every state, which is what keeps a training run from flickering.
	import { ArrowUp, Pause, Play, Shuffle } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { progress } from '$lib/data/progress.svelte';
	import { plateLabel } from '$lib/data/plates';
	import { lab, LR_SFT, type LabCorpus } from './rook-context.svelte';
	import { filmGame, replayGames, type GameFilm } from './chess-eval';
	import { polyline, scale } from './chart';
	import { filmView } from './film-view.svelte';
	import Gauge from '$lib/components/ui/Gauge.svelte';
	import MiniBoard from './MiniBoard.svelte';
	import GameSentence from './GameSentence.svelte';
	import BootRow from './BootRow.svelte';

	interface Pt {
		step: number;
		v: number;
	}

	/** What each diet's own games look like — the two ends of the experiment. */
	const RANDOM_RATE = 0.12;
	const GREEDY_RATE = 0.38;

	let running = $state(false);
	let evaling = $state(false);
	let switching = $state(false);
	let lossNow = $state(NaN);
	let sftSteps = $state(0);
	let capturePts = $state<Pt[]>([]);
	let valPts = $state<Pt[]>([]);
	let legalNow = $state<number | null>(null);
	let captureNow = $state<number | null>(null);
	let film = $state<GameFilm | null>(null);
	let reached = $state(false);

	const view = filmView(() => film);
	const switched = $derived(lab.corpus === 'sft');
	const chunkSize = 25;

	/** Measure, then photograph one game. Everything it writes is a value inside
	 * a structure that already exists, so no evaluation re-mounts anything. */
	async function evalNow(g: number): Promise<void> {
		if (!lab.engine || !lab.data) return;
		evaling = true;
		try {
			// honest drift metric: held-out loss against the corpus we LEFT
			const v = await lab.valLossOn('random');
			if (g !== lab.gen) return;
			valPts = [...valPts, { step: sftSteps, v }];
			// legality should HOLD where the model now lives, so probe the diet it
			// is actually eating; drift from the old world is the val line's job
			const rate = await lab.probeLegal(g, lab.corpus);
			if (rate === null || g !== lab.gen) return;
			legalNow = rate;
			// three games for the RATE (one is too noisy to chart), the first of
			// them filmed for the board — the display never shows a crowd
			const sampled = await lab.sampleGames(3, g);
			if (!sampled || g !== lab.gen) return;
			const r = await replayGames(sampled, lab.data.decode);
			const f = await filmGame(sampled[0], lab.data.decode);
			if (g !== lab.gen) return;
			film = f;
			view.point(null);
			captureNow = r.captureRate;
			capturePts = [...capturePts, { step: sftSteps, v: r.captureRate ?? 0 }];
			if ((r.captureRate ?? 0) >= 0.25 && !reached) {
				reached = true;
				progress.reach('rook:sft');
			}
		} catch {
			/* interrupted mid-eval — the next owner repaints */
		} finally {
			evaling = false;
		}
	}

	// One baseline once the plate is actually reached, so it is furnished before
	// the reader touches anything. Guarded: once, and never mid-loop.
	let seen = $state(false);
	let baselineDone = false;
	$effect(() => {
		if (seen && lab.phase === 'ready' && !baselineDone && lab.busy === '') {
			baselineDone = true;
			void evalNow(lab.gen);
		}
	});

	/** Another draw from the same weights — sampling is stochastic. */
	async function resample(): Promise<void> {
		if (lab.phase !== 'ready' || evaling || lab.busy !== '') return;
		evaling = true;
		try {
			const g = lab.gen;
			const sampled = await lab.sampleGames(1, g);
			if (!sampled || g !== lab.gen || !lab.data) return;
			const f = await filmGame(sampled[0], lab.data.decode);
			if (g !== lab.gen) return;
			film = f;
			view.point(null);
		} catch {
			/* interrupted */
		} finally {
			evaling = false;
		}
	}

	/** Swap the corpus. Weights and optimizer state survive — that is the point. */
	async function switchDiet(which: LabCorpus): Promise<void> {
		if (lab.phase !== 'ready' || switching || lab.corpus === which) return;
		switching = true;
		const g = await lab.beginLoop('sft');
		try {
			if (g !== lab.gen || !lab.engine) return;
			sftSteps = 0;
			capturePts = [];
			valPts = [];
			await lab.useCorpus(which);
			await lab.useLr(LR_SFT); // this plate only ever fine-tunes, at its own γ
			if (g !== lab.gen) return;
			// a baseline BEFORE any step on the new diet — the curve starts honest
			await evalNow(g);
		} finally {
			switching = false;
			lab.endLoop(g);
		}
	}

	async function toggleTrain(): Promise<void> {
		if (running) {
			running = false;
			void lab.engine?.stop();
			return;
		}
		// The button answers the click before any awaiting starts: serving the new
		// diet takes a baseline measurement first, and a transport that sits on
		// "Fine-tune" for several seconds reads as a broken button.
		running = true;
		try {
			if (!switched) {
				await switchDiet('sft');
				if (!running || lab.corpus !== 'sft') return;
			}
			const g = await lab.beginLoop('sft');
			if (g !== lab.gen || lab.phase !== 'ready') return;
			await lab.useCorpus('sft');
			await lab.useLr(LR_SFT);
			if (g !== lab.gen) return;
			// weights were rewound (or reinforced) since the last fine-tune — the old
			// curve would lie about these weights, so start it over
			if (lab.stage !== 'fine-tuned' && sftSteps > 0) {
				sftSteps = 0;
				capturePts = [];
				valPts = [];
				await evalNow(g);
				if (g !== lab.gen || !running) return;
			}
			while (running && g === lab.gen && lab.engine) {
				await lab.trainChunk(chunkSize, (m) => {
					lab.liveSteps = m.step;
					lossNow = m.loss;
					sftSteps++;
				});
				if (g !== lab.gen || !running) break;
				lab.stage = 'fine-tuned';
				await evalNow(g);
			}
			// photograph the paused weights — the arena fields this student
			if (g === lab.gen && lab.stage === 'fine-tuned') await lab.captureStage('fine-tuned');
			lab.endLoop(g);
		} finally {
			running = false;
		}
	}

	// ── the chart: fixed geometry, and an x domain that only grows in coarse
	//    steps, so the series does not re-scale under the reader every 25 steps ──
	// W is close to the panel's real width, so the SVG scales ~1:1 and its 10px
	// labels stay 10px instead of being magnified by a uniform fit
	const W = 760;
	const H = 138;
	const PAD = { l: 38, r: 8, t: 10, b: 22 };
	const maxStep = $derived(Math.max(200, Math.ceil(sftSteps / 200) * 200));
	const x = $derived(scale(0, maxStep, PAD.l, W - PAD.r));
	const y = $derived(scale(0, 0.55, H - PAD.b, PAD.t));
	const dots = $derived(capturePts.map((p) => ({ px: x(p.step), py: y(p.v) })));
	// Each measurement is three sampled games — sixty-odd plies, which is a noisy
	// estimate of a rate. Both are drawn: the measurements as dots, and an
	// exponential mean as the line, so the trend is readable without hiding the
	// spread it was drawn from.
	const capPath = $derived.by(() => {
		let e = NaN;
		return polyline(
			capturePts.map((p) => {
				e = Number.isNaN(e) ? p.v : e * 0.6 + p.v * 0.4;
				return [x(p.step), y(e)];
			})
		);
	});
	const TICKS = [0, 0.1, 0.2, 0.3, 0.4, 0.5];

	const drift = $derived.by(() => {
		if (valPts.length < 2) return null;
		const first = valPts[0].v;
		const last = valPts[valPts.length - 1].v;
		return { first, last, rising: last > first + 0.005 };
	});
</script>

<Plate
	id="sft"
	live
	title="Fine-tuning — the same weights, a better diet"
	caption="Nothing about the machine changed — same parameters, same loss, same update rule — only the corpus. Style follows the diet: within a hundred-odd steps the capture rate of Rook's own games climbs from the random-play 12% toward the greedy corpus's 38%, while the held-out loss on the corpus it left drifts up. That drift is the price, printed beside the prize. Stop while you are ahead: over-tuning keeps buying style and starts paying with competence."
>
	{#snippet status()}
		{#if lab.phase === 'ready'}
			<span>
				{lab.weightsLabel} · {switched ? 'greedy games' : 'random play'}
				{#if switched}· {sftSteps} steps{/if}
				{#if running && Number.isFinite(lossNow)}· loss {lossNow.toFixed(2)}{/if}
				{#if evaling}· sampling…{:else if running}· fine-tuning{/if}
			</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		{#if lab.phase === 'ready'}
			{#if !running}
				<Btn onclick={() => void resample()} disabled={evaling || lab.busy !== ''}>
					<Shuffle size={12} aria-hidden="true" /> Sample
				</Btn>
			{/if}
			<Btn kind={running ? 'ghost' : 'primary'} onclick={() => void toggleTrain()}>
				{#if running}
					<Pause size={12} aria-hidden="true" /> Pause
				{:else}
					<Play size={12} aria-hidden="true" /> Fine-tune
				{/if}
			</Btn>
		{/if}
	{/snippet}

	<div
		use:inview={() => {
			seen = true;
			void lab.power();
		}}
	>
		{#if lab.phase === 'ready'}
			<div class="grid grid-cols-1 gap-x-7 gap-y-5 px-4 py-4 lg:grid-cols-[300px_minmax(0,1fr)]">
				<!-- the model's own game, on the chapter's one board -->
				<div class="flex flex-col gap-1.5">
					{#if view.shown}
						<MiniBoard
							fen={view.shown.fen}
							move={view.shown.uci}
							tone={view.tone}
							arrow
							showCoordinates
							size={300}
							label="A game the fine-tuning model wrote, {view.caption}"
						/>
						<p class="num text-center text-[11.5px]" style="color: {view.color};">
							{view.caption}
						</p>
						<p class="text-center text-[10.5px] text-ink-3">
							the board the move was played from — the arrow is the move
						</p>
					{:else}
						<div
							class="flex aspect-square w-[300px] max-w-full items-center justify-center rounded-[3px] border border-line-soft px-6 text-center text-[12.5px] text-ink-3"
						>
							{evaling
								? 'sampling — the first run also compiles the GPU kernels…'
								: 'no sample yet'}
						</div>
					{/if}
				</div>

				<!-- one panel: which diet, what it bought, what it cost, what it wrote -->
				<div class="flex min-w-0 flex-col justify-between gap-4">
					<div class="flex flex-wrap items-center gap-x-4 gap-y-2">
						<span class="eyebrow">diet</span>
						<span class="seg" role="group" aria-label="Training corpus">
							<button
								class:on={!switched}
								disabled={switching || running}
								onclick={() => void switchDiet('random')}
							>
								random play
							</button>
							<button
								class:on={switched}
								disabled={switching || running}
								onclick={() => void switchDiet('sft')}
							>
								greedy games
							</button>
						</span>
						<span class="num text-[11px] text-ink-3">
							{switched ? `${sftSteps} fine-tuning steps` : 'the pretraining corpus'}
						</span>
					</div>

					<div class="border-t border-line-soft pt-3.5">
						<svg
							viewBox="0 0 {W} {H}"
							class="block w-full"
							role="img"
							aria-label="Capture rate of the model's own sampled games across fine-tuning steps, between the random-play diet at 12 percent and the greedy diet at 38 percent"
						>
							{#each TICKS as t (t)}
								<line
									x1={PAD.l}
									x2={W - PAD.r}
									y1={y(t)}
									y2={y(t)}
									stroke="var(--line-soft)"
									stroke-width="1"
								/>
								<text
									x={PAD.l - 6}
									y={y(t) + 3}
									text-anchor="end"
									class="num tick"
									fill="var(--ink-3)">{(t * 100).toFixed(0)}%</text
								>
							{/each}
							<!-- the two diets, as the ends of the experiment -->
							<line
								x1={PAD.l}
								x2={W - PAD.r}
								y1={y(RANDOM_RATE)}
								y2={y(RANDOM_RATE)}
								stroke="var(--ink-3)"
								stroke-width="1"
								stroke-dasharray="3 4"
							/>
							<text
								x={W - PAD.r}
								y={y(RANDOM_RATE) - 4}
								text-anchor="end"
								class="num tick"
								fill="var(--ink-3)">random-play diet ≈12%</text
							>
							<line
								x1={PAD.l}
								x2={W - PAD.r}
								y1={y(GREEDY_RATE)}
								y2={y(GREEDY_RATE)}
								stroke="var(--warm)"
								stroke-width="1"
								stroke-dasharray="3 4"
								opacity="0.55"
							/>
							<text
								x={W - PAD.r}
								y={y(GREEDY_RATE) - 4}
								text-anchor="end"
								class="num tick"
								fill="var(--warm)">greedy diet ≈38%</text
							>
							{#if capturePts.length > 0}
								{#each dots as d, i (i)}
									<circle cx={d.px} cy={d.py} r="1.9" fill="var(--warm)" opacity="0.45" />
								{/each}
								<path d={capPath} fill="none" stroke="var(--warm)" stroke-width="1.6" />
							{/if}
							<text x={PAD.l} y={H - 6} class="num tick" fill="var(--ink-3)">
								{capturePts.length > 1
									? 'fine-tuning steps →   ·   each dot is one measurement of three fresh games; the line is their running mean'
									: 'press Fine-tune — the curve starts here'}
							</text>
						</svg>
					</div>

					<div
						class="grid items-end gap-x-6 gap-y-2 border-t border-line-soft pt-3.5 sm:grid-cols-[minmax(0,260px)_minmax(0,1fr)]"
					>
						<Gauge label="captures per legal move" value={captureNow} tone="warm" />
						<div class="num flex flex-col gap-0.5 text-[11px] text-ink-3">
							<span>legal moves · {(legalNow === null ? 0 : legalNow * 100).toFixed(0)}%</span>
							<span>
								val · corpus it left ·
								{#if drift}
									<span class="text-ink">{drift.first.toFixed(2)} → {drift.last.toFixed(2)}</span>
									{#if drift.rising}<span style="color: var(--warm);"> ↑ forgetting</span>{/if}
								{:else}
									<span class="text-ink"
										>{valPts.length ? valPts[valPts.length - 1].v.toFixed(2) : '—'}</span
									>
								{/if}
							</span>
						</div>
					</div>

					<div class="border-t border-line-soft pt-3.5">
						{#if film}
							<GameSentence {view} />
						{:else}
							<p class="text-[12px] text-ink-3">
								{evaling ? 'sampling…' : 'the first sample arrives with the first measurement'}
							</p>
						{/if}
					</div>

					{#if reached}
						<p class="font-serif text-[13px] italic" style="color: var(--good);">
							The taste took — the capture rate crossed 25%.
							<a
								href="#rook-play"
								class="not-italic underline decoration-dotted underline-offset-4"
							>
								<ArrowUp size={11} aria-hidden="true" class="inline" /> play it in {plateLabel(
									'rook',
									'play'
								)}
							</a>
						</p>
					{/if}
				</div>
			</div>
		{:else}
			<BootRow />
		{/if}
	</div>
</Plate>

<style>
	.tick {
		font-size: 10px;
	}
</style>

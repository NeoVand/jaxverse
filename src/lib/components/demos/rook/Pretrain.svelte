<script lang="ts">
	// Pretraining, with a time machine. Four saved checkpoints from
	// the real 2,600-step run, a live train button for readers who want to feel
	// the loss move, and the chapter's honest metric: how often does a model
	// that was only ever asked to predict-the-next-token play a LEGAL move?
	//
	// The sample is ONE game on a board large enough to read, not three strips:
	// pointing at any move in the sentence plays it there, so the reader can
	// walk the model's own writing move by move and see exactly where the judge
	// stopped believing it.
	import { Pause, Play, Shuffle } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { lab, LR_PRETRAIN } from './rook-context.svelte';
	import { filmGame, type GameFilm } from './chess-eval';
	import { filmView } from './film-view.svelte';
	import Gauge from '$lib/components/ui/Gauge.svelte';
	import MiniBoard from './MiniBoard.svelte';
	import GameSentence from './GameSentence.svelte';
	import BootRow from './BootRow.svelte';
	import { sparkPath } from '$lib/viz/spark';

	let running = $state(false);
	let evaling = $state(false);
	let lossNow = $state(NaN);
	let msPerStep = $state(0);
	let valNow = $state<number | null>(null);
	let legalNow = $state<number | null>(null);
	/** The current sampled game, replayed ply by ply against the judge. */
	let film = $state<GameFilm | null>(null);
	/** Live-training histories for the telemetry strip; cleared on rewind. */
	let lossHist = $state<number[]>([]);
	let valHist = $state<number[]>([]);

	const waypoints = $derived(lab.manifest?.waypoints ?? []);
	const atWaypoint = $derived(lab.liveSteps === 0 && lab.stage === 'pretrained');
	const chunkSize = 25;

	const view = filmView(() => film);

	// One baseline measurement after the inview boot, so the gauge and sample
	// board aren't blank. Guarded: once, and never mid-loop.
	let baselineDone = false;
	$effect(() => {
		if (lab.phase === 'ready' && !baselineDone && lab.busy === '') {
			baselineDone = true;
			void evalNow(lab.gen);
		}
	});

	/** Draw one game from the empty board and replay it against chess.js. */
	async function sampleFilm(myGen: number): Promise<void> {
		if (!lab.data) return;
		const sampled = await lab.sampleGames(1, myGen);
		if (!sampled || myGen !== lab.gen) return;
		const f = await filmGame(sampled[0], lab.data.decode);
		if (myGen !== lab.gen) return;
		film = f;
		view.point(null);
	}

	/** Refresh val loss, run the legal-move probe, photograph one game. */
	async function evalNow(myGen: number): Promise<void> {
		if (!lab.engine || !lab.data) return;
		evaling = true;
		try {
			const v = await lab.valLossOn('random');
			if (myGen !== lab.gen) return;
			valNow = v;
			valHist = [...valHist.slice(-199), v];
			// the same measurement the manifest recorded per waypoint: argmax
			// next move from 32 fixed real positions, judged by chess.js
			const rate = await lab.probeLegal(myGen);
			if (rate === null || myGen !== lab.gen) return;
			legalNow = rate;
			await sampleFilm(myGen);
		} catch {
			/* interrupted mid-eval — the next owner repaints */
		} finally {
			evaling = false;
		}
	}

	/** Another draw from the same weights — sampling is stochastic, and one
	 * game is a sample of one. */
	async function resample(): Promise<void> {
		if (lab.phase !== 'ready' || evaling || lab.busy !== '') return;
		evaling = true;
		try {
			await sampleFilm(lab.gen);
		} catch {
			/* interrupted — the next owner repaints */
		} finally {
			evaling = false;
		}
	}

	async function toWaypoint(step: number): Promise<void> {
		if (lab.phase !== 'ready') return;
		running = false;
		await lab.loadWaypoint(step);
		valNow = null;
		legalNow = null;
		film = null;
		view.point(null);
		// the curves described weights that no longer exist — start them over
		lossHist = [];
		valHist = [];
		await evalNow(lab.gen);
	}

	async function toggleTrain(): Promise<void> {
		if (running) {
			running = false;
			void lab.engine?.stop();
			return;
		}
		const g = await lab.beginLoop('pretrain');
		if (g !== lab.gen || lab.phase !== 'ready') return;
		// this plate always trains on the random-play diet, at the run's own γ
		await lab.useCorpus('random');
		await lab.useLr(LR_PRETRAIN);
		if (g !== lab.gen) return;
		running = true;
		while (running && g === lab.gen && lab.engine) {
			await lab.trainChunk(chunkSize, (m) => {
				lab.liveSteps = m.step;
				lossNow = m.loss;
				lossHist = [...lossHist.slice(-399), m.loss];
				msPerStep = msPerStep ? msPerStep * 0.7 + m.stepMs * 0.3 : m.stepMs;
			});
			if (g !== lab.gen || !running) break;
			await evalNow(g);
		}
		lab.endLoop(g);
		running = false;
	}
</script>

<Plate
	id="pretrain"
	live
	title="Pretraining — predict the next move"
	caption="Nothing in the loss says “chess”: the model is only playing predict-the-next-token, exactly as in the chapter on next-token prediction. Legality arrives as a side effect — the gauge asks, at 32 positions from real games, whether Rook's top-choice move is legal there; rewind to step 0 and watch it climb. Beside it is one game the model wrote, replayed against the referee: point at any move to see it played, and at the ✕ to see the board where the judge stopped believing it."
>
	{#snippet status()}
		{#if lab.phase === 'ready'}
			<span>
				{lab.weightsLabel} · val {valNow === null ? '—' : valNow.toFixed(2)}
				{#if running && Number.isFinite(lossNow)}
					· loss {lossNow.toFixed(2)} · {msPerStep.toFixed(0)} ms/step{/if}
				{#if evaling}· sampling…{:else if running}· training{/if}
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
			<Btn onclick={() => void toggleTrain()}>
				{#if running}
					<Pause size={12} aria-hidden="true" /> Pause
				{:else}
					<Play size={12} aria-hidden="true" /> Train
				{/if}
			</Btn>
		{/if}
	{/snippet}

	<div use:inview={() => void lab.power()}>
		{#if lab.phase === 'ready'}
			<div class="grid grid-cols-1 gap-x-7 gap-y-5 px-4 py-4 lg:grid-cols-[300px_minmax(0,1fr)]">
				<!-- the board is the figure; its caption names what you are looking at -->
				<div class="flex flex-col gap-1.5">
					{#if view.shown}
						<MiniBoard
							fen={view.shown.fen}
							move={view.shown.uci}
							tone={view.tone}
							arrow
							showCoordinates
							size={300}
							label="The sampled game, {view.caption}"
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

				<!-- one panel, three groups, hairline-ruled: where the weights are,
				     what they measure, and what they just wrote -->
				<div class="flex min-w-0 flex-col justify-between gap-4">
					<div class="flex flex-wrap items-center gap-x-4 gap-y-2">
						<span class="eyebrow">time machine</span>
						<span
							class="flex flex-wrap items-center gap-1"
							role="group"
							aria-label="Saved checkpoints"
						>
							{#each waypoints as wp (wp.step)}
								<button
									class="chip num"
									class:chip-on={atWaypoint && lab.waypointStep === wp.step}
									onclick={() => void toWaypoint(wp.step)}
									title="Load the step-{wp.step} checkpoint ({Math.round(
										wp.legalRate * 100
									)}% legal)"
								>
									step {wp.step} · {Math.round(wp.legalRate * 100)}%
								</button>
							{/each}
						</span>
					</div>

					<div
						class="grid items-end gap-x-6 gap-y-2 border-t border-line-soft pt-3.5 sm:grid-cols-[minmax(0,300px)_minmax(0,1fr)]"
					>
						<Gauge label="legal moves · 32 real positions" value={legalNow} />
						<div class="num flex flex-col gap-0.5 text-[11px] text-ink-3 sm:text-right">
							<span>
								val <span class="text-ink">{valNow === null ? '—' : valNow.toFixed(3)}</span> · held out
							</span>
							<span>weights {lab.weightsLabel}{evaling ? ' · measuring…' : ''}</span>
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
				</div>
			</div>

			{#if lossHist.length > 1 || valHist.length > 1}
				<!-- slim telemetry strip: the live loss and the held-out check -->
				<div
					class="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line-soft px-4 py-2"
				>
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
							loss · held out
						</span>
						<svg
							viewBox="0 0 200 22"
							preserveAspectRatio="none"
							class="block h-[22px] w-full"
							role="img"
							aria-label="validation loss over evaluations"
						>
							<path
								d={sparkPath(valHist, 200, 22, { log: true })}
								fill="none"
								stroke="var(--warm)"
								stroke-width="1.4"
								vector-effect="non-scaling-stroke"
							/>
						</svg>
					</span>
					<span class="num text-[10.5px] whitespace-nowrap text-ink-3">
						{#if msPerStep > 0}{msPerStep.toFixed(0)} ms/step{/if}
					</span>
				</div>
			{/if}
		{:else}
			<BootRow />
		{/if}
	</div>
</Plate>

<style>
</style>

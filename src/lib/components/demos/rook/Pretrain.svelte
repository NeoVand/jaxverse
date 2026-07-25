<script lang="ts">
	// Plate I — pretraining, with a time machine. Four saved checkpoints from
	// the real 2,600-step run, a live train button for readers who want to feel
	// the loss move, and the chapter's honest metric: how often does a model
	// that was only ever asked to predict-the-next-token play a LEGAL move?
	import { Pause, Play } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import SpeedChips from '$lib/components/ui/SpeedChips.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { lab, LR_PRETRAIN } from './rook-context.svelte';
	import {
		replayGames,
		snapshotGame,
		snapCaption,
		snapTone,
		type BoardSnap,
		type PlyMark
	} from './chess-eval';
	import Gauge from './Gauge.svelte';
	import MiniBoard from './MiniBoard.svelte';
	import BootRow from './BootRow.svelte';

	interface GameView {
		snaps: BoardSnap[];
		plies: PlyMark[];
	}

	let running = $state(false);
	let evaling = $state(false);
	let speed = $state(1);
	let lossNow = $state(NaN);
	let msPerStep = $state(0);
	let valNow = $state<number | null>(null);
	let legalNow = $state<number | null>(null);
	let games = $state<GameView[]>([]);

	const waypoints = $derived(lab.manifest?.waypoints ?? []);
	const atWaypoint = $derived(lab.liveSteps === 0 && lab.stage === 'pretrained');
	const chunkSize = $derived(speed === 0 ? 150 : 25 * speed);

	// One baseline measurement after the inview boot, so the gauge and sample
	// boards aren't blank. Guarded: once, and never mid-loop.
	let baselineDone = false;
	$effect(() => {
		if (lab.phase === 'ready' && !baselineDone && lab.busy === '') {
			baselineDone = true;
			void evalNow(lab.gen);
		}
	});

	/** Refresh val loss, run the legal-move probe, photograph three games. */
	async function evalNow(myGen: number): Promise<void> {
		if (!lab.engine || !lab.data) return;
		evaling = true;
		try {
			const v = await lab.valLossOn('random');
			if (myGen !== lab.gen) return;
			valNow = v;
			// the same measurement the manifest recorded per waypoint: argmax
			// next move from 32 fixed real positions, judged by chess.js
			const rate = await lab.probeLegal(myGen);
			if (rate === null || myGen !== lab.gen) return;
			legalNow = rate;
			const sampled = await lab.sampleGames(3, myGen);
			if (!sampled || myGen !== lab.gen) return;
			const decode = lab.data.decode;
			const r = await replayGames(sampled, decode);
			const views: GameView[] = [];
			for (let i = 0; i < sampled.length; i++) {
				views.push({ snaps: await snapshotGame(sampled[i], decode), plies: r.games[i] });
			}
			if (myGen !== lab.gen) return;
			games = views;
		} catch {
			/* interrupted mid-eval — the next owner repaints */
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
		games = [];
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
	n={1}
	title="Pretraining — predict the next move"
	caption="The model is only ever playing predict-the-next-token, exactly as in Chapter 5 — nothing in its loss says “chess”. Legality is emerging as a side effect: the gauge asks, at 32 positions drawn from real games, whether Rook's top-choice next move is legal there — rewind to step 0 and watch it climb. Each sampled game is photographed at three moments; ✕ marks where the judge stopped believing it."
>
	{#snippet status()}
		{#if lab.phase === 'ready'}
			<span>
				{lab.weightsLabel} · val {valNow === null ? '—' : valNow.toFixed(2)}
				{#if evaling}· sampling…{:else if running}· training{/if}
			</span>
		{/if}
	{/snippet}

	<div use:inview={() => void lab.power()}>
		{#if lab.phase === 'ready'}
			<div class="flex flex-wrap items-center gap-x-5 gap-y-3 border-b border-line-soft px-4 py-3">
				<span class="flex flex-wrap items-center gap-1" role="group" aria-label="Saved checkpoints">
					<span class="eyebrow mr-1">time machine</span>
					{#each waypoints as wp (wp.step)}
						<button
							class="chip num"
							class:chip-on={atWaypoint && lab.waypointStep === wp.step}
							onclick={() => void toWaypoint(wp.step)}
							title="Load the step-{wp.step} checkpoint ({Math.round(wp.legalRate * 100)}% legal)"
						>
							step {wp.step} · {Math.round(wp.legalRate * 100)}%
						</button>
					{/each}
				</span>
				<span class="ml-auto flex flex-wrap items-center gap-x-4 gap-y-2">
					<SpeedChips bind:value={speed} />
					{#if running}
						<span class="num text-[11.5px] text-ink-3">
							loss {Number.isFinite(lossNow) ? lossNow.toFixed(2) : '—'} · {msPerStep.toFixed(0)} ms/step
						</span>
					{/if}
					<Btn onclick={() => void toggleTrain()}>
						{#if running}
							<Pause size={12} aria-hidden="true" /> Pause
						{:else}
							<Play size={12} aria-hidden="true" /> Train live from here
						{/if}
					</Btn>
				</span>
			</div>

			<div class="grid grid-cols-1 gap-x-8 gap-y-5 px-4 py-4 md:grid-cols-[240px_minmax(0,1fr)]">
				<div class="flex flex-col gap-4">
					<Gauge label="legal moves · 32 real positions" value={legalNow} />
					<div class="num flex flex-col gap-1 text-[11.5px] text-ink-2">
						<span>val loss (random-play corpus) · {valNow === null ? '—' : valNow.toFixed(3)}</span>
						<span class="text-ink-3">
							weights {lab.weightsLabel}
							{#if evaling}· measuring…{/if}
						</span>
					</div>
				</div>

				<div class="flex min-w-0 flex-col gap-4">
					<span class="eyebrow">three sampled games · temp 0.7 · its own moves as context</span>
					{#if games.length === 0}
						<p class="py-2 text-[12.5px] text-ink-3">
							{evaling
								? 'sampling — the first run also compiles the GPU kernels…'
								: 'no samples yet'}
						</p>
					{:else}
						{#each games as g, gi (gi)}
							<div
								class="flex flex-col gap-2 border-t border-line-soft pt-3 first:border-t-0 first:pt-0"
							>
								<div class="flex flex-wrap items-start gap-3">
									{#each g.snaps as s (s.ply + s.kind)}
										<MiniBoard
											fen={s.fen}
											move={s.uci}
											tone={snapTone(s)}
											caption={snapCaption(s)}
											size={132}
										/>
									{/each}
								</div>
								<p
									class="num flex flex-wrap gap-x-1.5 gap-y-0.5 text-[10px] leading-snug text-ink-3"
								>
									<span>⟨game⟩</span>
									{#each g.plies as p, pi (pi)}
										{#if p.legal}
											<span>{p.uci}</span>
										{:else}
											<span class="text-bad" title="illegal in the position where it was played"
												>{p.uci}✕</span
											>
										{/if}
									{/each}
								</p>
							</div>
						{/each}
					{/if}
				</div>
			</div>
		{:else}
			<BootRow />
		{/if}
	</div>
</Plate>

<style>
	.chip {
		font-size: 11px;
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

<script lang="ts">
	// RLVR, for real. Each iteration: sample G=8 continuations of a
	// real opening in lockstep, let chess.js grade every rollout, standardize
	// the rewards inside the group (GRPO's baseline trick), and apply a true
	// REINFORCE update on the generated tokens only. No demonstrations
	// anywhere — only a judge.
	import { ArrowUp, Pause, Play } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { progress } from '$lib/data/progress.svelte';
	import { plateLabel } from '$lib/data/plates';
	import { lab, LR_RL } from './rook-context.svelte';
	import {
		pickPrefix,
		positionAfter,
		scoreRollout,
		type BoardSnap,
		type RolloutPly
	} from './chess-eval';
	import { polyline, scale } from './chart';
	import MiniBoard from './MiniBoard.svelte';
	import BootRow from './BootRow.svelte';

	const G = 8; // group size — constant on purpose: the worker jits per shape
	const MAX_NEW = 14;
	const S = 128; // blockSize; rlStep rows are [G × (S+1)]

	interface GroupRow {
		plies: RolloutPly[];
		legalPlies: number;
		attempted: number;
		reward: number;
		adv: number | null;
		snap: BoardSnap;
	}

	let running = $state(false);
	let iter = $state(0);
	let updates = $state(0);
	let skips = $state(0);
	let compiling = $state(false);
	let skippedLast = $state(false);
	let prefixShown = $state<string[]>([]);
	/** The position all eight rollouts start from — the board's resting state. */
	let prefixFen = $state('');
	/** Which rollout the reader is pointing at; null shows the shared position. */
	let hovered = $state<number | null>(null);
	let group = $state<GroupRow[]>([]);
	let meanPts = $state<Array<{ it: number; v: number }>>([]);
	let legalPts = $state<Array<{ it: number; v: number }>>([]);
	let reached = $state(false);
	let errMsg = $state('');

	// display pacing only — the work itself always runs full-tilt
	const breathMs = 150;

	async function toggleRun(): Promise<void> {
		if (running) {
			running = false;
			return;
		}
		const g = await lab.beginLoop('rlvr');
		if (g !== lab.gen || lab.phase !== 'ready' || !lab.engine || !lab.data) return;
		// small steps: REINFORCE without a KL leash collapses at pretraining γ
		await lab.useLr(LR_RL);
		if (g !== lab.gen) return;
		// weights were rewound or re-tuned since the last run — restart the story
		if (lab.stage !== 'reinforced' && iter > 0) {
			iter = 0;
			updates = 0;
			skips = 0;
			meanPts = [];
			legalPts = [];
			group = [];
			skippedLast = false;
			prefixShown = [];
			prefixFen = '';
			hovered = null;
		}
		errMsg = '';
		running = true;
		while (running && g === lab.gen) {
			try {
				await iterate(g);
			} catch (e) {
				errMsg = e instanceof Error ? e.message : String(e);
				break;
			}
			// let the inspector paint between iterations
			if (breathMs > 0) await new Promise((r) => setTimeout(r, breathMs));
		}
		// photograph the paused weights — the arena (the arena) fields this student
		if (g === lab.gen && lab.stage === 'reinforced') await lab.captureStage('reinforced');
		lab.endLoop(g);
		running = false;
	}

	async function iterate(g: number): Promise<void> {
		const engine = lab.engine;
		const data = lab.data;
		if (!engine || !data) return;

		// 1 — a real opening, 4–10 plies, different every iteration
		const prefix = pickPrefix(data.tokens, lab.gameStarts);
		const prefixUci = prefix.slice(1).map((id) => data.decode(id));

		// 2 — G rollouts of the same position, sampled in lockstep
		compiling = iter === 0;
		const seqs = await engine.sampleGroup(prefix, {
			g: G,
			maxNew: MAX_NEW,
			temperature: 1.0,
			topK: 40
		});
		compiling = false;
		if (g !== lab.gen) return;

		// 3 — the judge grades every rollout from the prefix position
		const scores = [];
		for (const cont of seqs) scores.push(await scoreRollout(prefixUci, cont, data.decode));
		if (g !== lab.gen) return;
		const rewards = scores.map((s) => s.reward);
		const mean = rewards.reduce((a, b) => a + b, 0) / G;
		const sd = Math.sqrt(rewards.reduce((a, r) => a + (r - mean) ** 2, 0) / G);

		iter++;
		prefixShown = prefixUci;
		prefixFen = await positionAfter(prefixUci);
		if (g !== lab.gen) return;
		hovered = null;
		meanPts = [...meanPts.slice(-249), { it: iter, v: mean }];

		// 4 — group-relative advantages; equal rewards mean no gradient, honestly
		if (sd < 1e-6) {
			skippedLast = true;
			skips++;
			group = scores.map((s) => ({
				plies: s.plies,
				legalPlies: s.legalPlies,
				attempted: s.attempted,
				reward: s.reward,
				adv: null,
				snap: s.snap
			}));
		} else {
			skippedLast = false;
			const advs = rewards.map((r) => (r - mean) / sd);
			group = scores.map((s, i) => ({
				plies: s.plies,
				legalPlies: s.legalPlies,
				attempted: s.attempted,
				reward: s.reward,
				adv: advs[i],
				snap: s.snap
			}));
			// 5 — one REINFORCE update on the generated tokens only
			const rows = new Int32Array(G * (S + 1)).fill(-1);
			const starts: number[] = [];
			for (let i = 0; i < G; i++) {
				const cont = seqs[i].slice(0, scores[i].attempted); // scored plies only
				const full = [...prefix, ...cont];
				for (let t = 0; t < Math.min(full.length, S + 1); t++) rows[i * (S + 1) + t] = full[t];
				starts.push(prefix.length);
			}
			const res = await engine.rlStep(rows, advs, starts);
			if (g !== lab.gen) return;
			if (!res.skipped) {
				updates++;
				lab.liveSteps = res.step;
				lab.stage = 'reinforced';
			}
		}

		// 6 — every 5 iterations, an outside opinion: the same legal-move probe
		// Plates I and III use (argmax from 32 fixed real positions)
		if (iter % 5 === 0) {
			const rate = await lab.probeLegal(g);
			if (rate !== null && g === lab.gen) {
				legalPts = [...legalPts.slice(-99), { it: iter, v: rate }];
			}
		}

		// milestone: sustained mean reward, or an outside legal rate ≥ 95%
		const last5 = meanPts.slice(-5);
		const rollingMean = last5.reduce((a, p) => a + p.v, 0) / Math.max(last5.length, 1);
		const lastLegal = legalPts.length ? legalPts[legalPts.length - 1].v : 0;
		if (!reached && (lastLegal >= 0.95 || (last5.length === 5 && rollingMean >= 0.9))) {
			reached = true;
			progress.reach('rook:rlvr');
		}
	}

	// Ranked by what the judge paid, best first: the group IS the baseline, so
	// sorting makes the line between positive and negative advantage visible as
	// a line rather than as eight scattered signs.
	const ranked = $derived(
		group.map((r, i) => ({ ...r, n: i + 1 })).sort((a, b) => b.reward - a.reward)
	);
	/** First row the group's own mean rejects — the baseline, drawn where it falls. */
	const cutIndex = $derived(ranked.findIndex((r) => r.adv !== null && r.adv < 0));

	const shown = $derived(hovered === null ? null : (ranked.find((r) => r.n === hovered) ?? null));
	const boardFen = $derived(shown ? shown.snap.fen : prefixFen);
	const boardMove = $derived(shown ? shown.snap.uci : '');
	const boardTone = $derived(shown?.snap.kind === 'illegal' ? 'bad' : 'accent');
	const boardCaption = $derived.by(() => {
		if (!shown) return prefixFen ? 'the position all eight start from' : '';
		if (shown.snap.kind === 'illegal')
			return `rollout ${shown.n} · ply ${shown.snap.ply} ✕ ${shown.snap.uci}`;
		if (shown.attempted === 0) return `rollout ${shown.n} · ended immediately`;
		return `rollout ${shown.n} · held all ${shown.attempted}`;
	});

	// ── chart geometry. W is close to the panel's real width so the SVG scales
	//    ~1:1 and its 10px labels stay 10px — no preserveAspectRatio stretching ──
	const W = 370;
	const H = 240;
	const PAD = { l: 30, r: 6, t: 10, b: 20 };
	const TICKS = [0, 0.5, 1.0, 1.5];
	// the x domain grows in coarse steps, so the series does not re-scale under
	// the reader on every iteration
	const maxIt = $derived(Math.max(30, Math.ceil(iter / 30) * 30));
	const minIt = $derived(meanPts.length ? meanPts[0].it : 0);
	const x = $derived(scale(minIt, maxIt, PAD.l, W - PAD.r));
	const y = $derived(scale(0, 1.5, H - PAD.b, PAD.t));
	const meanPath = $derived(polyline(meanPts.map((p) => [x(p.it), y(p.v)])));
	const legalPath = $derived(polyline(legalPts.map((p) => [x(p.it), y(p.v)])));
	const legalDots = $derived(legalPts.map((p) => ({ px: x(p.it), py: y(p.v) })));
	const meanNow = $derived(meanPts.length ? meanPts[meanPts.length - 1].v : null);
	const legalNow = $derived(legalPts.length ? legalPts[legalPts.length - 1].v : null);
</script>

<Plate
	id="rlvr"
	live
	title="RLVR — reinforcement from a judge"
	caption="No example ever said “this move is good”. A verifier said “this rollout held up longer”, the group's own average set the bar, and the gradient did the rest — which is why the eight are ranked here with that average drawn as a line: everything above it gets reinforced, everything below it discouraged. Point at any rollout to see the board where the judge stopped it. Sample, verify, standardize, reinforce: a small, honest cousin of how frontier models learn to reason."
>
	{#snippet status()}
		{#if lab.phase === 'ready'}
			<span>
				G = 8 · iter {iter} · updates {updates} · skips {skips}
				{#if compiling}· compiling kernels…{:else if running}· running{/if}
			</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		{#if lab.phase === 'ready'}
			<Btn kind={running ? 'ghost' : 'primary'} onclick={() => void toggleRun()}>
				{#if running}
					<Pause size={12} aria-hidden="true" /> Pause
				{:else}
					<Play size={12} aria-hidden="true" /> Run RLVR
				{/if}
			</Btn>
		{/if}
	{/snippet}

	<div use:inview={() => void lab.power()}>
		{#if lab.phase === 'ready'}
			{#if errMsg}
				<div class="border-b border-line-soft px-4 py-2 text-[12px] text-bad">{errMsg}</div>
			{/if}

			<!-- one row, three roughly square cells: the position, the group graded
			     against it, and the run so far. Pointing at a rollout in the middle
			     cell paints the cell to its left. -->
			<div class="grid px-4 pt-1 pb-2 lg:grid-cols-[300px_minmax(0,1fr)_minmax(0,1fr)]">
				<section class="col lg:pr-6">
					{#if boardFen}
						<MiniBoard
							fen={boardFen}
							move={boardMove}
							tone={boardTone}
							arrow={!!boardMove}
							showCoordinates
							size={300}
							label={boardCaption}
						/>
						<p
							class="num text-center text-[11.5px]"
							style="color: {shown?.snap.kind === 'illegal' ? 'var(--bad)' : 'var(--ink-2)'};"
						>
							{boardCaption}
						</p>
					{:else}
						<div
							class="flex aspect-square w-[300px] max-w-full items-center justify-center rounded-[3px] border border-line-soft px-6 text-center text-[12.5px] text-ink-3"
						>
							{compiling ? 'compiling the group-sampling kernels…' : 'press Run RLVR'}
						</div>
					{/if}
					<div class="mt-auto min-h-[3rem]">
						{#if shown}
							<p class="num flex flex-wrap gap-x-1.5 gap-y-0.5 text-[10px] leading-snug text-ink-3">
								{#each shown.plies as p, pi (pi)}
									<span
										class:text-bad={p.state === 'illegal'}
										class:text-ink-2={p.state === 'legal'}
										>{p.uci}{#if p.state === 'illegal'}✕{/if}</span
									>
								{:else}
									<span>it ended the game immediately</span>
								{/each}
							</p>
						{:else if prefixShown.length > 0}
							<p class="num flex flex-wrap gap-x-1.5 gap-y-0.5 text-[10px] leading-snug text-ink-3">
								<span class="text-ink-2">⟨game⟩</span>
								{#each prefixShown as u, i (i)}
									<span>{u}</span>
								{/each}
							</p>
						{/if}
					</div>
				</section>

				<section class="col lg:border-l lg:px-6">
					<div class="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
						<span class="eyebrow">the group · 8 rollouts</span>
						{#if ranked.length > 0}
							<span class="num text-[10px] text-ink-3">point at one</span>
						{/if}
					</div>
					<div>
						{#each ranked as r, i (r.n)}
							{#if i === cutIndex && cutIndex > 0}
								<div class="baseline-rule">
									<span class="num">the group's own mean — the baseline</span>
								</div>
							{/if}
							<button
								type="button"
								class="roll"
								class:on={hovered === r.n}
								onmouseenter={() => (hovered = r.n)}
								onmouseleave={() => (hovered = null)}
								onfocus={() => (hovered = r.n)}
								onblur={() => (hovered = null)}
							>
								<span class="num text-ink-3">{r.n}</span>
								<span class="num text-ink-2">{r.legalPlies}/{r.attempted}</span>
								<span class="track">
									<span
										class="fill"
										style="width: {((r.reward / 1.5) * 100).toFixed(0)}%; background: var(--warm);"
									></span>
								</span>
								<span class="num text-right text-ink">{r.reward.toFixed(2)}</span>
								{#if r.adv === null}
									<span class="num text-right text-ink-3">—</span>
								{:else}
									<span
										class="num text-right"
										style="color: {r.adv >= 0 ? 'var(--good)' : 'var(--bad)'};"
									>
										{r.adv >= 0 ? '+' : '−'}{Math.abs(r.adv).toFixed(2)}
									</span>
								{/if}
							</button>
						{:else}
							<!-- the shape the group will take, so pressing Run fills a table that
							     is already there instead of growing one -->
							{#each Array.from({ length: 8 }, (_, k) => k) as k (k)}
								<div class="roll" style="opacity: 0.35;">
									<span class="num text-ink-3">{k + 1}</span>
									<span class="num text-ink-3">—/14</span>
									<span class="track"></span>
									<span class="num text-right text-ink-3">—</span>
									<span class="num text-right text-ink-3">—</span>
								</div>
							{/each}
						{/each}
					</div>
					<p class="mt-auto text-[10.5px] leading-snug text-ink-3">
						{#if ranked.length === 0}
							{compiling
								? 'compiling the group-sampling kernels…'
								: 'press Run RLVR — eight rollouts of one position, graded and ranked here'}
						{:else}
							rollout · legal / attempted · reward, max 1.5 · advantage Â against the group's mean
						{/if}
					</p>
				</section>

				<section class="col lg:border-l lg:pl-6">
					<span class="eyebrow">the run so far</span>
					<svg
						viewBox="0 0 {W} {H}"
						class="block w-full"
						role="img"
						aria-label="Mean reward of each group per iteration, and the legal-move probe every five iterations"
					>
						{#each TICKS as t (t)}
							<line
								x1={PAD.l}
								x2={W - PAD.r}
								y1={y(t)}
								y2={y(t)}
								stroke="var(--line-soft)"
								stroke-width="1"
								stroke-dasharray={t === 1 ? '3 4' : undefined}
							/>
							<text
								x={PAD.l - 6}
								y={y(t) + 3}
								text-anchor="end"
								class="num tick"
								fill="var(--ink-3)">{t.toFixed(1)}</text
							>
						{/each}
						{#if meanPts.length > 0}
							<path d={meanPath} fill="none" stroke="var(--warm)" stroke-width="1.6" />
						{/if}
						{#if legalPts.length > 1}
							<path d={legalPath} fill="none" stroke="var(--accent)" stroke-width="1.4" />
						{/if}
						{#each legalDots as p, i (i)}
							<circle cx={p.px} cy={p.py} r="2.2" fill="var(--accent)" />
						{/each}
						<text x={PAD.l} y={H - 5} class="num tick" fill="var(--ink-3)">
							{meanPts.length > 1 ? 'iterations →' : 'press Run RLVR'}
						</text>
					</svg>
					<p class="num flex flex-col gap-0.5 text-[10px] text-ink-3">
						<span
							><span style="color: var(--warm);">●</span> mean reward of the group · max 1.5</span
						>
						<span
							><span style="color: var(--accent);">●</span> legal-move probe · every 5 iterations</span
						>
					</p>

					<div class="mt-auto flex flex-col gap-2 border-t border-line-soft pt-3">
						<p class="num flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-3">
							<span>iter <span class="text-ink">{iter}</span></span>
							<span
								>reward <span class="text-ink">{meanNow === null ? '—' : meanNow.toFixed(2)}</span
								></span
							>
							<span
								>legal <span class="text-ink"
									>{legalNow === null ? '—' : `${(legalNow * 100).toFixed(0)}%`}</span
								></span
							>
							<span>updates <span class="text-ink">{updates}</span></span>
							<span>skipped <span class="text-ink">{skips}</span></span>
						</p>
						{#if skippedLast}
							<p class="text-[11px] leading-relaxed text-ink-2">
								All eight earned the same reward: the standard deviation is zero, the advantages are
								undefined and the step is skipped. When every answer is equally good there is no
								gradient. A lesson, not a bug.
							</p>
						{:else if reached}
							<p class="font-serif text-[12.5px] italic" style="color: var(--good);">
								The judge is running out of complaints.
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
				</section>
			</div>
		{:else}
			<BootRow />
		{/if}
	</div>
</Plate>

<style>
	/* Three ruled cells in one row: the position, the group, the run. The cells
	   stretch to a common height, so the table and the chart still line up with
	   the board even when they have less to say. */
	.col {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		border-color: var(--line-soft);
		padding-block: 0.75rem;
	}
	@media (max-width: 1023px) {
		.col + .col {
			border-top: 1px solid var(--line-soft);
		}
	}

	.tick {
		font-size: 10px;
	}

	/* one rollout per row: index, how far it held, what it earned, what the
	   group's mean made of that */
	.roll {
		display: grid;
		width: 100%;
		grid-template-columns: 1rem 2.6rem minmax(0, 1fr) 2.4rem 2.9rem;
		align-items: center;
		gap: 0.6rem;
		padding: 4.5px 4px;
		border-radius: 3px;
		font-size: 11px;
		text-align: left;
		cursor: default;
	}
	.roll:hover,
	.roll:focus-visible {
		background: color-mix(in srgb, var(--accent) 10%, transparent);
	}
	.on {
		background: color-mix(in srgb, var(--accent) 8%, transparent);
	}
	.track {
		height: 4px;
		overflow: hidden;
		border-radius: 999px;
		background: var(--surface-2);
	}
	.fill {
		display: block;
		height: 100%;
		border-radius: 999px;
	}

	/* where the group's own mean falls, drawn as a line rather than as signs */
	.baseline-rule {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin: 5px 0;
		color: var(--ink-3);
		font-size: 9.5px;
	}
	.baseline-rule::after {
		content: '';
		flex: 1;
		border-top: 1px dashed var(--line);
	}
</style>

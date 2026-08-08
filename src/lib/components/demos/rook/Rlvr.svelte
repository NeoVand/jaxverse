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
	import { pickPrefix, scoreRollout, type BoardSnap, type RolloutPly } from './chess-eval';
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

	/** Board caption for a rollout card. */
	function rolloutCaption(row: GroupRow): string {
		if (row.snap.kind === 'illegal') return `ply ${row.snap.ply} ✕ ${row.snap.uci}`;
		if (row.attempted === 0) return 'ended immediately';
		return `held all ${row.attempted}`;
	}

	// ── chart geometry ──
	const W = 560;
	const H = 170;
	const PAD = { l: 34, r: 10, t: 10, b: 20 };
	const chart = $derived.by(() => {
		if (meanPts.length === 0) return null;
		const maxIt = Math.max(30, iter);
		const minIt = meanPts[0].it;
		const x = scale(minIt, maxIt, PAD.l, W - PAD.r);
		const y = scale(0, 1.5, H - PAD.b, PAD.t);
		return {
			x,
			y,
			meanPath: polyline(meanPts.map((p) => [x(p.it), y(p.v)])),
			legalPath: polyline(legalPts.map((p) => [x(p.it), y(p.v)])),
			legalDots: legalPts.map((p) => ({ px: x(p.it), py: y(p.v) })),
			ticks: [0, 0.5, 1.0, 1.5].map((v) => ({ v, py: y(v) }))
		};
	});
	const meanNow = $derived(meanPts.length ? meanPts[meanPts.length - 1].v : null);
	const legalNow = $derived(legalPts.length ? legalPts[legalPts.length - 1].v : null);
</script>

<Plate
	id="rlvr"
	live
	title="RLVR — reinforcement from a judge"
	caption="No example ever said “this move is good”. A verifier said “this rollout held up longer”, the group's average set the bar, and the gradient did the rest. Each rollout's card shows the board where the judge stopped it — or the position it held to the end. This loop — sample, verify, standardize, reinforce — is a small, honest cousin of how frontier models learn to reason."
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
			<Btn kind="primary" onclick={() => void toggleRun()}>
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

			<div class="grid grid-cols-1 gap-x-8 gap-y-5 px-4 py-4 lg:grid-cols-[minmax(0,1fr)_230px]">
				<div class="min-w-0">
					{#if chart}
						<svg
							viewBox="0 0 {W} {H}"
							preserveAspectRatio="none"
							class="h-42 w-full"
							role="img"
							aria-label="Mean reward per iteration and the legal-move probe"
						>
							{#each chart.ticks as t (t.v)}
								<line
									x1={PAD.l}
									x2={W - PAD.r}
									y1={t.py}
									y2={t.py}
									stroke="var(--line-soft)"
									stroke-width="1"
									stroke-dasharray={t.v === 1 ? '3 4' : undefined}
								/>
								<text
									x={PAD.l - 6}
									y={t.py + 3}
									text-anchor="end"
									class="num"
									font-size="10"
									fill="var(--ink-3)">{t.v.toFixed(1)}</text
								>
							{/each}
							<path d={chart.meanPath} fill="none" stroke="var(--warm)" stroke-width="1.6" />
							{#if legalPts.length > 1}
								<path d={chart.legalPath} fill="none" stroke="var(--accent)" stroke-width="1.4" />
							{/if}
							{#each chart.legalDots as p, i (i)}
								<circle cx={p.px} cy={p.py} r="2.6" fill="var(--accent)" />
							{/each}
							<text x={PAD.l} y={H - 6} class="num" font-size="10" fill="var(--ink-3)"
								>iterations →</text
							>
						</svg>
						<p class="num mt-1 text-[11px] text-ink-3">
							<span style="color: var(--warm);">●</span> mean reward of the group (max 1.5) ·
							<span style="color: var(--accent);">●</span> legal-move probe (32 real positions), every
							5 iterations
						</p>
					{:else}
						<div class="flex h-40 items-center justify-center">
							<p class="max-w-md text-center text-[12.5px] text-ink-3">
								Press run. Fifty-odd iterations usually show the climb: the legal-move probe pushes
								past 95%, and the group's mean reward drifts up as more rollouts survive the judge.
							</p>
						</div>
					{/if}
				</div>

				<div class="num flex flex-col gap-2 text-[11.5px] text-ink-2">
					<span class="eyebrow">now</span>
					<span>iteration · {iter}</span>
					<span>mean reward · {meanNow === null ? '—' : meanNow.toFixed(2)}</span>
					<span
						>legal-move probe · {legalNow === null ? '—' : `${(legalNow * 100).toFixed(0)}%`}</span
					>
					<span>policy updates · {updates}</span>
					<span>skipped (no gradient) · {skips}</span>
					{#if reached}
						<span class="mt-1 font-serif text-[13px] italic" style="color: var(--good);">
							the judge is running out of complaints
						</span>
					{/if}
					{#if iter > 0}
						<a
							href="#rook-play"
							class="mt-1 inline-flex items-center gap-1.5 text-[11.5px] underline decoration-dotted underline-offset-4 hover:text-ink"
						>
							<ArrowUp size={11} aria-hidden="true" /> play the reinforced Rook — {plateLabel(
								'rook',
								'play'
							)} always plays the current weights
						</a>
					{/if}
				</div>
			</div>

			<div class="border-t border-line-soft px-4 py-4">
				<div class="flex flex-wrap items-baseline justify-between gap-2">
					<span class="eyebrow">group inspector — 8 rollouts of one position</span>
					{#if prefixShown.length > 0}
						<span class="num text-[10.5px] text-ink-3">prefix: ⟨game⟩ {prefixShown.join(' ')}</span>
					{/if}
				</div>

				{#if skippedLast}
					<div
						class="mt-2 rounded-md border border-line-soft bg-surface-2 px-3 py-2 text-[12px] text-ink-2"
					>
						All eight rollouts earned the same reward, so the group baseline leaves nothing to
						prefer — the standard deviation is zero and the step is skipped. When every answer is
						equally good, there is no gradient. A lesson, not a bug.
					</div>
				{/if}

				{#if group.length > 0}
					<div class="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
						{#each group as row, i (i)}
							<div class="flex gap-3 rounded-md border border-line-soft p-2.5">
								<MiniBoard
									fen={row.snap.fen}
									move={row.snap.uci}
									tone={row.snap.kind === 'illegal' ? 'bad' : 'accent'}
									caption={rolloutCaption(row)}
									size={118}
								/>
								<div class="flex min-w-0 flex-1 flex-col gap-1.5">
									<div class="flex items-center justify-between gap-2">
										<span class="num text-[10.5px] text-ink-3">rollout {i + 1}</span>
										{#if row.adv === null}
											<span class="num text-[11px] text-ink-3">Â —</span>
										{:else}
											<span
												class="num inline-block rounded px-1.5 py-0.5 text-[10.5px]"
												style="color: {row.adv >= 0
													? 'var(--good)'
													: 'var(--bad)'}; background: color-mix(in srgb, {row.adv >= 0
													? 'var(--good)'
													: 'var(--bad)'} 10%, transparent);"
											>
												Â {row.adv >= 0 ? '+' : '−'}{Math.abs(row.adv).toFixed(2)}
											</span>
										{/if}
									</div>
									<p class="num min-w-0 text-[10.5px] leading-snug break-words text-ink-2">
										{#each row.plies as p, pi (pi)}
											<span
												class="mr-1"
												class:text-bad={p.state === 'illegal'}
												class:text-ink-3={p.state === 'unchecked'}
												>{p.uci}{#if p.state === 'illegal'}✕{/if}</span
											>
										{:else}
											<span class="text-ink-3">(ended the game immediately)</span>
										{/each}
									</p>
									<div class="mt-auto flex items-center gap-2">
										<span class="num w-10 shrink-0 text-[10.5px] text-ink-2"
											>{row.legalPlies}/{row.attempted}</span
										>
										<span class="h-1 min-w-0 flex-1 overflow-hidden rounded-full bg-surface-2">
											<span
												class="block h-full rounded-full"
												style="width: {((row.reward / 1.5) * 100).toFixed(
													0
												)}%; background: var(--accent);"
											></span>
										</span>
										<span class="num w-9 shrink-0 text-right text-[10.5px]"
											>{row.reward.toFixed(2)}</span
										>
									</div>
								</div>
							</div>
						{/each}
					</div>
					<p class="num mt-2 text-[10.5px] text-ink-3">
						each card: the board where the rollout broke (✕, washed red) or its final held position
						· plies · legal / attempted · reward bar · advantage Â (greyed moves came after the
						first illegal one — unverifiable, not scored)
					</p>
				{:else}
					<p class="mt-2 text-[12.5px] text-ink-3">The first group will appear here.</p>
				{/if}
			</div>
		{:else}
			<BootRow />
		{/if}
	</div>
</Plate>

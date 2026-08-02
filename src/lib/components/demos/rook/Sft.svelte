<script lang="ts">
	// Plate III — supervised fine-tuning, honestly: the SAME weights, a better
	// corpus (2,381 greedy-material games), the same update rule. The headline
	// curve is taste (capture rate of sampled games); the quiet curve is the
	// price (val loss on the ORIGINAL random-play corpus, drifting up).
	import { ArrowUp, Pause, Play, Replace } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { progress } from '$lib/data/progress.svelte';
	import { lab, LR_SFT } from './rook-context.svelte';
	import {
		replayGames,
		snapshotGame,
		snapCaption,
		snapTone,
		type BoardSnap,
		type PlyMark
	} from './chess-eval';
	import { polyline, scale } from './chart';
	import Gauge from '$lib/components/ui/Gauge.svelte';
	import MiniBoard from './MiniBoard.svelte';
	import BootRow from './BootRow.svelte';

	interface Pt {
		step: number;
		v: number;
	}

	let running = $state(false);
	let evaling = $state(false);
	let switching = $state(false);
	let lossNow = $state(NaN);
	let sftSteps = $state(0);
	let capturePts = $state<Pt[]>([]);
	let valOrigPts = $state<Pt[]>([]);
	let legalNow = $state<number | null>(null);
	let captureNow = $state<number | null>(null);
	let snaps = $state<BoardSnap[]>([]);
	let games = $state<PlyMark[][]>([]);
	let reached = $state(false);

	const switched = $derived(lab.corpus === 'sft');
	const chunkSize = 25;

	async function evalNow(g: number): Promise<void> {
		if (!lab.engine || !lab.data) return;
		evaling = true;
		try {
			// honest drift metric: held-out loss against the corpus we LEFT
			const v = await lab.valLossOn('random');
			if (g !== lab.gen) return;
			valOrigPts = [...valOrigPts, { step: sftSteps, v }];
			// legality should HOLD where the model now lives: probe greedy-game
			// positions (drift from the OLD world is the val-loss line's job)
			const rate = await lab.probeLegal(g, 'sft');
			if (rate === null || g !== lab.gen) return;
			legalNow = rate;
			const sampled = await lab.sampleGames(3, g);
			if (!sampled || g !== lab.gen) return;
			const r = await replayGames(sampled, lab.data.decode);
			// photograph the freshest game as a chess-book strip
			const s = await snapshotGame(sampled[0], lab.data.decode);
			if (g !== lab.gen) return;
			games = r.games;
			snaps = s;
			captureNow = r.captureRate;
			capturePts = [...capturePts, { step: sftSteps, v: r.captureRate ?? 0 }];
			if ((r.captureRate ?? 0) >= 0.25 && !reached) {
				reached = true;
				progress.reach('rook:sft');
			}
		} catch {
			/* interrupted mid-eval */
		} finally {
			evaling = false;
		}
	}

	/** Swap the training corpus — weights and optimizer state survive. */
	async function switchDiet(): Promise<void> {
		if (lab.phase !== 'ready' || switching) return;
		switching = true;
		const g = await lab.beginLoop('sft');
		try {
			if (g !== lab.gen || !lab.engine) return;
			sftSteps = 0;
			capturePts = [];
			valOrigPts = [];
			await lab.useCorpus('sft');
			await lab.useLr(LR_SFT); // fine-tune gently — blasting erases the old competence
			if (g !== lab.gen) return;
			// baseline point BEFORE any fine-tuning step — both curves start honest
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
		const g = await lab.beginLoop('sft');
		if (g !== lab.gen || lab.phase !== 'ready') return;
		await lab.useCorpus('sft');
		await lab.useLr(LR_SFT);
		if (g !== lab.gen) return;
		// weights were rewound (or reinforced) since the last fine-tune —
		// the old curve would lie about these weights, so start it over
		if (lab.stage !== 'fine-tuned' && sftSteps > 0) {
			sftSteps = 0;
			capturePts = [];
			valOrigPts = [];
			await evalNow(g);
			if (g !== lab.gen) return;
		}
		running = true;
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
		// photograph the paused weights — the arena (Plate V) fields this student
		if (g === lab.gen && lab.stage === 'fine-tuned') await lab.captureStage('fine-tuned');
		lab.endLoop(g);
		running = false;
	}

	// ── chart geometry ──
	const W = 560;
	const H = 180;
	const PAD = { l: 38, r: 10, t: 12, b: 20 };
	const chart = $derived.by(() => {
		if (capturePts.length === 0) return null;
		const maxStep = Math.max(100, sftSteps);
		const x = scale(0, maxStep, PAD.l, W - PAD.r);
		const y = scale(0, 0.55, H - PAD.b, PAD.t);
		return {
			x,
			y,
			capPath: polyline(capturePts.map((p) => [x(p.step), y(p.v)])),
			pts: capturePts.map((p) => ({ px: x(p.step), py: y(p.v) })),
			ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5].map((v) => ({ v, py: y(v) }))
		};
	});
	const valDrift = $derived.by(() => {
		if (valOrigPts.length === 0) return null;
		return { first: valOrigPts[0].v, last: valOrigPts[valOrigPts.length - 1].v };
	});
	const sparkPath = $derived.by(() => {
		if (valOrigPts.length < 2) return '';
		let lo = Infinity;
		let hi = -Infinity;
		for (const p of valOrigPts) {
			lo = Math.min(lo, p.v);
			hi = Math.max(hi, p.v);
		}
		if (hi - lo < 1e-6) hi = lo + 1e-6;
		const x = scale(0, valOrigPts.length - 1, 0, 110);
		const y = scale(lo, hi, 26, 4);
		return polyline(valOrigPts.map((p, i) => [x(i), y(p.v)]));
	});
</script>

<Plate
	n={3}
	title="Fine-tuning — the same weights, a better diet"
	caption="Nothing about the machine changed — same parameters, same loss, same update rule — only the corpus. Style follows the diet: within a hundred-odd steps the capture rate of Rook's own games climbs from 8% toward the greedy corpus's 38%, while the quiet numbers show the price — rising loss on the abandoned random-play corpus. Stop while you are ahead: over-tuning keeps buying style and starts paying with competence."
>
	{#snippet status()}
		{#if lab.phase === 'ready'}
			<span>
				{lab.weightsLabel} · diet: {switched ? 'greedy games' : 'random play'}
				{#if switched}· {sftSteps} steps{/if}
				{#if running && Number.isFinite(lossNow)}· loss {lossNow.toFixed(2)}{/if}
				{#if evaling}· sampling…{:else if running}· fine-tuning{/if}
			</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		{#if lab.phase === 'ready'}
			{#if !switched}
				<Btn kind="primary" onclick={() => void switchDiet()} disabled={switching}>
					<Replace size={12} aria-hidden="true" />
					{switching ? 'Switching…' : 'Switch the diet'}
				</Btn>
			{:else}
				<Btn onclick={() => void toggleTrain()}>
					{#if running}
						<Pause size={12} aria-hidden="true" /> Pause
					{:else}
						<Play size={12} aria-hidden="true" /> Fine-tune
					{/if}
				</Btn>
			{/if}
		{/if}
	{/snippet}

	<div use:inview={() => void lab.power()}>
		{#if lab.phase === 'ready'}
			{#if switched && chart}
				<div class="grid grid-cols-1 gap-x-8 gap-y-5 px-4 py-4 lg:grid-cols-[minmax(0,1fr)_240px]">
					<div class="min-w-0">
						<svg
							viewBox="0 0 {W} {H}"
							preserveAspectRatio="none"
							class="h-44 w-full"
							role="img"
							aria-label="Capture rate of sampled games across fine-tuning steps"
						>
							{#each chart.ticks as t (t.v)}
								<line
									x1={PAD.l}
									x2={W - PAD.r}
									y1={t.py}
									y2={t.py}
									stroke="var(--line-soft)"
									stroke-width="1"
								/>
								<text
									x={PAD.l - 6}
									y={t.py + 3}
									text-anchor="end"
									class="num"
									font-size="10"
									fill="var(--ink-3)">{(t.v * 100).toFixed(0)}%</text
								>
							{/each}
							<!-- the two diets, as dashed reference lines -->
							<line
								x1={PAD.l}
								x2={W - PAD.r}
								y1={chart.y(0.08)}
								y2={chart.y(0.08)}
								stroke="var(--ink-3)"
								stroke-width="1"
								stroke-dasharray="3 4"
							/>
							<text
								x={W - PAD.r}
								y={chart.y(0.08) - 4}
								text-anchor="end"
								class="num"
								font-size="10"
								fill="var(--ink-3)">random-play diet ≈8%</text
							>
							<line
								x1={PAD.l}
								x2={W - PAD.r}
								y1={chart.y(0.38)}
								y2={chart.y(0.38)}
								stroke="var(--warm)"
								stroke-width="1"
								stroke-dasharray="3 4"
								opacity="0.55"
							/>
							<text
								x={W - PAD.r}
								y={chart.y(0.38) - 4}
								text-anchor="end"
								class="num"
								font-size="10"
								fill="var(--warm)">greedy diet ≈38%</text
							>
							<path d={chart.capPath} fill="none" stroke="var(--warm)" stroke-width="1.6" />
							{#each chart.pts as p, i (i)}
								<circle cx={p.px} cy={p.py} r="2.6" fill="var(--warm)" />
							{/each}
							<text x={PAD.l} y={H - 6} class="num" font-size="10" fill="var(--ink-3)"
								>fine-tuning steps →</text
							>
						</svg>
						<p class="num mt-1 text-[11px] text-ink-3">
							<span style="color: var(--warm);">●</span> capture rate of sampled games — the style the
							new diet teaches
						</p>
					</div>

					<div class="flex flex-col gap-4">
						<Gauge label="legal moves · 32 greedy-game positions" value={legalNow} />
						<Gauge label="captures per legal move" value={captureNow} tone="warm" />
						<div class="num text-[11px] text-ink-3">
							<span class="eyebrow">the price · drift</span>
							<div class="mt-1 flex items-center gap-2">
								{#if sparkPath}
									<svg
										width="110"
										height="30"
										aria-label="Validation loss on the original corpus"
										role="img"
									>
										<path d={sparkPath} fill="none" stroke="var(--ink-3)" stroke-width="1.3" />
									</svg>
								{/if}
								{#if valDrift}
									<span>
										val loss on random-play corpus<br />
										{valDrift.first.toFixed(2)} → {valDrift.last.toFixed(2)}
										{#if valDrift.last > valDrift.first + 0.005}(rising — it is forgetting its old
											diet){/if}
									</span>
								{/if}
							</div>
						</div>
						{#if reached}
							<p class="font-serif text-[13px] italic" style="color: var(--good);">
								the taste took — the capture rate crossed 25%
							</p>
						{/if}
						{#if sftSteps > 0}
							<a
								href="#rook-play"
								class="num inline-flex items-center gap-1.5 text-[11.5px] text-ink-2 underline decoration-dotted underline-offset-4 hover:text-ink"
							>
								<ArrowUp size={11} aria-hidden="true" /> play the fine-tuned Rook — Plate II always plays
								the current weights
							</a>
						{/if}
					</div>
				</div>

				<div class="flex min-w-0 flex-col gap-3 border-t border-line-soft px-4 py-4">
					<span class="eyebrow">the freshest sampled game, photographed · captures washed warm</span
					>
					{#if snaps.length === 0}
						<p class="py-1 text-[12.5px] text-ink-3">
							{evaling ? 'sampling…' : 'fine-tune to sample'}
						</p>
					{:else}
						<div class="flex flex-wrap items-start gap-3">
							{#each snaps as s (s.ply + s.kind)}
								<MiniBoard
									fen={s.fen}
									move={s.uci}
									tone={snapTone(s)}
									caption={snapCaption(s)}
									size={132}
								/>
							{/each}
						</div>
						{#each games as plies, gi (gi)}
							<p class="num flex flex-wrap gap-x-1.5 gap-y-0.5 text-[10px] leading-snug text-ink-3">
								<span>⟨game⟩</span>
								{#each plies as p, pi (pi)}
									{#if !p.legal}
										<span class="text-bad" title="illegal in the position where it was played"
											>{p.uci}✕</span
										>
									{:else if p.capture}
										<span style="color: var(--warm);" title="a capture">{p.uci}×</span>
									{:else}
										<span>{p.uci}</span>
									{/if}
								{/each}
							</p>
						{/each}
					{/if}
				</div>
			{:else if switched}
				<div class="px-4 py-6">
					<p class="text-[12.5px] text-ink-3">measuring the baseline…</p>
				</div>
			{:else}
				<div class="px-4 py-5">
					<p class="max-w-2xl text-[13px] text-ink-2">
						Rook currently imitates its pretraining diet: random legal chess, where only ≈8% of
						moves happen to be captures. The button above swaps in the curated corpus — same model,
						same objective, new examples — and the plate then measures what its <em>own</em> games look
						like as fine-tuning proceeds.
					</p>
				</div>
			{/if}
		{:else}
			<BootRow />
		{/if}
	</div>
</Plate>

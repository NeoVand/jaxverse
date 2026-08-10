<script lang="ts">
	// The arena. One position, every stage of Rook asked at once:
	// the frozen pretraining waypoint, the fine-tuned snapshot (photographed
	// when the fine-tuning plate pauses), and the reinforced snapshot (the RLVR plate). Each
	// answers with its full next-move distribution; the board shows every
	// decision as a colored arrow, and the stage you picked actually replies.
	// There is only one engine, so lab.compareStages swaps weights in place
	// and restores the resident set after every question.
	import { RotateCcw } from 'lucide-svelte';
	import type { Square } from 'chess.js';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { plateLabel } from '$lib/data/plates';
	import { lab, type ArenaStage } from './rook-context.svelte';
	import { analyzeRow, loadChess, type ChessGame, type RowVerdict } from './chess-eval';
	import Board, { type BoardArrow } from './Board.svelte';
	import type { BoardTone } from './board';
	import BootRow from './BootRow.svelte';

	const STAGES: Array<{
		id: ArenaStage;
		label: string;
		color: string;
		tone: BoardTone;
		hint: string;
	}> = [
		{
			id: 'pretrained',
			label: 'pretrained',
			color: 'var(--accent)',
			tone: 'accent',
			hint: 'the raw student — always fielded'
		},
		{
			id: 'fine-tuned',
			label: 'fine-tuned',
			color: 'var(--cat-8)',
			tone: 'violet',
			hint: `pause the fine-tuning in ${plateLabel('rook', 'sft')} to field it`
		},
		{
			id: 'reinforced',
			label: 'RLVR',
			color: 'var(--good)',
			tone: 'good',
			hint: `pause the RLVR run in ${plateLabel('rook', 'rlvr')} to field it`
		}
	];

	// chess.js isn't reactive; the template keys off the fen mirror. $state.raw
	// so the reference itself is reactive (it is passed to <Board>) without
	// proxying the game object.
	let chess = $state.raw<ChessGame | null>(null);
	let historyIds: number[] = [0];

	let boardReady = $state(false);
	let fen = $state('');
	let thinking = $state(false);
	let outcome = $state('');
	let lastMove = $state<{ from: string; to: string } | null>(null);
	let vocabMisses = $state(0);
	let boardErr = $state('');
	let verdicts = $state<Partial<Record<ArenaStage, RowVerdict>>>({});
	/** The position the current verdicts were measured on — arrows only make
	 * sense while the board still shows it. */
	let comparedFen = $state('');
	/** Set when a compare was cancelled mid-turn (another plate claimed the
	 * engine) — offers the reader a resume button instead of a wedged game. */
	let interrupted = $state(false);

	let opponent = $state<ArenaStage>('pretrained');
	let opponentTouched = false;

	const fielded = $derived(
		STAGES.filter((s) => s.id === 'pretrained' || lab.hasSnapshot[s.id]).map((s) => s.id)
	);

	// until the reader chooses, Black is played by the most-trained Rook
	$effect(() => {
		if (opponentTouched) return;
		opponent = lab.hasSnapshot.reinforced
			? 'reinforced'
			: lab.hasSnapshot['fine-tuned']
				? 'fine-tuned'
				: 'pretrained';
	});

	let initing = false;
	$effect(() => {
		if (lab.phase === 'ready' && !boardReady && !initing) {
			initing = true;
			void initBoard();
		}
	});

	async function initBoard(): Promise<void> {
		try {
			const Chess = await loadChess();
			chess = new Chess();
			resetGame();
			boardReady = true;
		} catch (e) {
			boardErr = e instanceof Error ? e.message : String(e);
		}
	}

	function resetGame(): void {
		if (!chess) return;
		chess.reset();
		fen = chess.fen();
		historyIds = [0];
		outcome = '';
		lastMove = null;
		verdicts = {};
		comparedFen = '';
		interrupted = false;
		vocabMisses = 0;
	}

	// ── interaction: the board handles picking a piece up; this decides ──
	const yourTurn = $derived.by(() => {
		void fen;
		return Boolean(chess) && !thinking && !outcome && !interrupted && chess?.turn() === 'w';
	});

	function playUser(from: string, to: string): boolean {
		if (!chess || !yourTurn) return false;
		let mv;
		try {
			mv = chess.move({ from: from as Square, to: to as Square, promotion: 'q' });
		} catch {
			return false; // not a legal move here — the board keeps the piece
		}
		fen = chess.fen();
		lastMove = { from: mv.from, to: mv.to };
		pushMove(mv.from + mv.to + (mv.promotion ?? ''));
		if (!checkOver()) void stageReply();
		return true;
	}

	function pushMove(uci: string): void {
		const id = lab.data?.idOf.get(uci);
		// A legal move can be absent from the vocabulary (never seen in the
		// training games) — skip it in the models' context rather than crash.
		if (id !== undefined) historyIds.push(id);
		else vocabMisses++;
	}

	function checkOver(): boolean {
		if (!chess?.isGameOver()) return false;
		outcome = chess.isCheckmate()
			? chess.turn() === 'b'
				? 'Checkmate — you win.'
				: 'Checkmate — Rook wins.'
			: chess.isStalemate()
				? 'Stalemate.'
				: 'Draw.';
		return true;
	}

	/** Ask every fielded stage, show the arrows for a beat, then let the
	 * chosen opponent's masked argmax actually answer on the board. */
	async function stageReply(): Promise<void> {
		if (!lab.engine || !lab.data || !chess) return;
		thinking = true;
		interrupted = false;
		try {
			const rows = await lab.compareStages(historyIds, fielded);
			if (!rows || !chess) {
				interrupted = true;
				return;
			}
			const legal = chess.moves({ verbose: true }).map((m) => m.from + m.to + (m.promotion ?? ''));
			const v: Partial<Record<ArenaStage, RowVerdict>> = {};
			for (const stage of fielded) {
				const row = rows[stage];
				if (row) v[stage] = analyzeRow(row, legal, lab.data.idOf, lab.data.vocab.moves, 5);
			}
			verdicts = v;
			comparedFen = fen;
			// a beat with every stage's arrow on the board, then the reply
			await new Promise((r) => setTimeout(r, 1100));
			const pick = v[opponent]?.pick;
			if (!pick) {
				interrupted = true;
				return;
			}
			const mv = chess.move({
				from: pick.slice(0, 2),
				to: pick.slice(2, 4),
				promotion: pick.length > 4 ? pick[4] : 'q'
			});
			fen = chess.fen();
			lastMove = { from: mv.from, to: mv.to };
			pushMove(pick);
			checkOver();
		} catch (e) {
			boardErr = e instanceof Error ? e.message : String(e);
		} finally {
			thinking = false;
		}
	}

	const arrows = $derived.by((): BoardArrow[] => {
		if (fen !== comparedFen) return [];
		const out: BoardArrow[] = [];
		for (const s of STAGES) {
			const pick = verdicts[s.id]?.pick;
			if (pick) out.push({ from: pick.slice(0, 2), to: pick.slice(2, 4), tone: s.tone });
		}
		return out;
	});

	function pickOpponent(id: ArenaStage): void {
		opponentTouched = true;
		opponent = id;
	}
</script>

<Plate
	id="arena"
	live
	title="The arena — every stage, same position"
	caption="Same board, three students. After each of your moves, every fielded Rook is asked for its next-move beliefs — the weights are swapped in place on the one engine, questioned, and restored — and each decision lands on the board as an arrow: blue pretrained, violet fine-tuned, green RLVR. The columns keep score honestly: legal mass climbs stage by stage, and the fine-tuned Rook starts hunting your pieces. The stage you pick under “plays Black” is the one that actually answers."
>
	{#snippet status()}
		{#if lab.phase === 'ready'}
			<span>
				{fielded.length} of 3 fielded
				{#if thinking}· asking every stage…{/if}
			</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		{#if lab.phase === 'ready' && boardReady}
			<Btn onclick={resetGame}><RotateCcw size={12} aria-hidden="true" /> New game</Btn>
		{/if}
	{/snippet}

	<div use:inview={() => void lab.power()}>
		{#if lab.phase === 'ready' && boardReady}
			<div
				class="grid px-4 pt-1 pb-2 lg:grid-cols-[360px_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)]"
			>
				<section class="col lg:pr-6">
					<Board
						{chess}
						{fen}
						{lastMove}
						{arrows}
						input={yourTurn ? 'w' : null}
						onMove={playUser}
						ariaLabel="Chess board — you play White; every stage's chosen reply appears as an arrow"
					/>
					<div class="flex flex-wrap items-center gap-x-3 gap-y-2">
						<span class="eyebrow">plays Black</span>
						<span class="seg" role="group" aria-label="Which stage answers">
							{#each STAGES as s (s.id)}
								<button
									class:on={opponent === s.id}
									disabled={!fielded.includes(s.id)}
									onclick={() => pickOpponent(s.id)}
									title={fielded.includes(s.id) ? undefined : s.hint}
								>
									{s.label}
								</button>
							{/each}
						</span>
					</div>
					<div class="mt-auto flex min-h-6 flex-wrap items-center gap-3">
						{#if thinking}
							<span class="eyebrow">asking every stage…</span>
						{:else if interrupted && !outcome}
							<Btn onclick={() => void stageReply()}>Resume — let Black move</Btn>
						{:else if outcome}
							<span class="font-serif text-[13.5px] text-ink italic">{outcome}</span>
						{:else}
							<span class="num text-[11px] text-ink-3">move, and all three answer at once</span>
						{/if}
						{#if boardErr}
							<span class="text-[11.5px] text-bad">{boardErr}</span>
						{/if}
						{#if vocabMisses > 0}
							<span class="text-[11px] text-ink-3">
								{vocabMisses}
								{vocabMisses === 1 ? 'move' : 'moves'} here never occurred in the training corpus — skipped
								in the models' context.
							</span>
						{/if}
					</div>
				</section>

				<!-- one column per student, the same rows in each: the comparison IS
				     the plate, so the three read side by side -->
				{#each STAGES as s, i (s.id)}
					{@const v = verdicts[s.id]}
					{@const here = fielded.includes(s.id)}
					<section class="col lg:border-l {i === STAGES.length - 1 ? 'lg:pl-6' : 'lg:px-6'}">
						<span class="num flex items-center gap-1.5 text-[11.5px] font-medium text-ink">
							<span
								class="inline-block h-2 w-2 rounded-full"
								style="background: {s.color};"
								aria-hidden="true"
							></span>
							{s.label}
							{#if opponent === s.id}<span class="eyebrow ml-1">plays black</span>{/if}
						</span>

						{#if !here}
							<p class="text-[11.5px] leading-relaxed text-ink-3">{s.hint}</p>
						{:else if !v}
							<p class="text-[11.5px] text-ink-3">make a move — it will answer here</p>
						{:else}
							<div class="flex flex-col gap-1">
								<span class="eyebrow">it would play</span>
								<span class="num text-[15px]" style="color: {s.color};">{v.pick ?? '—'}</span>
								<span class="num text-[10.5px] text-ink-3">
									{(v.pickP * 100).toFixed(1)}% of its belief
								</span>
							</div>

							<div class="flex flex-col gap-1">
								<div class="flex items-baseline justify-between gap-2">
									<span class="eyebrow">legal mass</span>
									<span class="num text-[11px] text-ink">{(v.legalMass * 100).toFixed(0)}%</span>
								</div>
								<span class="track">
									<span
										class="fill"
										style="width: {(Math.min(v.legalMass, 1) * 100).toFixed(
											0
										)}%; background: {s.color};"
									></span>
								</span>
							</div>

							<div class="flex flex-col gap-1">
								<span class="eyebrow">raw top-5 — before the mask</span>
								{#each v.top as t (t.uci)}
									<span class="num flex items-baseline gap-2 text-[10.5px]">
										<span class="w-11 text-ink-2">{t.uci}</span>
										<span class="w-9 text-right text-ink-3">{(t.p * 100).toFixed(1)}%</span>
										<span style="color: {t.legal ? 'var(--good)' : 'var(--bad)'};"
											>{t.legal ? '✓' : '✕'}</span
										>
									</span>
								{/each}
							</div>
						{/if}
					</section>
				{/each}
			</div>
		{:else}
			<BootRow />
		{/if}
	</div>
</Plate>

<style>
	/* Four ruled cells: the board, then one per student. The cells stretch to a
	   common height, so three stages that have said different amounts still line
	   up row for row. */
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
	.track {
		display: block;
		height: 4px;
		max-width: 11rem;
		overflow: hidden;
		border-radius: 999px;
		background: var(--surface-2);
	}
	.fill {
		display: block;
		height: 100%;
		border-radius: 999px;
		transition: width 300ms ease;
	}
</style>

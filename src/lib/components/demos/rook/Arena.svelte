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
	import BootRow from './BootRow.svelte';

	const STAGES: Array<{ id: ArenaStage; label: string; color: string; hint: string }> = [
		{
			id: 'pretrained',
			label: 'pretrained',
			color: 'var(--accent)',
			hint: 'the raw student — always fielded'
		},
		{
			id: 'fine-tuned',
			label: 'fine-tuned',
			color: 'var(--cat-8)',
			hint: `pause the fine-tuning in ${plateLabel('rook', 'sft')} to field it`
		},
		{
			id: 'reinforced',
			label: 'RLVR',
			color: 'var(--good)',
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
	let selected = $state<Square | null>(null);
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
		selected = null;
		lastMove = null;
		verdicts = {};
		comparedFen = '';
		interrupted = false;
		vocabMisses = 0;
	}

	// ── interaction ──
	function tap(sq: string): void {
		if (!chess || thinking || outcome || interrupted || chess.turn() !== 'w') return;
		if (selected && chess.moves({ square: selected, verbose: true }).some((m) => m.to === sq)) {
			playUser(selected, sq as Square);
			return;
		}
		const piece = chess.get(sq as Square);
		if (piece && piece.color === 'w') {
			selected = selected === sq ? null : (sq as Square);
			return;
		}
		selected = null;
	}

	function playUser(from: Square, to: Square): void {
		if (!chess) return;
		const mv = chess.move({ from, to, promotion: 'q' });
		fen = chess.fen();
		selected = null;
		lastMove = { from: mv.from, to: mv.to };
		pushMove(mv.from + mv.to + (mv.promotion ?? ''));
		if (!checkOver()) void stageReply();
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
				if (row) v[stage] = analyzeRow(row, legal, lab.data.idOf, lab.data.vocab.moves);
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
			if (pick) out.push({ from: pick.slice(0, 2), to: pick.slice(2, 4), color: s.color });
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
				class="grid grid-cols-1 items-start gap-x-8 gap-y-5 px-4 py-4 md:grid-cols-[minmax(0,360px)_minmax(0,1fr)]"
			>
				<div>
					<div class="max-w-[360px]">
						<Board
							{chess}
							{fen}
							{selected}
							{lastMove}
							{arrows}
							onTap={tap}
							ariaLabel="Chess board — you play White; every stage's chosen reply appears as an arrow"
						/>
					</div>
					<div class="mt-2.5 flex min-h-7 flex-wrap items-center gap-3">
						{#if thinking}
							<span class="eyebrow">asking every stage…</span>
						{:else if interrupted && !outcome}
							<Btn onclick={() => void stageReply()}>Resume — let Black move</Btn>
						{:else if outcome}
							<span class="font-serif text-[13.5px] text-ink italic">{outcome}</span>
						{/if}
						{#if boardErr}
							<span class="text-[11.5px] text-bad">{boardErr}</span>
						{/if}
					</div>
				</div>

				<div class="flex min-w-0 flex-col gap-3">
					<div class="flex flex-wrap items-center gap-2">
						<span class="eyebrow">plays Black</span>
						{#each STAGES as s (s.id)}
							<Btn
								pressed={opponent === s.id}
								disabled={!fielded.includes(s.id)}
								onclick={() => pickOpponent(s.id)}
								title={fielded.includes(s.id) ? undefined : s.hint}
							>
								<span
									class="inline-block h-2 w-2 rounded-full"
									style="background: {s.color};"
									aria-hidden="true"
								></span>
								{s.label}
							</Btn>
						{/each}
					</div>

					{#if Object.keys(verdicts).length > 0}
						<span class="eyebrow"
							>{fen === comparedFen
								? 'this position, stage by stage'
								: 'the last decision, stage by stage'}</span
						>
					{/if}
					{#each STAGES as s (s.id)}
						{@const v = verdicts[s.id]}
						{@const here = fielded.includes(s.id)}
						<div
							class="rounded-md border border-line-soft px-3 py-2.5"
							style={here ? '' : 'opacity: 0.55;'}
						>
							<div class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
								<span class="num flex items-center gap-1.5 text-[11px] font-medium text-ink">
									<span
										class="inline-block h-2 w-2 rounded-full"
										style="background: {s.color};"
										aria-hidden="true"
									></span>
									{s.label}
								</span>
								{#if !here}
									<span class="text-[11px] text-ink-3">{s.hint}</span>
								{:else if v?.pick}
									<span class="num text-[11.5px]" style="color: {s.color};"
										>plays {v.pick}
										<span class="text-ink-3">· {(v.pickP * 100).toFixed(1)}% of its belief</span
										></span
									>
								{:else}
									<span class="text-[11px] text-ink-3">make a move — it will answer here</span>
								{/if}
							</div>
							{#if here && v}
								<div class="mt-2 flex items-center gap-2">
									<span class="eyebrow shrink-0">legal mass</span>
									<div class="h-1 min-w-0 flex-1 overflow-hidden rounded-full bg-surface-2">
										<div
											class="h-full rounded-full"
											style="width: {(Math.min(v.legalMass, 1) * 100).toFixed(
												0
											)}%; background: {s.color}; transition: width 300ms ease;"
										></div>
									</div>
									<span class="num w-10 shrink-0 text-right text-[11px] text-ink"
										>{(v.legalMass * 100).toFixed(0)}%</span
									>
								</div>
								<div class="num mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[10.5px]">
									<span class="text-ink-3">raw top-3:</span>
									{#each v.top as t (t.uci)}
										<span style="color: {t.legal ? 'var(--good)' : 'var(--bad)'};"
											>{t.uci} {(t.p * 100).toFixed(1)}%{t.legal ? '' : ' ✕'}</span
										>
									{/each}
								</div>
							{/if}
						</div>
					{/each}

					{#if vocabMisses > 0}
						<p class="text-[11px] text-ink-3">
							{vocabMisses}
							{vocabMisses === 1 ? 'move' : 'moves'} in this game never occurred in the training corpus
							— skipped in the models' context, noted here honestly.
						</p>
					{/if}
				</div>
			</div>
		{:else}
			<BootRow />
		{/if}
	</div>
</Plate>

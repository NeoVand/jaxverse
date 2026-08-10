<script lang="ts">
	// Play the language model. The board is chess.js; Rook's reply is
	// a masked sample from its next-token distribution (the ChessLab port from
	// LLMVibes): exp the log-probs, renormalize over the legal moves, draw by
	// inverse CDF, fall back to uniform when the model puts ~no mass on
	// legality. This plate always plays the CURRENT shared weights — the stage
	// badge says which act they last came through.
	import { RotateCcw } from 'lucide-svelte';
	import type { Square } from 'chess.js';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { lab } from './rook-context.svelte';
	import { loadChess, type ChessGame } from './chess-eval';
	import Board from './Board.svelte';
	import Gauge from '$lib/components/ui/Gauge.svelte';
	import BootRow from './BootRow.svelte';

	// chess.js isn't reactive; the template keys off the fen mirror. $state.raw
	// so the reference itself is reactive (it is passed to <Board>) without
	// proxying the game object.
	let chess = $state.raw<ChessGame | null>(null);
	// Rook's context — token 0 opens the game. $state.raw, not $state: the array
	// is posted to the worker, and a reactive proxy is not structured-cloneable.
	let historyIds = $state.raw<number[]>([0]);

	let boardReady = $state(false);
	let fen = $state('');
	let thinking = $state(false);
	let outcome = $state('');
	let lastMove = $state<{ from: string; to: string } | null>(null);
	let hoverSq = $state<{ from: string; to: string } | null>(null);
	let historyUci = $state<string[]>([]);
	let vocabMisses = $state(0);
	let legalMass = $state<number | null>(null);
	let topRaw = $state<Array<{ uci: string; p: number; legal: boolean }>>([]);
	let boardErr = $state('');

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
		historyUci = [];
		outcome = '';
		lastMove = null;
		hoverSq = null;
		legalMass = null;
		topRaw = [];
		vocabMisses = 0;
	}

	const movePairs = $derived.by(() => {
		const pairs: Array<{ w: string; b: string }> = [];
		for (let i = 0; i < historyUci.length; i += 2) {
			pairs.push({ w: historyUci[i], b: historyUci[i + 1] ?? '' });
		}
		return pairs;
	});

	const stageBadge = $derived(
		lab.stage === 'fine-tuned'
			? { label: 'fine-tuned weights', color: 'var(--cat-8)' }
			: lab.stage === 'reinforced'
				? { label: 'RLVR weights', color: 'var(--good)' }
				: { label: 'pretrained weights', color: 'var(--accent)' }
	);

	// ── interaction: the board handles picking a piece up; this decides ──
	const yourTurn = $derived.by(() => {
		void fen;
		return Boolean(chess) && !thinking && !outcome && chess?.turn() === 'w';
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
		if (!checkOver()) void rookReply();
		return true;
	}

	function pushMove(uci: string): void {
		historyUci = [...historyUci, uci];
		const id = lab.data?.idOf.get(uci);
		// A legal move can be absent from the vocabulary (never seen in the
		// training games) — skip it in the model's context rather than crash.
		if (id !== undefined) historyIds = [...historyIds, id];
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

	function hoverPly(uci: string | null): void {
		hoverSq = uci ? { from: uci.slice(0, 2), to: uci.slice(2, 4) } : null;
	}

	/** Rook's move: full next-token distribution, masked to the legal set. */
	async function rookReply(): Promise<void> {
		if (!lab.engine || !lab.data || !chess) return;
		thinking = true;
		try {
			// the arena (the arena) swaps weights in place while it compares stages —
			// wait until the resident set is back, or "the current weights" would
			// briefly be some contestant's
			while (lab.busy === 'arena') await new Promise((r) => setTimeout(r, 60));
			const row = await lab.engine.nextDistribution(historyIds);
			const moves = lab.data.vocab.moves;
			const idOf = lab.data.idOf;
			const legal = chess.moves({ verbose: true });
			const legalIds = legal.map((m) => {
				const uci = m.from + m.to + (m.promotion ?? '');
				return { uci, id: idOf.get(uci) };
			});
			// probabilities from log-probs
			const probs = new Float64Array(row.length);
			let total = 0;
			for (let i = 0; i < row.length; i++) {
				probs[i] = Math.exp(row[i]);
				total += probs[i];
			}
			// raw top-5 for the panel — what Rook wanted before the mask
			const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
			const legalSet = new Set(legalIds.filter((l) => l.id !== undefined).map((l) => l.id));
			topRaw = order.slice(0, 8).map((i) => ({
				uci: i === 0 ? '⟨game⟩' : moves[i - 1],
				p: probs[i] / total,
				legal: legalSet.has(i)
			}));
			// legal mass + masked inverse-CDF sampling
			let mass = 0;
			const cands: Array<{ uci: string; p: number }> = [];
			for (const l of legalIds) {
				const p = l.id !== undefined ? probs[l.id] / total : 0;
				mass += p;
				cands.push({ uci: l.uci, p });
			}
			legalMass = mass;
			let pick: string;
			if (mass > 1e-9) {
				let r = Math.random() * mass;
				pick = cands[cands.length - 1].uci;
				for (const c of cands) {
					r -= c.p;
					if (r <= 0) {
						pick = c.uci;
						break;
					}
				}
			} else {
				// the model put ~nothing on legality — fall back to a uniform draw
				pick = cands[Math.floor(Math.random() * cands.length)].uci;
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

	const topP = $derived(topRaw.length ? Math.max(...topRaw.map((t) => t.p), 1e-9) : 1);
</script>

<div id="rook-play">
	<Plate
		id="play"
		live
		title="Play it — you are White"
		caption="You are playing a language model: every reply is a sampled sentence-continuation, masked to legality. The gauge is the honest score — how much probability Rook put on legal moves before the mask saved it. And this board always plays the CURRENT weights: fine-tune or reinforce below, come back, and feel the difference."
	>
		{#snippet status()}
			{#if lab.phase === 'ready'}
				<span>
					{lab.weightsLabel} ·
					<span style="color: {stageBadge.color};">{lab.stage}</span>
					{#if thinking}· thinking…{/if}
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
				<div class="grid px-4 pt-1 pb-2 lg:grid-cols-[400px_minmax(0,1fr)_minmax(0,1fr)]">
					<section class="col lg:pr-7">
						<Board
							{chess}
							{fen}
							{lastMove}
							{hoverSq}
							input={yourTurn ? 'w' : null}
							onMove={playUser}
						/>
						<div class="flex min-h-6 flex-wrap items-center gap-3">
							{#if thinking}
								<span class="eyebrow">Rook is thinking…</span>
							{:else if outcome}
								<span class="font-serif text-[13.5px] text-ink italic">{outcome}</span>
							{:else}
								<span class="num text-[11px] text-ink-3">drag or click a white piece to move</span>
							{/if}
							{#if boardErr}
								<span class="text-[11.5px] text-bad">{boardErr}</span>
							{/if}
						</div>
					</section>

					<!-- what it believed, before legality was imposed on it -->
					<section class="col lg:border-l lg:px-7">
						<Gauge label="how much of Rook's belief was legal" value={legalMass} />
						<span class="eyebrow">raw top-8 — before the mask</span>
						<div class="flex flex-col gap-1">
							{#each topRaw as t (t.uci)}
								<div class="num belief">
									<span class="text-ink">{t.uci}</span>
									<span class="track">
										<span
											class="fill"
											style="width: {Math.min((t.p / topP) * 100, 100).toFixed(
												0
											)}%; background: {t.legal ? 'var(--good)' : 'var(--bad)'};"
										></span>
									</span>
									<span class="text-right text-ink-3">{(t.p * 100).toFixed(1)}%</span>
									<span
										style="color: {t.legal ? 'var(--good)' : 'var(--bad)'};"
										title={t.legal ? 'legal in this position' : 'illegal in this position'}
									>
										{t.legal ? '✓' : '✕'}
									</span>
								</div>
							{:else}
								<p class="text-[12px] text-ink-3">Make a move — Rook's raw beliefs appear here.</p>
							{/each}
						</div>
						<p class="mt-auto text-[10.5px] leading-snug text-ink-3">
							Bars are drawn against its own favourite, not against 100% — the whole distribution is
							this flat. ✕ marks a move that is illegal here, which the mask then throws away.
						</p>
					</section>

					<!-- and what has actually been played -->
					<section class="col lg:border-l lg:pl-7">
						<span class="eyebrow">moves — point at one to see it</span>
						<div class="num max-h-[19rem] overflow-y-auto text-[11px]">
							{#each movePairs as pr, i (i)}
								<div class="sheet">
									<span class="text-ink-3">{i + 1}.</span>
									<button
										type="button"
										class="ply"
										onmouseenter={() => hoverPly(pr.w)}
										onmouseleave={() => hoverPly(null)}
										onfocus={() => hoverPly(pr.w)}
										onblur={() => hoverPly(null)}>{pr.w}</button
									>
									{#if pr.b}
										<button
											type="button"
											class="ply"
											onmouseenter={() => hoverPly(pr.b)}
											onmouseleave={() => hoverPly(null)}
											onfocus={() => hoverPly(pr.b)}
											onblur={() => hoverPly(null)}>{pr.b}</button
										>
									{/if}
								</div>
							{:else}
								<p class="text-[12px] text-ink-3">Your move — you are White.</p>
							{/each}
						</div>
						<!-- the chapter's thesis, where the reader is playing: the model gets
						     this row of numbers and nothing else -->
						<div class="mt-auto flex flex-col gap-1.5 border-t border-line-soft pt-3.5">
							<span class="eyebrow">what Rook actually reads · {historyIds.length} tokens</span>
							<p
								class="num flex flex-wrap gap-x-1.5 gap-y-0.5 text-[10.5px] leading-snug text-ink-3"
							>
								{#each historyIds as id, i (i)}
									<span class={id === 0 ? 'text-ink-2' : ''}
										>{id === 0 ? '\u27e8game\u27e9' : id}</span
									>
								{/each}
							</p>
							<p class="text-[10.5px] leading-snug text-ink-3">
								No board ever reaches the model — only this row of numbers.
							</p>
						</div>

						<div class="flex flex-col gap-2">
							{#if vocabMisses > 0}
								<p class="text-[11px] leading-snug text-ink-3">
									{vocabMisses}
									{vocabMisses === 1 ? 'move' : 'moves'} here never occurred in Rook's training corpus,
									so no token exists for {vocabMisses === 1 ? 'it' : 'them'} — skipped in its context,
									noted honestly.
								</p>
							{/if}
							{#if lab.busy !== ''}
								<p class="text-[11px] leading-snug text-ink-3">
									The same weights are training in another plate right now — you are playing a
									moving target.
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
</div>

<style>
	/* Three ruled cells: the board, what it believed, what has been played. The
	   cells stretch to a common height and the hairlines run their full length,
	   so a column that says less still sits in the same frame. */
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

	/* the belief table: a token, a short bar, its number, its verdict — the bar
	   is capped so a 7% favourite cannot draw itself across the plate */
	.belief {
		display: grid;
		grid-template-columns: 3.4rem minmax(0, 1fr) 2.8rem 0.9rem;
		align-items: center;
		gap: 0.5rem;
		font-size: 11px;
	}
	.track {
		height: 4px;
		max-width: 6.5rem;
		overflow: hidden;
		border-radius: 999px;
		background: var(--surface-2);
	}
	.fill {
		display: block;
		height: 100%;
		border-radius: 999px;
	}

	/* the score sheet: number, White, Black */
	.sheet {
		display: grid;
		grid-template-columns: 1.6rem 4.4rem 4.4rem;
		align-items: baseline;
		gap: 0.25rem;
		color: var(--ink-2);
	}

	.ply {
		padding: 0 1px;
		border-radius: 3px;
		color: inherit;
		cursor: default;
	}
	.ply:hover,
	.ply:focus-visible {
		color: var(--ink);
		background: color-mix(in srgb, var(--accent) 12%, transparent);
	}
</style>

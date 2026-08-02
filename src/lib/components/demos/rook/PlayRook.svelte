<script lang="ts">
	// Plate II — play the language model. The board is chess.js; Rook's reply is
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
	let historyIds: number[] = [0]; // Rook's context — token 0 opens the game

	let boardReady = $state(false);
	let fen = $state('');
	let thinking = $state(false);
	let outcome = $state('');
	let selected = $state<Square | null>(null);
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
		selected = null;
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
			? { label: 'fine-tuned weights', color: 'var(--warm)' }
			: lab.stage === 'reinforced'
				? { label: 'RLVR weights', color: 'var(--good)' }
				: { label: 'pretrained weights', color: 'var(--accent)' }
	);

	// ── interaction ──
	function tap(sq: string): void {
		if (!chess || thinking || outcome || chess.turn() !== 'w') return;
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
		if (!checkOver()) void rookReply();
	}

	function pushMove(uci: string): void {
		historyUci = [...historyUci, uci];
		const id = lab.data?.idOf.get(uci);
		// A legal move can be absent from the vocabulary (never seen in the
		// training games) — skip it in the model's context rather than crash.
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

	function hoverPly(uci: string | null): void {
		hoverSq = uci ? { from: uci.slice(0, 2), to: uci.slice(2, 4) } : null;
	}

	/** Rook's move: full next-token distribution, masked to the legal set. */
	async function rookReply(): Promise<void> {
		if (!lab.engine || !lab.data || !chess) return;
		thinking = true;
		try {
			// the arena (Plate V) swaps weights in place while it compares stages —
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
			topRaw = order.slice(0, 5).map((i) => ({
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
		n={2}
		title="Play it — you are White"
		caption="You are playing a language model: every reply is a sampled sentence-continuation, masked to legality. The gauge is the honest score — how much probability Rook put on legal moves before the mask saved it. And this board always plays the CURRENT weights: fine-tune or reinforce below, come back, and feel the difference."
	>
		{#snippet status()}
			{#if lab.phase === 'ready'}
				<span>
					{lab.weightsLabel} · {lab.stage}
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
				<div
					class="grid grid-cols-1 items-start gap-x-8 gap-y-5 px-4 py-4 md:grid-cols-[minmax(0,360px)_minmax(0,1fr)]"
				>
					<div>
						<div class="max-w-[360px]">
							<Board {chess} {fen} {selected} {lastMove} {hoverSq} onTap={tap} />
						</div>
						<div class="mt-2.5 flex min-h-7 flex-wrap items-center gap-3">
							{#if thinking}
								<span class="eyebrow">Rook is thinking…</span>
							{:else if outcome}
								<span class="font-serif text-[13.5px] text-ink italic">{outcome}</span>
							{/if}
							{#if boardErr}
								<span class="text-[11.5px] text-bad">{boardErr}</span>
							{/if}
						</div>
					</div>

					<div class="flex min-w-0 flex-col gap-4">
						<div class="flex flex-wrap items-center gap-2">
							<span class="eyebrow">playing</span>
							<span
								class="num rounded px-1.5 py-0.5 text-[11px]"
								style="color: {stageBadge.color}; background: color-mix(in srgb, {stageBadge.color} 10%, transparent);"
							>
								{stageBadge.label} · {lab.weightsLabel}
							</span>
						</div>
						<Gauge label="how much of Rook's belief was legal" value={legalMass} />
						{#if topRaw.length > 0}
							<div>
								<span class="eyebrow">raw top-5 for this position — before the mask</span>
								<div class="mt-1.5 flex flex-col gap-1">
									{#each topRaw as t (t.uci)}
										<div class="num flex items-center gap-2 text-[11px]">
											<span class="w-14 shrink-0 text-ink">{t.uci}</span>
											<div class="h-1 min-w-0 flex-1 overflow-hidden rounded-full bg-surface-2">
												<div
													class="h-full rounded-full"
													style="width: {Math.min((t.p / topP) * 100, 100).toFixed(
														0
													)}%; background: {t.legal ? 'var(--good)' : 'var(--bad)'};"
												></div>
											</div>
											<span class="w-11 shrink-0 text-right text-ink-3"
												>{(t.p * 100).toFixed(1)}%</span
											>
											<span class="w-28 shrink-0 text-[10.5px]">
												{#if t.legal}
													<span style="color: var(--good);">✓ legal here</span>
												{:else}
													<span class="text-bad">✕ would be illegal</span>
												{/if}
											</span>
										</div>
									{/each}
								</div>
							</div>
						{:else}
							<p class="text-[12.5px] text-ink-3">
								Make a move — Rook's raw next-move beliefs will appear here.
							</p>
						{/if}
						{#if historyUci.length > 0}
							<div class="min-w-0">
								<span class="eyebrow">moves — hover one to see it on the board</span>
								<div
									class="num mt-1 max-h-36 overflow-y-auto text-[11px] leading-relaxed text-ink-2"
								>
									{#each movePairs as pr, i (i)}
										<span class="mr-3 inline-block whitespace-nowrap">
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
										</span>
									{/each}
								</div>
							</div>
						{/if}
						{#if vocabMisses > 0}
							<p class="text-[11px] text-ink-3">
								{vocabMisses}
								{vocabMisses === 1 ? 'move' : 'moves'} in this game never occurred in Rook's training
								corpus, so no token exists for {vocabMisses === 1 ? 'it' : 'them'} — skipped in its context,
								noted here honestly.
							</p>
						{/if}
						{#if lab.busy !== ''}
							<p class="text-[11px] text-ink-3">
								The same weights are training in another plate right now — you are playing a moving
								target.
							</p>
						{/if}
					</div>
				</div>
			{:else}
				<BootRow />
			{/if}
		</div>
	</Plate>
</div>

<style>
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

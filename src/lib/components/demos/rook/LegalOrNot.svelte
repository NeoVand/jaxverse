<script lang="ts">
	// Not a plate — a figure in the reading column, because it makes one point
	// and the vocabulary plate above makes another. The point: a token carries no
	// permission. All three of these are ordinary members of the vocabulary; the
	// position, and only the position, decides which of them are moves.
	import MiniBoard from './MiniBoard.svelte';

	/** After ⟨game⟩ e2e4 e7e5 — White to move. */
	const POSITION = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';

	const CANDIDATES = [
		{ uci: 'g1f3', legal: true, why: 'the knight leaves the back rank for an empty square' },
		{ uci: 'f1c4', legal: true, why: 'the diagonal opened when the e-pawn stepped out of the way' },
		{ uci: 'e4e5', legal: false, why: 'e5 is occupied — a pawn captures diagonally, never ahead' }
	];
</script>

<figure class="my-8">
	<span class="eyebrow">after ⟨game⟩ e2e4 e7e5 — three tokens Rook could name next</span>
	<div class="mt-3 grid grid-cols-1 gap-x-4 gap-y-5 sm:grid-cols-3">
		{#each CANDIDATES as c (c.uci)}
			<div class="flex min-w-0 flex-col gap-1.5">
				<MiniBoard
					fen={POSITION}
					move={c.uci}
					tone={c.legal ? 'accent' : 'bad'}
					arrow
					showCoordinates
					size={196}
					label="The move {c.uci} is {c.legal ? 'legal' : 'illegal'} in this position: {c.why}"
				/>
				<div class="flex items-baseline gap-2">
					<span class="num text-[12.5px] text-ink">{c.uci}</span>
					<span class="num text-[10.5px]" style="color: {c.legal ? 'var(--good)' : 'var(--bad)'};">
						{c.legal ? '✓ legal here' : '✕ illegal here'}
					</span>
				</div>
				<span class="text-[11.5px] leading-snug text-ink-3">{c.why}</span>
			</div>
		{/each}
	</div>
	<figcaption class="mt-4 text-[13.5px] leading-relaxed text-ink-2 italic">
		Every one of these is a perfectly ordinary token: <span class="num text-[12px] not-italic"
			>e4e5</span
		>
		is a fine move in thousands of other positions, and nothing in the vocabulary marks it as wrong in
		this one. Legality is a property of the board, not of the string — and the only thing on this page
		that knows the difference is chess.js, which pretraining never asks.
	</figcaption>
</figure>

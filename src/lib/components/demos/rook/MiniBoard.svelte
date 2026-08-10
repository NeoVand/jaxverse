<script lang="ts">
	// A chess-book diagram: one small position with a move washed in a tone —
	// and, when it is the point of the figure, drawn as an arrow — under a
	// quiet .num caption. Non-interactive by construction: no move input, no
	// animation, so a strip of them can be repainted as fast as it is sampled.
	import ChessBoard, { type MoveArrow, type SquareMark } from './ChessBoard.svelte';
	import { square, type BoardTone } from './board';

	interface Props {
		fen: string;
		/** UCI of the highlighted move — its from/to squares get the wash. */
		move?: string;
		/** accent = a played move, warm = a capture, bad = an illegal attempt. */
		tone?: BoardTone;
		caption?: string;
		/** Board edge in px. Shrinks with a narrow column; never grows past this. */
		size?: number;
		/** Draw the move as an arrow as well — for a board large enough to BE the
		 * figure, where the wash alone leaves the direction ambiguous. */
		arrow?: boolean;
		showCoordinates?: boolean;
		/** What a screen reader should hear. Defaults to the caption. */
		label?: string;
	}

	let {
		fen,
		move = '',
		tone = 'accent',
		caption,
		size = 148,
		arrow = false,
		showCoordinates = false,
		label
	}: Props = $props();

	const from = $derived(square(move, 'from'));
	const to = $derived(square(move, 'to'));

	const marks = $derived.by(() => {
		const out: SquareMark[] = [];
		if (from) out.push({ square: from, kind: 'wash', tone });
		if (to) out.push({ square: to, kind: 'wash', tone });
		return out;
	});

	const arrows = $derived.by((): MoveArrow[] => (arrow && from && to ? [{ from, to, tone }] : []));

	const captionColor = $derived(
		tone === 'bad'
			? 'var(--bad)'
			: tone === 'warm'
				? 'var(--warm)'
				: tone === 'good'
					? 'var(--good)'
					: 'var(--ink-3)'
	);
</script>

<div style="width: {size}px; max-width: 100%;">
	<ChessBoard
		{fen}
		{marks}
		{arrows}
		{showCoordinates}
		label={label ?? caption ?? 'chess position'}
	/>
	{#if caption}
		<p class="num mt-1 text-center text-[10px] leading-tight" style="color: {captionColor};">
			{caption}
		</p>
	{/if}
</div>

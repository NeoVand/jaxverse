<script lang="ts">
	// A chess-book diagram: one small non-interactive position with the last
	// (or attempted) move washed in a tone, and a quiet .num caption beneath.
	// Rendered straight from FEN — no chess.js needed to display.
	import { PIECE_GLYPH } from './chess-eval';

	interface Props {
		fen: string;
		/** UCI of the highlighted move — its from/to squares get the wash. */
		move?: string;
		/** accent = a played move, warm = a capture, bad = an illegal attempt. */
		tone?: 'accent' | 'warm' | 'bad';
		caption?: string;
		/** Board edge in px. */
		size?: number;
	}

	let { fen, move = '', tone = 'accent', caption, size = 148 }: Props = $props();

	const cells = $derived.by(() => {
		// FEN placement is rank 8 → rank 1, files a → h: already display order
		const placement = fen.split(' ')[0] ?? '';
		const out: Array<{ glyph: string; white: boolean } | null> = [];
		for (const ch of placement) {
			if (ch === '/') continue;
			if (ch >= '1' && ch <= '8') {
				for (let k = 0; k < Number(ch); k++) out.push(null);
			} else {
				out.push({ glyph: PIECE_GLYPH[ch.toLowerCase()] ?? '', white: ch === ch.toUpperCase() });
			}
		}
		return out.slice(0, 64);
	});

	/** Display index (0 = a8) of a square name, or -1. */
	function idxOf(sq: string): number {
		if (sq.length < 2) return -1;
		const col = sq.charCodeAt(0) - 97;
		const rank = Number(sq[1]);
		if (col < 0 || col > 7 || !(rank >= 1 && rank <= 8)) return -1;
		return (8 - rank) * 8 + col;
	}

	const fromIdx = $derived(idxOf(move.slice(0, 2)));
	const toIdx = $derived(idxOf(move.slice(2, 4)));
	const toneColor = $derived(
		tone === 'warm' ? 'var(--warm)' : tone === 'bad' ? 'var(--bad)' : 'var(--accent)'
	);
	const glyphPx = $derived((size / 8) * 0.74);
</script>

<div style="width: {size}px">
	<div
		class="grid grid-cols-8 overflow-hidden rounded-[4px] border border-line select-none"
		role="img"
		aria-label={caption ?? 'chess position'}
	>
		{#each cells as c, i (i)}
			<span
				class="relative flex aspect-square items-center justify-center"
				style="background: {(Math.floor(i / 8) + (i % 8)) % 2 === 1
					? 'var(--surface-2)'
					: 'var(--surface)'};"
			>
				{#if i === fromIdx || i === toIdx}
					<span
						class="absolute inset-0"
						style="background: color-mix(in srgb, {toneColor} 22%, transparent);"
						aria-hidden="true"
					></span>
				{/if}
				{#if c}
					<span
						class="piece relative"
						class:piece-w={c.white}
						class:piece-b={!c.white}
						style="font-size: {glyphPx.toFixed(1)}px;">{c.glyph}</span
					>
				{/if}
			</span>
		{/each}
	</div>
	{#if caption}
		<p
			class="num mt-1 text-center text-[10px] leading-tight"
			style="color: {tone === 'bad'
				? 'var(--bad)'
				: tone === 'warm'
					? 'var(--warm)'
					: 'var(--ink-3)'};"
		>
			{caption}
		</p>
	{/if}
</div>

<style>
	/* Theme-INDEPENDENT piece colors: chess sets don't change sides when the
	   room goes dark. Both colors use filled glyphs; a thin opposing outline
	   keeps each side legible on both square shades in both themes. */
	.piece {
		line-height: 1;
	}
	.piece-w {
		color: #f2efe6;
		-webkit-text-stroke: 0.8px #35332b;
	}
	.piece-b {
		color: #1b1a17;
		-webkit-text-stroke: 0.8px #d8d4c6;
	}
</style>

<script module lang="ts">
	export interface BoardArrow {
		from: string;
		to: string;
		color: string;
	}
</script>

<script lang="ts">
	// The chapter's interactive chess board, extracted from Plate II so the
	// arena (Plate V) can reuse it. chess.js isn't reactive — every derivation
	// keys off the `fen` mirror the owner maintains. The `arrows` overlay is
	// how the arena draws each stage's chosen move on one shared position;
	// coinciding arrows are fanned apart so agreement stays visible.
	import { SvelteMap } from 'svelte/reactivity';
	import type { Square } from 'chess.js';
	import { PIECE_GLYPH, type ChessGame } from './chess-eval';

	interface Props {
		chess: ChessGame | null;
		fen: string;
		selected?: Square | null;
		lastMove?: { from: string; to: string } | null;
		hoverSq?: { from: string; to: string } | null;
		arrows?: BoardArrow[];
		onTap?: (sq: string) => void;
		ariaLabel?: string;
	}

	const {
		chess,
		fen,
		selected = null,
		lastMove = null,
		hoverSq = null,
		arrows = [],
		onTap,
		ariaLabel = 'Chess board — you play White, at the bottom'
	}: Props = $props();

	const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

	// ── board-facing derivations — row 0 renders on top and is rank 8, so
	// White (ranks 1–2) sits at the bottom of the screen ──
	const cells = $derived.by(() => {
		void fen;
		if (!chess) return [];
		const rows = chess.board(); // row 0 = rank 8
		const out: Array<{ sq: string; glyph: string | null; white: boolean; dark: boolean }> = [];
		for (let r = 0; r < 8; r++) {
			for (let f = 0; f < 8; f++) {
				const p = rows[r][f];
				out.push({
					sq: FILES[f] + (8 - r),
					glyph: p ? (PIECE_GLYPH[p.type] ?? null) : null,
					white: p?.color === 'w',
					dark: (r + f) % 2 === 1
				});
			}
		}
		return out;
	});

	/** Legal destinations of the selected piece → whether each is a capture. */
	const targets = $derived.by(() => {
		void fen;
		const map = new SvelteMap<string, boolean>();
		if (!chess || !selected) return map;
		for (const m of chess.moves({ square: selected, verbose: true })) {
			map.set(m.to, m.captured !== undefined);
		}
		return map;
	});

	const checkSq = $derived.by(() => {
		void fen;
		if (!chess || !chess.inCheck()) return null;
		const turn = chess.turn();
		for (const row of chess.board()) {
			for (const p of row) if (p && p.type === 'k' && p.color === turn) return p.square as string;
		}
		return null;
	});

	// ── arrow geometry, in an 8×8 viewBox laid over the grid ──
	function center(sq: string): { x: number; y: number } {
		return { x: FILES.indexOf(sq[0]) + 0.5, y: 8 - Number(sq[1]) + 0.5 };
	}

	const drawnArrows = $derived.by(() => {
		// fan arrows that share the same from→to, so agreement reads as a
		// bundle instead of one arrow hiding the others
		const seen: Record<string, number> = {};
		return arrows.map((a) => {
			const key = a.from + a.to;
			const n = seen[key] ?? 0;
			seen[key] = n + 1;
			const A = center(a.from);
			const B = center(a.to);
			const dx = B.x - A.x;
			const dy = B.y - A.y;
			const len = Math.hypot(dx, dy) || 1;
			const ux = dx / len;
			const uy = dy / len;
			// perpendicular offset: 0, +0.16, −0.16, +0.32, …
			const off = (n % 2 === 1 ? 1 : -1) * Math.ceil(n / 2) * 0.16;
			const px = -uy * off;
			const py = ux * off;
			const headLen = 0.34;
			const headW = 0.17;
			// shaft stops where the head begins; tip pulls up short of center
			const tipX = B.x + px - ux * 0.18;
			const tipY = B.y + py - uy * 0.18;
			const baseX = tipX - ux * headLen;
			const baseY = tipY - uy * headLen;
			return {
				color: a.color,
				x1: A.x + px + ux * 0.3,
				y1: A.y + py + uy * 0.3,
				x2: baseX,
				y2: baseY,
				head: `${tipX},${tipY} ${baseX - uy * headW},${baseY + ux * headW} ${baseX + uy * headW},${baseY - ux * headW}`
			};
		});
	});
</script>

<div class="board-wrap relative select-none">
	<div
		class="board overflow-hidden rounded-md border border-line"
		role="grid"
		aria-label={ariaLabel}
	>
		{#each cells as c (c.sq)}
			<button
				type="button"
				class="relative aspect-square"
				style="background: {c.dark ? 'var(--surface-2)' : 'var(--surface)'};"
				onclick={() => onTap?.(c.sq)}
				aria-label="{c.sq}{c.glyph ? ` ${c.white ? 'white' : 'black'} ${c.glyph}` : ''}"
			>
				{#if lastMove && (lastMove.from === c.sq || lastMove.to === c.sq)}
					<span
						class="absolute inset-0"
						style="background: color-mix(in srgb, var(--accent) 12%, transparent);"
						aria-hidden="true"
					></span>
				{/if}
				{#if checkSq === c.sq}
					<span
						class="absolute inset-0"
						style="background: color-mix(in srgb, var(--bad) 18%, transparent);"
						aria-hidden="true"
					></span>
				{/if}
				{#if hoverSq && (hoverSq.from === c.sq || hoverSq.to === c.sq)}
					<span
						class="absolute inset-0"
						style="background: color-mix(in srgb, var(--accent) 16%, transparent); box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--accent) 55%, transparent);"
						aria-hidden="true"
					></span>
				{/if}
				{#if c.sq[0] === 'a'}
					<span class="coord num top-[3%] left-[5%]">{c.sq[1]}</span>
				{/if}
				{#if c.sq[1] === '1'}
					<span class="coord num right-[5%] bottom-[2%]">{c.sq[0]}</span>
				{/if}
				{#if c.glyph}
					<span
						class="piece absolute inset-0 flex items-center justify-center"
						class:piece-w={c.white}
						class:piece-b={!c.white}>{c.glyph}</span
					>
				{/if}
				{#if targets.has(c.sq)}
					{#if targets.get(c.sq)}
						<span
							class="absolute rounded-full"
							style="inset: 7%; border: 2px solid color-mix(in srgb, var(--accent) 55%, transparent);"
							aria-hidden="true"
						></span>
					{:else}
						<span
							class="absolute rounded-full"
							style="inset: 38%; background: color-mix(in srgb, var(--accent) 45%, transparent);"
							aria-hidden="true"
						></span>
					{/if}
				{/if}
				{#if selected === c.sq}
					<span
						class="absolute inset-0"
						style="background: color-mix(in srgb, var(--accent) 14%, transparent); box-shadow: inset 0 0 0 2px var(--accent);"
						aria-hidden="true"
					></span>
				{/if}
			</button>
		{/each}
	</div>
	{#if drawnArrows.length > 0}
		<svg
			class="pointer-events-none absolute inset-0 h-full w-full"
			viewBox="0 0 8 8"
			aria-hidden="true"
		>
			{#each drawnArrows as a, i (i)}
				<line
					x1={a.x1}
					y1={a.y1}
					x2={a.x2}
					y2={a.y2}
					stroke={a.color}
					stroke-width="0.16"
					stroke-linecap="round"
					opacity="0.85"
				/>
				<polygon points={a.head} fill={a.color} opacity="0.85" />
			{/each}
		</svg>
	{/if}
</div>

<style>
	.board {
		display: grid;
		grid-template-columns: repeat(8, 1fr);
	}
	/* Theme-INDEPENDENT piece colors: a chess set doesn't change sides when the
	   room goes dark. Both colors use filled glyphs; a thin opposing outline
	   keeps each side legible on both square shades in both themes. */
	.piece {
		font-size: clamp(19px, 5.4vw, 31px);
		line-height: 1;
	}
	.piece-w {
		color: #f2efe6;
		-webkit-text-stroke: 1px #35332b;
	}
	.piece-b {
		color: #1b1a17;
		-webkit-text-stroke: 1px #d8d4c6;
	}
	.coord {
		position: absolute;
		font-size: 8.5px;
		color: var(--ink-3);
		pointer-events: none;
	}
</style>

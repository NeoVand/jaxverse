<script lang="ts">
	// The board with nothing on it but its own vocabulary: sixty-four squares
	// wearing their own names. This is the key to every other board in the
	// chapter — `e2e4` means nothing until you can see where e2 and e4 are.
	//
	// Drawn by hand rather than through cm-chessboard, which puts pieces ON
	// squares and has no notion of writing IN them. The wood comes from the same
	// two tokens, so this and the playable boards read as one set.
	import type { BoardTone } from './board';

	interface Props {
		/** The move to light: its from-square, then its to-square. */
		from?: string;
		to?: string;
		/** Board edge in px. Shrinks with a narrow column; never grows past this. */
		size?: number;
	}

	let { from = '', to = '', size = 320 }: Props = $props();

	const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

	// row 0 renders on top and is rank 8, so rank 1 sits at the bottom
	const cells = FILES.flatMap((_, r) =>
		FILES.map((f, c) => ({
			name: `${FILES[c]}${8 - r}`,
			x: c,
			y: r,
			dark: (r + c) % 2 === 1
		}))
	);

	const toneOf = (name: string): BoardTone | null =>
		name === from ? 'accent' : name === to ? 'warm' : null;

	const colorFor = (t: BoardTone) => (t === 'warm' ? 'var(--warm)' : 'var(--accent)');

	function centre(sq: string): { x: number; y: number } {
		return { x: FILES.indexOf(sq[0]) + 0.5, y: 8 - Number(sq[1]) + 0.5 };
	}

	// the same arrow the diagram boards draw, so one move looks like one move
	// wherever the chapter shows it
	const shaft = $derived.by(() => {
		if (!FILES.includes(from[0]) || !FILES.includes(to[0])) return null;
		const a = centre(from);
		const b = centre(to);
		const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
		const ux = (b.x - a.x) / len;
		const uy = (b.y - a.y) / len;
		const headLen = Math.min(0.34, len * 0.3);
		const headW = Math.min(0.17, len * 0.15);
		const tip = Math.min(0.34, len * 0.3);
		const tipX = b.x - ux * tip;
		const tipY = b.y - uy * tip;
		const baseX = tipX - ux * headLen;
		const baseY = tipY - uy * headLen;
		const start = Math.min(0.34, len * 0.28);
		return {
			x1: a.x + ux * start,
			y1: a.y + uy * start,
			x2: baseX,
			y2: baseY,
			head: `${tipX},${tipY} ${baseX - uy * headW},${baseY + ux * headW} ${baseX + uy * headW},${baseY - ux * headW}`
		};
	});
</script>

<svg
	viewBox="0 0 8 8"
	class="board"
	style="width: {size}px;"
	role="img"
	aria-label="An empty chessboard with every square labelled by its name: files a to h from left to right, ranks 1 to 8 from bottom to top. The square {from} and the square {to} are highlighted."
>
	{#each cells as c (c.name)}
		{@const tone = toneOf(c.name)}
		<rect
			x={c.x}
			y={c.y}
			width="1"
			height="1"
			fill={c.dark ? 'var(--sq-dark)' : 'var(--sq-light)'}
		/>
		{#if tone}
			<rect x={c.x} y={c.y} width="1" height="1" fill={colorFor(tone)} opacity="var(--sq-wash)" />
		{/if}
		<text
			x={c.x + 0.5}
			y={c.y + 0.5}
			text-anchor="middle"
			dominant-baseline="central"
			class="name"
			class:lit={tone !== null}
			style={tone ? `fill: ${colorFor(tone)};` : ''}
		>
			{c.name}
		</text>
	{/each}
	{#if shaft}
		<line
			x1={shaft.x1}
			y1={shaft.y1}
			x2={shaft.x2}
			y2={shaft.y2}
			stroke="var(--accent)"
			stroke-width="0.09"
			stroke-linecap="round"
			opacity="0.75"
		/>
		<polygon points={shaft.head} fill="var(--accent)" opacity="0.75" />
	{/if}
</svg>

<style>
	.board {
		display: block;
		max-width: 100%;
		border-radius: var(--r-1);
		overflow: hidden;
	}
	.name {
		font-family: var(--font-mono);
		font-size: 0.26px;
		fill: var(--ink);
		opacity: 0.62;
		user-select: none;
	}
	/* the two squares the token above is made of, said twice: wash and weight */
	.lit {
		opacity: 1;
		font-weight: 600;
	}
</style>

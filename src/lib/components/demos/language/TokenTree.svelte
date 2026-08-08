<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';
	// How two words fare under the shipped 300 merges — real history, read off
	// static/data/text-merges.json (the same snapshot bpe.test.ts guards).
	// "␣little" is common in this corpus and assembles into one token by merge
	// №124; "␣wagged" is rare and is still four pieces when the merging stops.
	// Drawn bottom-up like a parse tree: characters on the floor, each fusion
	// one level higher, labelled with the merge that elected it.

	interface TNode {
		text: string;
		merge: number;
		/** leaf span [first, last], for horizontal placement */
		span: [number, number];
		depth: number; // 1 = fuses leaves, higher = fuses earlier fusions
		kids: Array<{ x: number; y: number }>;
		final?: boolean;
	}

	const SLOT = 33;
	const LEAF_Y = 152;
	const rowY = (depth: number) => LEAF_Y - 14 - depth * 27;
	const leafX = (x0: number, i: number) => x0 + i * SLOT;
	const mid = (x0: number, span: [number, number]) => x0 + ((span[0] + span[1]) / 2) * SLOT;

	function build(
		x0: number,
		nodes: Array<Omit<TNode, 'kids'> & { kids: Array<[number] | [number, number]> }>
	): TNode[] {
		// a kid is [leafIndex] (a floor character) or [span-mid leaf, depth] of an
		// earlier fusion; both become a point the connector drops to
		return nodes.map((n) => ({
			...n,
			kids: n.kids.map((k) =>
				k.length === 1
					? { x: leafX(x0, k[0]), y: LEAF_Y - 9 }
					: { x: x0 + k[0] * SLOT, y: rowY(k[1]) + 9 }
			)
		}));
	}

	// ── ␣little: six fusions, one token ──
	const L0 = 34;
	const LITTLE = [...'␣little'];
	const littleNodes = build(L0, [
		{ text: '␣l', merge: 20, span: [0, 1], depth: 1, kids: [[0], [1]] },
		{ text: 'it', merge: 17, span: [2, 3], depth: 1, kids: [[2], [3]] },
		{ text: 'le', merge: 45, span: [5, 6], depth: 1, kids: [[5], [6]] },
		{ text: 'itt', merge: 108, span: [2, 4], depth: 2, kids: [[2.5, 1], [4]] },
		{
			text: 'ittle',
			merge: 118,
			span: [2, 6],
			depth: 3,
			kids: [
				[3, 2],
				[5.5, 1]
			]
		},
		{
			text: '␣little',
			merge: 124,
			span: [0, 6],
			depth: 4,
			kids: [
				[0.5, 1],
				[4, 3]
			],
			final: true
		}
	]);

	// ── ␣wagged: three fusions, then nothing — four pieces remain ──
	const W0 = 356;
	const WAGGED = [...'␣wagged'];
	const waggedNodes = build(W0, [
		{ text: '␣w', merge: 5, span: [0, 1], depth: 1, kids: [[0], [1]] },
		{ text: 'ed', merge: 8, span: [5, 6], depth: 1, kids: [[5], [6]] },
		{ text: '␣wa', merge: 14, span: [0, 2], depth: 2, kids: [[0.5, 1], [2]], final: true }
	]);
	// the pieces the word is left in: ␣wa and ed (highlighted pills above) plus
	// two bare g's that never found a partner — circled on the floor
	const WAGGED_LOOSE = [3, 4];

	const pillW = (text: string) => text.length * 6.6 + 12;
</script>

<Plate
	id="tokentree"
	title="Two words, as the tokenizer leaves them"
	caption="Two words, as this book's actual tokenizer leaves them. Every pill is a merge the corpus voted for, numbered in the order it was elected; the tinted pieces are the tokens the model will actually read. Frequency decides everything — &quot;little&quot; is everywhere in children's stories, &quot;wagged&quot; is not."
>
	<svg
		viewBox="0 0 620 178"
		class="mx-auto block w-full max-w-[850px]"
		role="img"
		aria-label="Two byte-pair merge trees from this book's actual tokenizer. The common word ' little' fuses, merge by merge, into a single token by merge 124. The rare word ' wagged' has only three of its pairs elected in 300 merges and remains four pieces: ' wa', 'g', 'g', 'ed'."
	>
		<!-- ── the common word: one token by merge №124 ── -->
		{#each LITTLE as c, i (i)}
			<text x={leafX(L0, i)} y={LEAF_Y} text-anchor="middle" class="leaf" class:sp={c === '␣'}
				>{c}</text
			>
		{/each}
		{#each littleNodes as n (n.text)}
			{@const x = mid(L0, n.span)}
			{@const y = rowY(n.depth)}
			{#each n.kids as k, ki (ki)}
				<path
					d="M {x} {y + 9} C {x} {y + 20}, {k.x} {y + 12}, {k.x} {k.y}"
					fill="none"
					stroke="var(--line)"
					stroke-width="1"
				/>
			{/each}
			<rect
				x={x - pillW(n.text) / 2}
				y={y - 9}
				width={pillW(n.text)}
				height="18"
				rx="9"
				fill={n.final ? 'var(--accent-soft)' : 'var(--surface)'}
				stroke={n.final ? 'var(--accent)' : 'var(--line)'}
			/>
			<text {x} y={y + 3.5} text-anchor="middle" class="piece" class:piece-final={n.final}
				>{n.text}</text
			>
			<text x={x + pillW(n.text) / 2 + 4} y={y + 3} class="mno">№{n.merge}</text>
		{/each}
		<text x={mid(L0, [0, 6])} y="174" text-anchor="middle" class="cap">
			common — one token after 124 merges
		</text>

		<!-- the fold -->
		<line x1="310" y1="14" x2="310" y2="166" stroke="var(--line-soft)" stroke-width="1" />

		<!-- ── the rare word: still in pieces when the merging stops ── -->
		{#each WAGGED as c, i (i)}
			<text x={leafX(W0, i)} y={LEAF_Y} text-anchor="middle" class="leaf" class:sp={c === '␣'}
				>{c}</text
			>
		{/each}
		{#each waggedNodes as n (n.text)}
			{@const x = mid(W0, n.span)}
			{@const y = rowY(n.depth)}
			{#each n.kids as k, ki (ki)}
				<path
					d="M {x} {y + 9} C {x} {y + 20}, {k.x} {y + 12}, {k.x} {k.y}"
					fill="none"
					stroke="var(--line)"
					stroke-width="1"
				/>
			{/each}
			<rect
				x={x - pillW(n.text) / 2}
				y={y - 9}
				width={pillW(n.text)}
				height="18"
				rx="9"
				fill={n.final || n.text === 'ed' ? 'var(--accent-soft)' : 'var(--surface)'}
				stroke={n.final || n.text === 'ed' ? 'var(--accent)' : 'var(--line)'}
			/>
			<text
				{x}
				y={y + 3.5}
				text-anchor="middle"
				class="piece"
				class:piece-final={n.final || n.text === 'ed'}>{n.text}</text
			>
			<text x={x + pillW(n.text) / 2 + 4} y={y + 3} class="mno">№{n.merge}</text>
		{/each}
		{#each WAGGED_LOOSE as i (i)}
			<circle
				cx={leafX(W0, i)}
				cy={LEAF_Y - 4.5}
				r="10"
				fill="none"
				stroke="var(--accent)"
				opacity="0.55"
			/>
		{/each}
		<text x={mid(W0, [0, 6])} y="174" text-anchor="middle" class="cap">
			rare — still ␣wa + g + g + ed after 300
		</text>
	</svg>
</Plate>

<style>
	.leaf {
		font-family: var(--font-mono);
		font-size: 11.5px;
		fill: var(--ink);
	}
	.sp {
		fill: var(--ink-3);
	}
	.piece {
		font-family: var(--font-mono);
		font-size: 10px;
		fill: var(--ink-2);
	}
	.piece-final {
		fill: var(--accent);
		font-weight: 600;
	}
	.mno {
		font-family: var(--font-mono);
		font-size: 7.5px;
		fill: var(--warm);
	}
	.cap {
		font-family: var(--font-mono);
		font-size: 8.5px;
		fill: var(--ink-3);
	}
</style>

<script lang="ts">
	// The chapter's vocabulary: what a move token is, where its two halves point,
	// and what the whole language looks like.
	//
	// Three ruled columns, one idea each — a token, the dictionary it comes from,
	// a sentence written in it. The cells stretch to a common height and are
	// separated by hairlines, so the columns read as one figure rather than three
	// blocks that happened to land near each other.
	//
	// The figure answers to a pointer: every token in either table drives the
	// headline anatomy AND the named board, so a reader can walk the dictionary
	// and watch each string resolve into two squares. e2e4 is what it rests on.
	//
	// Legality is deliberately NOT here. It is a different claim, and it gets its
	// own figure further down the page.
	//
	// Every id is read off the shipped vocabulary (static/data/rook-vocab.json),
	// which is sorted, so a1a2 really is token 1 and h8h7 really is token 1,930.
	import Plate from '$lib/components/ui/Plate.svelte';
	import SquareNames from './SquareNames.svelte';

	// ── the dictionary, sampled the whole way down ──
	const LEXICON: Array<{ id: string; tok?: string; note?: string }> = [
		{ id: '0', tok: '⟨game⟩', note: 'the marker' },
		{ id: '1', tok: 'a1a2' },
		{ id: '2', tok: 'a1a3' },
		{ id: '3', tok: 'a1a4' },
		{ id: '⋮' },
		{ id: '222', tok: 'b1c3' },
		{ id: '⋮' },
		{ id: '470', tok: 'c1f4' },
		{ id: '⋮' },
		{ id: '987', tok: 'e1g1', note: 'castling' },
		{ id: '⋮' },
		{ id: '1013', tok: 'e2e4' },
		{ id: '⋮' },
		{ id: '1305', tok: 'f3e5' },
		{ id: '⋮' },
		{ id: '1497', tok: 'g1f3' },
		{ id: '1712', tok: 'g8f6' },
		{ id: '⋮' },
		{ id: '1906', tok: 'h7h8q', note: 'a promotion' },
		{ id: '1929', tok: 'h8h6' },
		{ id: '1930', tok: 'h8h7' }
	];

	// ── a real opening, as Rook reads it: eight plies and the question ──
	const SENTENCE: Array<{ ply: string; tok: string; id: string; ask?: boolean }> = [
		{ ply: '', tok: '⟨game⟩', id: '0' },
		{ ply: '1', tok: 'e2e4', id: '1013' },
		{ ply: '2', tok: 'e7e5', id: '1186' },
		{ ply: '3', tok: 'g1f3', id: '1497' },
		{ ply: '4', tok: 'b8c6', id: '436' },
		{ ply: '5', tok: 'f1b5', id: '1232' },
		{ ply: '6', tok: 'a7a6', id: '161' },
		{ ply: '7', tok: 'b5a4', id: '330' },
		{ ply: '8', tok: 'g8f6', id: '1712' },
		{ ply: '9', tok: '?', id: '—', ask: true }
	];

	/** A four- or five-character move string; ⟨game⟩ and ? are not moves. */
	const isMove = (t: string) => /^[a-h][1-8][a-h][1-8][qrbn]?$/.test(t);

	/** What the reader is pointing at; e2e4 is where the figure rests. */
	let hovered = $state<string | null>(null);
	const token = $derived(hovered ?? 'e2e4');

	// ── the anatomy, laid out in a 300-wide viewBox ──
	// JetBrains Mono advances 0.6em, so a 48px glyph is 28.8px wide. Every
	// bracket and label below is placed off that one number, which is why a
	// five-character token can slot a third segment in without a reflow.
	const ADV = 28.8;
	const x0 = $derived(150 - (token.length * ADV) / 2);
	const segments = $derived.by(() => {
		const out = [
			{ text: token.slice(0, 2), x: x0, w: 2 * ADV, tone: 'var(--accent)', label: 'from square' },
			{
				text: token.slice(2, 4),
				x: x0 + 2 * ADV,
				w: 2 * ADV,
				tone: 'var(--warm)',
				label: 'to square'
			}
		];
		if (token.length > 4) {
			out.push({
				text: token.slice(4),
				x: x0 + 4 * ADV,
				w: ADV,
				tone: 'var(--cat-8)',
				label: 'piece'
			});
		}
		return out;
	});
	const anatomyNote = $derived(
		token.length > 4
			? 'four characters for the squares, a fifth for the new piece'
			: 'no separator, no piece letter, nothing else'
	);

	function point(tok: string): void {
		if (isMove(tok)) hovered = tok;
	}
</script>

<Plate
	id="vocab"
	title="Moves are tokens"
	caption="Rook's entire language: 1,930 move strings and one marker. A token is the square a piece leaves and the square it lands on, glued together — point at any of them and the board says where those squares are, which is the moment e2e4 stops being a code and becomes a move. Nothing here says which moves are playable; that is a different question, and it comes next."
>
	<div class="grid px-4 pt-1 pb-2 lg:grid-cols-3">
		<!-- ── one token, and the squares it points at ── -->
		<section class="col lg:pr-7">
			<div class="flex items-baseline justify-between gap-3">
				<span class="eyebrow">one token, one move</span>
				<span class="num text-[10px] whitespace-nowrap text-ink-3">point at any token</span>
			</div>

			<svg
				viewBox="0 0 300 100"
				class="block w-full max-w-[340px]"
				role="img"
				aria-label="The anatomy of the move token {token}: the square the piece leaves, then the square it lands on."
			>
				<!-- one <text> per segment, each at its own x: the glyphs are monospace
				     and the offsets exact, so the string reads contiguous without
				     relying on whitespace between tspans surviving a formatter -->
				{#each segments as s (s.label)}
					<text x={s.x} y="50" class="tok-big" fill={s.tone}>{s.text}</text>
				{/each}
				{#each segments as s (s.label)}
					<path
						d="M {s.x + 1.5} 60 L {s.x + 1.5} 66 L {s.x + s.w - 1.5} 66 L {s.x + s.w - 1.5} 60"
						fill="none"
						stroke={s.tone}
						stroke-width="1"
						opacity="0.65"
					/>
					<text x={s.x + s.w / 2} y="82" text-anchor="middle" class="cap" fill={s.tone}>
						{s.label}
					</text>
				{/each}
				<text x="150" y="96" text-anchor="middle" class="cap dim">{anatomyNote}</text>
			</svg>

			<SquareNames from={token.slice(0, 2)} to={token.slice(2, 4)} size={340} />

			<p class="mt-auto text-[12px] leading-relaxed text-ink-3">
				Every square carries a name — the letter of its column, then the number of its row. That is
				the whole notation, and the only thing a move token is made of.
			</p>
		</section>

		<!-- ── the dictionary those tokens come from ── -->
		<section class="col lg:border-l lg:px-7">
			<span class="eyebrow">the vocabulary · 1,931 tokens</span>

			<div class="lex num">
				{#each LEXICON as row, i (i)}
					{#if !row.tok}
						<div class="lex-row lex-gap">
							<span></span>
							<span class="lex-more">⋮</span>
						</div>
					{:else if isMove(row.tok)}
						<button
							type="button"
							class="lex-row"
							class:on={row.tok === token}
							onmouseenter={() => point(row.tok ?? '')}
							onmouseleave={() => (hovered = null)}
							onfocus={() => point(row.tok ?? '')}
							onblur={() => (hovered = null)}
						>
							<span class="lex-id">{row.id}</span>
							<span class="tok">{row.tok}</span>
							<span class="lex-note">{row.note ?? ''}</span>
						</button>
					{:else}
						<div class="lex-row">
							<span class="lex-id">{row.id}</span>
							<span>{row.tok}</span>
							<span class="lex-note">{row.note ?? ''}</span>
						</div>
					{/if}
				{/each}
			</div>

			<p class="mt-auto text-[12px] leading-relaxed text-ink-3">
				Sorted alphabetically, so an id carries no meaning of its own — the model has to learn what
				<span class="num text-ink">1013</span> does from the company it keeps.
			</p>
		</section>

		<!-- ── and a sentence written in them ── -->
		<section class="col lg:border-l lg:pl-7">
			<span class="eyebrow">a game is a sentence written in them</span>

			<div class="lex num">
				{#each SENTENCE as s (s.tok + s.ply)}
					{#if isMove(s.tok)}
						<button
							type="button"
							class="ply-row"
							class:on={s.tok === token}
							onmouseenter={() => point(s.tok)}
							onmouseleave={() => (hovered = null)}
							onfocus={() => point(s.tok)}
							onblur={() => (hovered = null)}
						>
							<span class="lex-id">{s.ply}</span>
							<span class="tok">{s.tok}</span>
							<span class="lex-note">{s.id}</span>
						</button>
					{:else}
						<div class="ply-row" class:ask={s.ask}>
							<span class="lex-id">{s.ply}</span>
							<span style={s.ask ? 'color: var(--accent);' : ''}>{s.tok}</span>
							<span class="lex-note">{s.ask ? 'next token' : s.id}</span>
						</div>
					{/if}
				{/each}
			</div>

			<p class="text-[12px] leading-relaxed text-ink-3">
				That column is the <em>whole</em> input: no board, no piece list, no rules. Rook reads the numbers
				and has to name the next one.
			</p>

			<div class="mt-auto flex flex-col gap-1.5 border-t border-line-soft pt-4">
				<span class="eyebrow">one token runs longer</span>
				<p class="text-[12px] leading-relaxed text-ink-3">
					142 of the strings run to five characters, like
					<span class="num text-ink">h7h8q</span>: a pawn reaching the last rank, and the piece it
					turns into. Point at that one and the anatomy grows a third bracket.
				</p>
			</div>
		</section>
	</div>
</Plate>

<style>
	/* One ruled cell per idea. The cells stretch to a common height and the
	   hairlines run their full length, so a column that says less still sits in
	   the same frame as one that says more. */
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

	.tok-big {
		font-family: var(--font-mono);
		font-size: 48px;
		letter-spacing: 0;
	}
	.cap {
		font-family: var(--font-mono);
		font-size: 8.5px;
	}
	.dim {
		fill: var(--ink-3);
	}

	/* the dictionary and the sentence share one table voice: id, token, aside */
	.lex {
		font-size: 11.5px;
		border-top: 1px solid var(--line-soft);
	}
	.lex-row,
	.ply-row {
		display: grid;
		width: 100%;
		align-items: baseline;
		gap: 0.5rem;
		padding: 3px 0;
		border-bottom: 1px solid var(--line-soft);
		color: var(--ink);
		text-align: left;
		cursor: default;
	}
	.lex-row {
		grid-template-columns: 3.25rem 4.5rem minmax(0, 1fr);
	}
	.ply-row {
		grid-template-columns: 1.5rem 4.5rem minmax(0, 1fr);
		padding-block: 4.5px;
	}
	/* pointing is the whole interaction: the ground lights, never the type, so a
	   token keeps its colour whether or not it is the one being read */
	button.lex-row:hover,
	button.ply-row:hover,
	button.lex-row:focus-visible,
	button.ply-row:focus-visible {
		background: color-mix(in srgb, var(--accent) 11%, transparent);
	}
	.on {
		background: color-mix(in srgb, var(--accent) 8%, transparent);
	}
	.on .tok {
		color: var(--accent);
	}
	.ask {
		background: color-mix(in srgb, var(--accent) 7%, transparent);
	}
	.lex-gap {
		padding-block: 0;
	}
	.lex-more {
		color: var(--ink-3);
		line-height: 1.2;
	}
	.lex-id {
		color: var(--ink-3);
		text-align: right;
	}
	.lex-note {
		font-size: 10px;
		color: var(--ink-3);
	}
</style>
